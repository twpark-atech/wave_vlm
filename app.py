# app.py
import os, io, re, uuid, tempfile, json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from ultralytics import YOLO

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import yaml

from prompts import get_vlm_prompt, get_system_prompt, reduce_to_top3


def load_yaml_config() -> Dict[str, Any]:
    # 우선순위: 환경변수 APP_CONFIG → ./config.yml → ./config.yaml
    cand = os.environ.get("APP_CONFIG", "")
    here = Path(__file__).resolve().parent
    paths = [Path(cand)] if cand else []
    paths += [here / "config.yml", here / "config.yaml"]

    cfg: Dict[str, Any] = {}
    for p in paths:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                cfg.update(data)
            break

    # 기본값 보강
    cfg.setdefault("device", "cuda")
    cfg.setdefault("max_side", 896)
    cfg.setdefault("qwen_model", "Qwen/Qwen2-VL-7B-Instruct")
    cfg.setdefault("lang", "ko")
    gen = cfg.get("gen") or {}
    gen.setdefault("max_new_tokens", 1024)
    gen.setdefault("do_sample", True)
    gen.setdefault("temperature", 0.7)
    gen.setdefault("top_p", 0.9)
    gen.setdefault("top_k", 50)
    cfg["gen"] = gen
    return cfg


CONFIG = load_yaml_config()

# ------------------------- 공통 설정 -------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "deepfashion2_yolov8s-seg.pt")
DEVICE = os.environ.get("DEVICE", CONFIG.get("device", "cuda"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs/seg_cutout"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = Path(os.getcwd()) / "main.html"
DATAURL_RE = re.compile(r"^data:(?P<mime>[\w/+.\-]+);base64,(?P<b64>.+)$", re.IGNORECASE)

app = FastAPI(title="Outfit Segmentation + VLM (Qwen2-VL)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 서빙(필요 시)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="outputs")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# ------------------------- HTML 서빙 -------------------------
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(HTML_PATH))

@app.get("/main.html", include_in_schema=False)
def serve_main():
    return FileResponse(str(HTML_PATH))


# ------------------------- 세그 스키마 -------------------------
class SegmentReq(BaseModel):
    image: str
    meta: Optional[Dict[str, Any]] = None

class SegmentRes(BaseModel):
    ok: bool
    url: str
    filename: str
    width: int
    height: int
    mode: str
    detections: int = 0
    overlay_url: Optional[str] = None


# ------------------------- 공용 유틸 -------------------------
def _b64decode(b64str: str) -> bytes:
    import base64
    return base64.b64decode(b64str)

def _b64encode(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")

def pil_to_dataurl(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = _b64encode(buf.getvalue())
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def decode_image_from_dataurl(image_str: str) -> Image.Image:
    s = (image_str or "").strip()
    if not s or s == "data:,":
        raise HTTPException(status_code=400, detail="empty_image: got blank data URL")

    m = DATAURL_RE.match(s)
    b64 = m.group("b64") if m else s

    raw = _b64decode(b64)
    if not raw:
        raise HTTPException(status_code=400, detail="empty_image_bytes")

    try:
        img = Image.open(io.BytesIO(raw))
        return img.convert("RGB")
    except Exception:
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="decode_error: unsupported or corrupted image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

def bgr_to_rgba_with_alpha(orig_bgr: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")

def build_union_alpha(
    masks_bin: List[np.ndarray],
    h: int, w: int,
    dilate_px: int = 1,
    soften_sigma: float = 1.5
) -> np.ndarray:
    if not masks_bin:
        return np.zeros((h, w), dtype=np.uint8)

    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks_bin:
        union = cv2.bitwise_or(union, m)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        union = cv2.dilate(union, k, iterations=1)

    if soften_sigma > 0:
        blurred = cv2.GaussianBlur(union.astype(np.float32), (0, 0), soften_sigma)
        maxi = blurred.max()
        if maxi > 0:
            blurred = blurred * (255.0 / maxi)
        union = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        union = (union > 0).astype(np.uint8) * 255

    return union

def select_masks(result) -> List[np.ndarray]:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None or len(masks_obj.data) == 0:
        return []
    H, W = result.orig_img.shape[:2]
    masks_tf = masks_obj.data  # (N,h,w)
    masks_bin_list = []
    for i in range(masks_tf.shape[0]):
        m_small = masks_tf[i].cpu().numpy()
        m_small = (m_small > 0.5).astype(np.uint8)
        m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        masks_bin_list.append(m)
    return masks_bin_list

def save_cutout_png(img_rgba: Image.Image) -> Path:
    # (유지용) 파일 저장이 필요한 흐름에서만 사용
    stem = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{stem}_cutout.png"
    img_rgba.save(out_path)
    return out_path

def save_overlay_jpg(bgr: np.ndarray, masks: List[np.ndarray]) -> Path:
    # (유지용) 여기서는 실제 파일 저장을 생략
    overlay = bgr.copy()
    for m in masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (60, 200, 255), 2)
    stem = uuid.uuid4().hex[:12]
    p = OUTPUT_DIR / f"{stem}_overlay.jpg"
    # cv2.imwrite(str(p), overlay)  # 저장 비활성화
    return p


# ------------------------- YOLO 모델 (전역 1회) -------------------------
_yolo_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


# ------------------------- 세그 추론 -------------------------
def run_cutout_from_path(
    img_path: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = DEVICE
) -> Tuple[Image.Image, List[np.ndarray], np.ndarray]:
    model = get_model()
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False
    )
    res = results[0]
    orig_bgr = res.orig_img
    H, W = orig_bgr.shape[:2]
    masks = select_masks(res)

    if not masks:
        alpha = np.zeros((H, W), dtype=np.uint8)
    else:
        alpha = build_union_alpha(masks, H, W, dilate_px=1, soften_sigma=1.5)

    cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
    return cutout, masks, orig_bgr

# === NEW: 메모리 기반 세그 (ndarray 입력) ===
def run_cutout_from_array(
    bgr: np.ndarray,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = DEVICE
) -> Tuple[Image.Image, List[np.ndarray], np.ndarray]:
    model = get_model()
    results = model.predict(
        source=bgr,  # ndarray 직접 입력
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False
    )
    res = results[0]
    orig_bgr = res.orig_img
    H, W = orig_bgr.shape[:2]
    masks = select_masks(res)

    if not masks:
        alpha = np.zeros((H, W), dtype=np.uint8)
    else:
        alpha = build_union_alpha(masks, H, W, dilate_px=1, soften_sigma=1.5)

    cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
    return cutout, masks, orig_bgr

# === NEW: VLM 입력용 RGB 생성(흰 배경 합성 + 리사이즈) ===
def cutout_rgba_to_vlm_rgb(cutout_rgba: Image.Image, max_side: int) -> Image.Image:
    assert cutout_rgba.mode == "RGBA"
    bg = Image.new("RGBA", cutout_rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(cutout_rgba)
    rgb = bg.convert("RGB")
    w, h = rgb.size
    ms = max(w, h)
    if ms > max_side:
        scale = max_side / ms
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return rgb


# ------------------------- 세그 API (기존: 파일 저장 흐름) -------------------------
@app.post("/api/v1/segment/json", response_model=SegmentRes)
def segment_json(body: SegmentReq):
    pil_img = decode_image_from_dataurl(body.image)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        pil_img.save(tf.name)
        tmp_path = Path(tf.name)

    try:
        cutout, masks, orig_bgr = run_cutout_from_path(tmp_path, imgsz=640, conf=0.25, device=DEVICE)
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass

    out_path = save_cutout_png(cutout)           # 파일 저장 (기존 유지)
    overlay_path = save_overlay_jpg(orig_bgr, masks)
    w, h = cutout.size
    return SegmentRes(
        ok=True,
        url=f"/outputs/{out_path.name}",
        filename=out_path.name,
        width=w, height=h,
        mode="json_dataurl_or_base64",
        detections=len(masks),
        overlay_url=f"/outputs/{overlay_path.name}",
    )

@app.post("/api/v1/segment/file", response_model=SegmentRes)
async def segment_file(file: UploadFile = File(...)):
    raw = await file.read()
    suffix = Path(file.filename or "img").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(raw)
        tmp_path = Path(tf.name)

    try:
        cutout, masks, orig_bgr = run_cutout_from_path(tmp_path, imgsz=640, conf=0.25, device=DEVICE)
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass

    out_path = save_cutout_png(cutout)           # 파일 저장 (기존 유지)
    overlay_path = save_overlay_jpg(orig_bgr, masks)
    w, h = cutout.size
    return SegmentRes(
        ok=True,
        url=f"/outputs/{out_path.name}",
        filename=out_path.name,
        width=w, height=h,
        mode="multipart_file",
        detections=len(masks),
        overlay_url=f"/outputs/{overlay_path.name}",
    )

# 하위 호환 alias
@app.post("/api/v1/intake/json", response_model=SegmentRes)
def intake_json_alias(body: SegmentReq):
    return segment_json(body)

@app.post("/api/v1/intake/file", response_model=SegmentRes)
async def intake_file_alias(file: UploadFile = File(...)):
    return await segment_file(file)


# ========================= VLM (Qwen2-VL) =========================

# ---- VLM 스키마 ----
class VLMJsonReq(BaseModel):
    cutout_url: Optional[str] = None   # 예: "/outputs/xxxx_cutout.png"
    image: Optional[str] = None        # dataURL(base64)
    meta: Optional[Dict[str, Any]] = None

class VLMRes(BaseModel):
    ok: bool
    used_image_url: Optional[str] = None
    raw: str
    result: Dict[str, Any]
    model_id: str
    gen: Dict[str, Any]


# ---- VLM 유틸 ----
def map_url_to_path(url: str) -> Path:
    name = Path(url).name
    return OUTPUT_DIR / name

def load_cutout_image(path: Path, max_side: int) -> Image.Image:
    # RGBA cutout → 흰 배경 합성 후 RGB + 리사이즈
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    img = bg.convert("RGB")

    w, h = img.size
    ms = max(w, h)
    if ms > max_side:
        scale = max_side / ms
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def parse_json_safely(text: str) -> Dict[str, Any]:
    import re as _re
    s = (text or "").strip()

    # 코드블록 제거
    if s.startswith("```"):
        s = _re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = s.rstrip("`").rstrip()
        s = s.replace("```", "")

    # 바깥 중괄호만 추출
    b, e = s.find("{"), s.rfind("}")
    if b >= 0 and e > b:
        s = s[b:e+1]

    # 흔한 오류 정정: 작은따옴표, True/False/None
    s = _re.sub(r"(?<!\\)\'", '"', s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = _re.sub(r"\bTrue\b", "true", s)
    s = _re.sub(r"\bFalse\b", "false", s)
    s = _re.sub(r"\bNone\b", "null", s)

    try:
        return json.loads(s)
    except Exception:
        return {"raw": text, "_error": "json_parse_failed"}

# ---- 정렬만(정규화 없음) ----
def _safe_prob(x) -> float:
    try:
        if isinstance(x, str):
            s = x.strip().lower().replace(" ", "")
            return float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
        return float(x)
    except Exception:
        return 0.0

def sort_candidates_only(parsed: dict) -> dict:
    """
    정규화 없이, candidates를 확률 내림차순으로 정렬하고
    최상위 항목을 top_prediction으로 올린 뒤 candidates에서 제거.
    """
    raw_cands = parsed.get("candidates") or []
    items = []
    for it in raw_cands:
        lab = it.get("job", "")
        prob = _safe_prob(it.get("prob", 0))
        items.append((lab, prob))

    if not items:
        top = parsed.get("top_prediction") or {"job": "", "confidence": 0.0}
        return {
            "top_prediction": {"job": top.get("job", ""), "confidence": float(_safe_prob(top.get("confidence", 0)))},
            "candidates": [],
            "evidence": parsed.get("evidence", []),
        }

    items.sort(key=lambda x: x[1], reverse=True)
    top_label, top_prob = items[0]
    rest = [{"job": lab, "prob": prob} for lab, prob in items[1:]]

    return {
        "top_prediction": {"job": top_label, "confidence": float(top_prob)},
        "candidates": rest,
        "evidence": parsed.get("evidence", []),
    }


# ---- VLM 모델 로더(싱글톤) ----
_vlm_model = None
_vlm_processor = None
_vlm_dtype = None

def get_qwen():
    global _vlm_model, _vlm_processor, _vlm_dtype
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor, _vlm_dtype

    model_id = CONFIG["qwen_model"]
    device_pref = CONFIG.get("device", "cuda").lower()

    if device_pref == "cuda" and torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = "cpu"

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    mdl = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    mdl.eval()

    _vlm_model, _vlm_processor, _vlm_dtype = mdl, proc, torch_dtype
    return _vlm_model, _vlm_processor, _vlm_dtype


@torch.inference_mode()
def qwen_infer(image: Image.Image, prompt: str, gen_cfg: Dict[str, Any], lang: str = "ko") -> str:
    mdl, proc, _ = get_qwen()

    system_text = get_system_prompt(lang)  # 한국어 고정 안내 등
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]

    chat_text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[chat_text], images=[image], return_tensors="pt").to(mdl.device)
    in_len = inputs["input_ids"].shape[-1]

    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", 1024),
        do_sample=gen_cfg.get("do_sample", True),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        top_k=gen_cfg.get("top_k", 50),
        pad_token_id=mdl.generation_config.pad_token_id,
        eos_token_id=mdl.generation_config.eos_token_id,
    )
    gen_only = out_ids[:, in_len:]
    raw_text = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
    return raw_text


# ------------------------- VLM API (기존: 파일 경로/업로드) -------------------------
@app.post("/api/v1/vlm/json", response_model=VLMRes)
def vlm_json(body: VLMJsonReq):
    cfg = CONFIG
    max_side = int(cfg.get("max_side", 896))
    gen_cfg = dict(cfg.get("gen", {}))
    lang = cfg.get("lang", "ko")

    used_image_url: Optional[str] = None

    if body.cutout_url:
        p = map_url_to_path(body.cutout_url)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"cutout not found: {body.cutout_url}")
        used_image_url = f"/outputs/{p.name}"
        image = load_cutout_image(p, max_side=max_side)

    elif body.image:
        # 업로드 이미지를 파일로 보존한 다음 로드(기존 흐름 유지)
        pil_img = decode_image_from_dataurl(body.image)  # RGB
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            pil_img.save(tf.name)
            tmp = Path(tf.name)
        out_copy = OUTPUT_DIR / f"{uuid.uuid4().hex[:12]}_uploaded.png"
        tmp.replace(out_copy)
        used_image_url = f"/outputs/{out_copy.name}"
        image = load_cutout_image(out_copy, max_side=max_side)

    else:
        raise HTTPException(status_code=400, detail="provide cutout_url or image(dataURL)")

    prompt = get_vlm_prompt(lang)
    raw = qwen_infer(image, prompt, gen_cfg, lang=lang)
    parsed = parse_json_safely(raw)
    parsed = reduce_to_top3(parsed)
    result_sorted = sort_candidates_only(parsed)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
    )

@app.post("/api/v1/vlm/file", response_model=VLMRes)
async def vlm_file(file: UploadFile = File(...)):
    cfg = CONFIG
    max_side = int(cfg.get("max_side", 896))
    gen_cfg = dict(cfg.get("gen", {}))
    lang = cfg.get("lang", "ko")

    raw = await file.read()
    suffix = Path(file.filename or "img").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(raw)
        tmp = Path(tf.name)

    out_copy = OUTPUT_DIR / f"{uuid.uuid4().hex[:12]}_uploaded{suffix if suffix else '.png'}"
    tmp.replace(out_copy)
    used_image_url = f"/outputs/{out_copy.name}"

    image = load_cutout_image(out_copy, max_side=max_side)
    prompt = get_vlm_prompt(lang)
    raw_text = qwen_infer(image, prompt, gen_cfg, lang=lang)
    parsed = parse_json_safely(raw_text)
    parsed = reduce_to_top3(parsed)
    result_sorted = sort_candidates_only(parsed)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw_text,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
    )


# ========================= 메모리 전용(파일 저장 없음) =========================
class SegmentInlineRes(BaseModel):
    ok: bool
    cutout_dataurl: str
    detections: int

class InlineReq(BaseModel):
    image: str
    meta: Optional[Dict[str, Any]] = None

@app.post("/api/v1/segment/inline", response_model=SegmentInlineRes)
def segment_inline(body: SegmentReq):
    """dataURL → cutout dataURL (파일 저장 없음)"""
    pil = decode_image_from_dataurl(body.image)           # RGB
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cutout_rgba, masks, _ = run_cutout_from_array(bgr, imgsz=640, conf=0.25, device=DEVICE)
    cutout_dataurl = pil_to_dataurl(cutout_rgba, fmt="PNG")
    return SegmentInlineRes(ok=True, cutout_dataurl=cutout_dataurl, detections=len(masks))

@app.post("/api/v1/predict/inline", response_model=VLMRes)
def predict_inline(body: InlineReq):
    """
    dataURL → (세그 RGBA cutout) → 흰 배경 합성 RGB → Qwen2-VL
    전 과정 메모리에서 처리. 파일 저장 없음.
    """
    cfg = CONFIG
    max_side = int(cfg.get("max_side", 896))
    gen_cfg = dict(cfg.get("gen", {}))
    lang = cfg.get("lang", "ko")

    # 1) dataURL → PIL → BGR
    pil_rgb = decode_image_from_dataurl(body.image)
    bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)

    # 2) 메모리 기반 세그
    cutout_rgba, masks, _ = run_cutout_from_array(bgr, imgsz=640, conf=0.25, device=DEVICE)

    # 3) VLM 입력용 RGB(흰 배경 합성 + 리사이즈)
    vlm_rgb = cutout_rgba_to_vlm_rgb(cutout_rgba, max_side=max_side)

    # 4) VLM
    prompt = get_vlm_prompt(lang)
    raw_text = qwen_infer(vlm_rgb, prompt, gen_cfg, lang=lang)
    parsed = parse_json_safely(raw_text)
    parsed = reduce_to_top3(parsed)
    result_sorted = sort_candidates_only(parsed)

    return VLMRes(
        ok=True,
        used_image_url=None,   # 파일 사용 X
        raw=raw_text,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
    )
