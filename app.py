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

# ==== VLM(Qwen2-VL) ====
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ==== YAML CONFIG ====
import yaml

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
    gen.setdefault("max_new_tokens", 256)
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

# 정적 서빙: 세그/오버레이 결과
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="outputs")


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
    stem = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{stem}_cutout.png"
    img_rgba.save(out_path)
    return out_path

def save_overlay_jpg(bgr: np.ndarray, masks: List[np.ndarray]) -> Path:
    overlay = bgr.copy()
    for m in masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (60, 200, 255), 2)
    stem = uuid.uuid4().hex[:12]
    p = OUTPUT_DIR / f"{stem}_overlay.jpg"
    cv2.imwrite(str(p), overlay)
    return p


# ------------------------- YOLO 모델 (전역 1회) -------------------------
_yolo_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


# ------------------------- 세그 추론(파일 경로 기반) -------------------------
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


# ------------------------- 세그 API -------------------------
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

    out_path = save_cutout_png(cutout)
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

    out_path = save_cutout_png(cutout)
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
    # RGBA cutout → 흰 배경으로 합성하여 RGB로 변환 후 리사이즈
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

def build_prompt(lang: str = "ko") -> str:
    if lang == "ko":
        return (
            "아래 이미지는 배경이 제거된 사람의 의상(누끼)입니다.\n"
            "의상 스타일과 소지품 단서를 바탕으로 예상 직업을 추정해 주세요.\n"
            "반드시 JSON으로만 응답하세요 (설명 문장 금지).\n"
            "{\n"
            '  "top_prediction": {"job": "직업명", "confidence": 0.0~1.0},\n'
            '  "candidates": [\n'
            '    {"job": "사무/개발", "prob": 0.0~1.0},\n'
            '    {"job": "영업/사무", "prob": 0.0~1.0},\n'
            '    {"job": "학생/연구", "prob": 0.0~1.0},\n'
            '    {"job": "체육/레저", "prob": 0.0~1.0},\n'
            '    {"job": "기타", "prob": 0.0~1.0}\n'
            "  ],\n"
            '  "evidence": ["셔츠","재킷","백팩","운동복","안전조끼"]\n'
            "}"
        )
    else:
        return (
            "This is a background-removed outfit cutout.\n"
            "Infer the likely occupation based on clothing and visible items.\n"
            "Respond strictly in JSON ONLY:\n"
            "{\n"
            '  "top_prediction": {"job": "job_name", "confidence": 0.0-1.0},\n'
            '  "candidates": [\n'
            '    {"job": "Office/Developer", "prob": 0.0-1.0},\n'
            '    {"job": "Sales/Office", "prob": 0.0-1.0},\n'
            '    {"job": "Student/Research", "prob": 0.0-1.0},\n'
            '    {"job": "Sports/Leisure", "prob": 0.0-1.0},\n'
            '    {"job": "Other", "prob": 0.0-1.0}\n'
            "  ],\n"
            '  "evidence": ["shirt","blazer","backpack","sportswear","safety vest"]\n'
            "}"
        )

def parse_json_safely(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        try:
            s = text.find("{"); e = text.rfind("}")
            if s >= 0 and e > s:
                return json.loads(text[s:e+1])
        except Exception:
            pass
    return {"raw": text}

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
def qwen_infer(image: Image.Image, prompt: str, gen_cfg: Dict[str, Any]) -> str:
    mdl, proc, _ = get_qwen()
    inputs = proc(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(mdl.device)

    gen_kwargs = dict(
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=gen_cfg.get("do_sample", True),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        top_k=gen_cfg.get("top_k", 50),
        pad_token_id=mdl.generation_config.pad_token_id,
        eos_token_id=mdl.generation_config.eos_token_id,
    )

    out_ids = mdl.generate(**inputs, **gen_kwargs)
    out_text = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return out_text


# ------------------------- VLM API -------------------------
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

    prompt = build_prompt(lang)
    raw = qwen_infer(image, prompt, gen_cfg)
    parsed = parse_json_safely(raw)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw,
        result=parsed,
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
    prompt = build_prompt(lang)
    raw_text = qwen_infer(image, prompt, gen_cfg)
    parsed = parse_json_safely(raw_text)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw_text,
        result=parsed,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
    )