# app.py
from __future__ import annotations
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

# ===================== prompts.py에서 제공 =====================
from prompts import (
    get_vlm_prompt,
    get_system_prompt,
    get_stylist_prompt,
    get_system_prompt_stylist,
)

# ===================== 설정 로딩 =====================
def load_yaml_config() -> Dict[str, Any]:
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

    cfg.setdefault("device", "cuda")
    cfg.setdefault("max_side", 896)
    cfg.setdefault("qwen_model", "Qwen/Qwen2-VL-7B-Instruct")
    cfg.setdefault("lang", "ko")
    gen = cfg.get("gen") or {}
    gen.setdefault("max_new_tokens", 1024)
    gen.setdefault("do_sample", False)   # 분류는 기본 greedy
    gen.setdefault("temperature", 0.5)
    gen.setdefault("top_p", 0.6)
    gen.setdefault("top_k", 0)
    cfg["gen"] = gen
    return cfg

CONFIG = load_yaml_config()

# ====== 사람 세그 모델로 교체 ======
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo12l-person-seg.pt")
DEVICE = os.environ.get("DEVICE", CONFIG.get("device", "cuda"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs/person_seg")); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = Path(os.getcwd()) / "main.html"
DATAURL_RE = re.compile(r"^data:(?P<mime>[\w/+.\-]+);base64,(?P<b64>.+)$", re.IGNORECASE)

app = FastAPI(title="Person Segmentation + VLM (Qwen2-VL)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR), html=False), name="outputs")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# ===================== HTML 서빙 =====================
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(HTML_PATH))

@app.get("/main.html", include_in_schema=False)
def serve_main():
    return FileResponse(str(HTML_PATH))

# ===================== 스키마 =====================
class SegmentReq(BaseModel):
    image: str
    meta: Optional[Dict[str, Any]] = None  # 예: {"c1":"#22d3ee"}

class SegmentRes(BaseModel):
    ok: bool
    url: str
    filename: str
    width: int
    height: int
    mode: str
    detections: int = 0
    overlay_url: Optional[str] = None
    picked_index: Optional[int] = None

class VLMJsonReq(BaseModel):
    cutout_url: Optional[str] = None
    image: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # 예: {"c1":"#22d3ee"}

class VLMRes(BaseModel):
    ok: bool
    used_image_url: Optional[str] = None
    raw: str
    result: Dict[str, Any]
    model_id: str
    gen: Dict[str, Any]
    top3: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[str]] = None
    stylist_text: Optional[str] = None

class InlineReq(BaseModel):
    image: str
    meta: Optional[Dict[str, Any]] = None  # 예: {"c1":"#22d3ee"}

class InlineRes(BaseModel):
    ok: bool
    overlay_url: Optional[str]
    cutout_url: Optional[str]
    vlm: VLMRes
    picked_index: Optional[int] = None

class StylistReq(BaseModel):
    job: str
    top3: Optional[List[str]] = None
    evidence: Optional[List[str]] = None

class StylistTextRes(BaseModel):
    ok: bool
    job: str
    text: str

# ===================== 공용 유틸 =====================
def _b64decode(b64str: str) -> bytes:
    import base64; return base64.b64decode(b64str)

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
    if not s or s == "data:,":  # 빈 dataURL
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

# ======== 색 변환 및 고가시성 오버레이 생성 ========
def hex_to_bgr(hx: Optional[str], default=(34, 211, 238)) -> Tuple[int,int,int]:
    """'#22d3ee' → (238,211,34) BGR. 실패 시 default(BGR)"""
    if not hx:
        return default
    try:
        s = hx.strip().lstrip('#')
        if len(s) == 3:
            s = ''.join([c*2 for c in s])
        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return default

def render_pretty_overlay(
    bgr: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    fill_color=(34,211,238),   # BGR (기본: #22d3ee)
    edge_color=(255,255,255),  # BGR
    fill_alpha=0.38,           # 내부 색상 투명도
    darken_outside=0.5,        # 외곽 어둡게
    edge_thick=3,
    glow_radius=12,
    feather=2
) -> np.ndarray:
    h, w = bgr.shape[:2]
    out = bgr.copy()

    if mask is None:
        return out

    m = (mask > 0).astype(np.uint8) * 255

    if feather > 0:
        m = cv2.GaussianBlur(m, (0,0), feather)
        m = np.clip(m, 0, 255).astype(np.uint8)

    # 1) 배경 어둡게
    if darken_outside > 0:
        inv = cv2.bitwise_not(m)
        dark_layer = (out * (1.0 - darken_outside)).astype(np.uint8)
        inv_3 = cv2.merge([inv, inv, inv])
        out = np.where(inv_3>0, dark_layer, out)

    # 2) 내부 색 채우기
    if fill_alpha > 0:
        fill = np.full_like(out, fill_color, dtype=np.uint8)
        m_f = (m.astype(np.float32) / 255.0 * fill_alpha)[:, :, None]
        out = (out.astype(np.float32) * (1.0 - m_f) + fill.astype(np.float32) * m_f).astype(np.uint8)

    # 3) 경계선 + 글로우
    cnts, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # 밝은 경계
        cv2.drawContours(out, cnts, -1, edge_color, thickness=edge_thick, lineType=cv2.LINE_AA)

        if glow_radius > 0:
            glow = np.zeros_like(out)
            cv2.drawContours(glow, cnts, -1, fill_color, thickness=max(edge_thick, 2), lineType=cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (0,0), glow_radius)
            out = np.clip(out.astype(np.int32) + glow.astype(np.int32) * 0.25, 0, 255).astype(np.uint8)

    return out

def save_overlay_jpg_single(
    bgr: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    c1_hex: Optional[str] = None
) -> Path:
    fill_col = hex_to_bgr(c1_hex, default=(34,211,238))  # 기본 #22d3ee
    overlay = render_pretty_overlay(
        bgr, mask,
        fill_color=fill_col,
        edge_color=(255,255,255),
        fill_alpha=0.38,
        darken_outside=0.5,
        edge_thick=3,
        glow_radius=12,
        feather=2
    )
    stem = uuid.uuid4().hex[:12]
    p = OUTPUT_DIR / f"{stem}_overlay.jpg"
    cv2.imwrite(str(p), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return p

# ===================== YOLO 모델 (싱글톤; 사람 세그) =====================
_yolo_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model

# ===================== 세그 추론(가장 큰 사람 1명 선택) =====================
def pick_largest_mask(masks: List[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    if not masks:
        return None, -1
    # 픽셀 합(=면적) 기준
    areas = [int(m.sum()) for m in masks]
    idx = int(np.argmax(areas))
    return masks[idx], idx

def run_cutout_from_image(
    pil_img: Image.Image,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = DEVICE
) -> Tuple[Image.Image, List[np.ndarray], np.ndarray, int, Optional[np.ndarray]]:
    """
    사람 세그 모델을 사용하여 가장 큰 인물 1명의 누끼를 생성.
    반환: (cutout_rgba, masks_all, orig_bgr, picked_idx, picked_mask)
    """
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    model = get_model()
    res = model.predict(source=bgr, imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
    orig_bgr = res.orig_img
    H, W = orig_bgr.shape[:2]
    masks = select_masks(res)

    picked, idx = pick_largest_mask(masks)
    if picked is None:
        alpha = np.zeros((H, W), dtype=np.uint8)
        cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
        return cutout, [], orig_bgr, -1, None

    alpha = build_union_alpha([picked], H, W, dilate_px=1, soften_sigma=1.5)
    cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
    return cutout, masks, orig_bgr, idx, picked

def run_cutout_from_path(
    img_path: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = DEVICE
) -> Tuple[Image.Image, List[np.ndarray], np.ndarray, int, Optional[np.ndarray]]:
    model = get_model()
    results = model.predict(source=str(img_path), imgsz=imgsz, device=device, conf=conf, verbose=False)
    res = results[0]
    orig_bgr = res.orig_img
    H, W = orig_bgr.shape[:2]
    masks = select_masks(res)

    picked, idx = pick_largest_mask(masks)
    if picked is None:
        alpha = np.zeros((H, W), dtype=np.uint8)
        cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
        return cutout, [], orig_bgr, -1, None

    alpha = build_union_alpha([picked], H, W, dilate_px=1, soften_sigma=1.5)
    cutout = bgr_to_rgba_with_alpha(orig_bgr, alpha)
    return cutout, masks, orig_bgr, idx, picked

# ===================== VLM 준비/유틸 =====================
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

def map_url_to_path(url: str) -> Path:
    return OUTPUT_DIR / Path(url).name

def load_cutout_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return cutout_rgba_to_vlm_rgb(img, max_side=max_side)

def parse_json_safely(text: str) -> Dict[str, Any]:
    import re as _re
    s = (text or "").strip()
    if s.startswith("```"):
        s = _re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = s.rstrip("`").rstrip()
        s = s.replace("```", "")
    b, e = s.find("{"), s.rfind("}")
    if b >= 0 and e > b:
        s = s[b:e+1]
    s = _re.sub(r"(?<!\\)\'", '"', s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = _re.sub(r"\bTrue\b", "true", s)
    s = _re.sub(r"\bFalse\b", "false", s)
    s = _re.sub(r"\bNone\b", "null", s)
    try:
        return json.loads(s)
    except Exception:
        return {"raw": text, "_error": "json_parse_failed"}

def _safe_prob(x) -> float:
    try:
        if isinstance(x, str):
            s = x.strip().lower().replace(" ", "")
            return float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
        return float(x)
    except Exception:
        return 0.0

def sort_candidates_only(parsed: dict) -> dict:
    raw_cands = parsed.get("candidates") or []
    items = []
    for it in raw_cands:
        lab = it.get("job", "")
        prob = _safe_prob(it.get("prob", 0))
        if lab:
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

def extract_top3(result_sorted: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tp = result_sorted.get("top_prediction") or {}
    if tp.get("job"):
        out.append({"job": tp["job"], "prob": float(_safe_prob(tp.get("confidence", 0)))})
    for it in result_sorted.get("candidates", []):
        if len(out) >= 3: break
        out.append({"job": it.get("job", ""), "prob": float(_safe_prob(it.get("prob", 0)))})
    out = [o for o in out if o.get("job")]
    out = sorted(out, key=lambda x: x["prob"], reverse=True)[:3]
    return out

# ===================== VLM 모델(싱글톤) + 생성 함수 =====================
_vlm_model = _vlm_processor = _vlm_dtype = None

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
    system_text = get_system_prompt(lang)  # 한국어 JSON 강제
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
        do_sample=gen_cfg.get("do_sample", False),
        temperature=gen_cfg.get("temperature", 0.5),
        top_p=gen_cfg.get("top_p", 0.6),
        top_k=gen_cfg.get("top_k", 0),
        pad_token_id=mdl.generation_config.pad_token_id,
        eos_token_id=mdl.generation_config.eos_token_id,
    )
    gen_only = out_ids[:, in_len:]
    raw_text = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
    return raw_text

@torch.inference_mode()
def qwen_infer_text(prompt: str, gen_cfg: Dict[str, Any]) -> str:
    mdl, proc, _ = get_qwen()
    system_text = get_system_prompt_stylist("ko")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    chat_text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[chat_text], return_tensors="pt").to(mdl.device)
    in_len = inputs["input_ids"].shape[-1]
    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=min(512, gen_cfg.get("max_new_tokens", 1024)),
        do_sample=True,             # 스타일 제안은 다양성
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=mdl.generation_config.pad_token_id,
        eos_token_id=mdl.generation_config.eos_token_id,
    )
    gen_only = out_ids[:, in_len:]
    return proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

# ===================== 세그 API (사람 세그; 가장 큰 인물만) =====================
@app.post("/api/v1/segment/json", response_model=SegmentRes)
def segment_json(body: SegmentReq):
    pil_img = decode_image_from_dataurl(body.image)
    cutout, masks, orig_bgr, picked_idx, picked_mask = run_cutout_from_image(
        pil_img, imgsz=640, conf=0.25, device=DEVICE
    )
    out_path = save_cutout_png(cutout)
    theme_c1 = (body.meta or {}).get("c1")
    overlay_path = save_overlay_jpg_single(orig_bgr, picked_mask, c1_hex=theme_c1)
    w, h = cutout.size
    return SegmentRes(
        ok=True,
        url=f"/outputs/{out_path.name}",
        filename=out_path.name,
        width=w, height=h, mode="json_dataurl_or_base64",
        detections=len(masks), overlay_url=f"/outputs/{overlay_path.name}",
        picked_index=picked_idx
    )

@app.post("/api/v1/segment/file", response_model=SegmentRes)
async def segment_file(file: UploadFile = File(...)):
    raw = await file.read()
    suffix = Path(file.filename or "img").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(raw)
        tmp_path = Path(tf.name)
    try:
        cutout, masks, orig_bgr, picked_idx, picked_mask = run_cutout_from_path(
            tmp_path, imgsz=640, conf=0.25, device=DEVICE
        )
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass
    out_path = save_cutout_png(cutout)
    overlay_path = save_overlay_jpg_single(orig_bgr, picked_mask)  # 업로드 파일 경로는 meta 없음
    w, h = cutout.size
    return SegmentRes(
        ok=True,
        url=f"/outputs/{out_path.name}",
        filename=out_path.name,
        width=w, height=h,
        mode="multipart_file",
        detections=len(masks),
        overlay_url=f"/outputs/{overlay_path.name}",
        picked_index=picked_idx
    )

# 하위 호환 alias
@app.post("/api/v1/intake/json", response_model=SegmentRes)
def intake_json_alias(body: SegmentReq):
    return segment_json(body)

@app.post("/api/v1/intake/file", response_model=SegmentRes)
async def intake_file_alias(file: UploadFile = File(...)):
    return await segment_file(file)

# ===================== VLM API (cutout URL 또는 dataURL) =====================
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
        pil_img = decode_image_from_dataurl(body.image)
        # 사람 세그 수행 후 가장 큰 인물 컷아웃 저장 → 그걸 VLM 입력
        cutout, masks, orig_bgr, picked_idx, picked_mask = run_cutout_from_image(
            pil_img, imgsz=640, conf=0.25, device=DEVICE
        )
        out_copy = save_cutout_png(cutout)
        used_image_url = f"/outputs/{out_copy.name}"
        image = load_cutout_image(out_copy, max_side=max_side)
    else:
        raise HTTPException(status_code=400, detail="provide cutout_url or image(dataURL)")

    raw_json = qwen_infer(image, get_vlm_prompt(lang), gen_cfg, lang=lang)
    parsed = parse_json_safely(raw_json)
    result_sorted = sort_candidates_only(parsed)

    top3 = extract_top3(result_sorted)
    evidence = result_sorted.get("evidence", []) or []
    top_job = (result_sorted.get("top_prediction") or {}).get("job") or (top3[0]["job"] if top3 else "사무/기업")
    sty_prompt = get_stylist_prompt(job_ko=top_job, evidence=evidence)
    stylist_text = qwen_infer_text(sty_prompt, gen_cfg)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw_json,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
        top3=top3,
        evidence=evidence,
        stylist_text=stylist_text,
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

    # 업로드 이미지를 바로 VLM에 넣지 않고, 사람 세그 → 가장 큰 인물 컷아웃 저장
    try:
        pil_img = Image.open(tmp).convert("RGB")
    except Exception:
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="decode_error")
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    try:
        cutout, masks, orig_bgr, picked_idx, picked_mask = run_cutout_from_image(
            pil_img, imgsz=640, conf=0.25, device=DEVICE
        )
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass

    out_copy = save_cutout_png(cutout)
    used_image_url = f"/outputs/{out_copy.name}"

    image = load_cutout_image(out_copy, max_side=max_side)
    raw_json = qwen_infer(image, get_vlm_prompt(lang), gen_cfg, lang=lang)
    parsed = parse_json_safely(raw_json)
    result_sorted = sort_candidates_only(parsed)

    top3 = extract_top3(result_sorted)
    evidence = result_sorted.get("evidence", []) or []
    top_job = (result_sorted.get("top_prediction") or {}).get("job") or (top3[0]["job"] if top3 else "사무/기업")
    sty_prompt = get_stylist_prompt(job_ko=top_job, evidence=evidence)
    stylist_text = qwen_infer_text(sty_prompt, gen_cfg)

    return VLMRes(
        ok=True,
        used_image_url=used_image_url,
        raw=raw_json,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
        top3=top3,
        evidence=evidence,
        stylist_text=stylist_text,
    )

# ===================== 스타일러 API(문장형 텍스트 전용) =====================
@app.post("/api/v1/stylist/json", response_model=StylistTextRes)
def stylist_json(body: StylistReq):
    gen_cfg = dict(CONFIG.get("gen", {}))
    job = (body.job or "").strip() or "사무/기업"
    ev = body.evidence or []
    prompt = get_stylist_prompt(job_ko=job, evidence=ev)
    text = qwen_infer_text(prompt, gen_cfg)
    return StylistTextRes(ok=True, job=job, text=text)

# ===================== 원샷 파이프라인(세그→VLM→스타일 추천) =====================
@app.post("/api/v1/predict/inline", response_model=InlineRes)
def predict_inline(body: InlineReq):
    cfg = CONFIG
    max_side = int(cfg.get("max_side", 896))
    gen_cfg = dict(cfg.get("gen", {}))
    lang = cfg.get("lang", "ko")

    pil_img = decode_image_from_dataurl(body.image)

    # 사람 세그: 가장 큰 인물 선택
    cutout, masks, orig_bgr, picked_idx, picked_mask = run_cutout_from_image(
        pil_img, imgsz=640, conf=0.25, device=DEVICE
    )
    cutout_path = save_cutout_png(cutout)

    # 프런트에서 넘어온 테마 색(#hex) → 누끼 오버레이에 반영
    theme_c1 = (body.meta or {}).get("c1") if body.meta else None
    overlay_path = save_overlay_jpg_single(orig_bgr, picked_mask, c1_hex=theme_c1)

    # VLM 입력 이미지
    vlm_input = cutout_rgba_to_vlm_rgb(cutout, max_side=max_side)

    # 분류(JSON)
    raw_json = qwen_infer(vlm_input, get_vlm_prompt(lang), gen_cfg, lang=lang)
    parsed = parse_json_safely(raw_json)
    result_sorted = sort_candidates_only(parsed)

    # Top-3, evidence
    top3 = extract_top3(result_sorted)
    evidence = result_sorted.get("evidence", []) or []
    top_job = (result_sorted.get("top_prediction") or {}).get("job") or (top3[0]["job"] if top3 else "사무/기업")

    # 스타일 제안(문장형)
    sty_prompt = get_stylist_prompt(job_ko=top_job, evidence=evidence)
    stylist_text = qwen_infer_text(sty_prompt, gen_cfg)

    vlm_res = VLMRes(
        ok=True,
        used_image_url=f"/outputs/{cutout_path.name}",
        raw=raw_json,
        result=result_sorted,
        model_id=cfg["qwen_model"],
        gen=gen_cfg,
        top3=top3,
        evidence=evidence,
        stylist_text=stylist_text,
    )

    return InlineRes(
        ok=True,
        overlay_url=f"/outputs/{overlay_path.name}",
        cutout_url=f"/outputs/{cutout_path.name}",
        vlm=vlm_res,
        picked_index=picked_idx
    )
