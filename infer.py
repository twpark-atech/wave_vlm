# infer.py
# YOLOv8(DeepFashion2)로 의류 세그먼트 → 원본 색 보존 + 배경 투명(누끼) PNG 생성
# 사용법:
#   python infer.py --source test.jpg --out runs/seg --model deepfashion2_yolov8s-seg.pt --device cuda

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 DeepFashion2 segmentation → cutout PNG")
    p.add_argument("--source", type=str, required=True, help="입력 이미지 경로(또는 폴더)")
    p.add_argument("--out", type=str, default="runs/seg_cutout", help="출력 폴더")
    p.add_argument("--model", type=str, default="deepfashion2_yolov8s-seg.pt", help="YOLOv8 세그 모델 가중치")
    p.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", type=str, default="cuda", help="cuda 또는 cpu")
    p.add_argument("--keep-classes", type=str, default="", 
                   help="유지할 클래스(쉼표로 구분, 이름 또는 id). 비우면 모든 의류 마스크 사용")
    p.add_argument("--save-individual", action="store_true", help="인스턴스별 누끼 PNG도 저장")
    p.add_argument("--alpha-soften", type=float, default=1.5, help="알파 경계 부드럽게(가우시안 sigma)")
    p.add_argument("--alpha-dilate", type=int, default=1, help="알파 경계 팽창(픽셀)")
    return p.parse_args()


def to_rgba_with_alpha(orig_bgr: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """
    orig_bgr: HxWx3 BGR (uint8)
    alpha: HxW [0..255] (uint8)
    return: PIL RGBA with original colors + given alpha
    """
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def build_union_alpha(
    masks_bin: List[np.ndarray], 
    h: int, w: int, 
    dilate_px: int = 1, 
    soften_sigma: float = 1.5
) -> np.ndarray:
    """
    여러 이진 마스크를 합쳐서(OR) 알파 채널(0..255) 생성.
    dilate_px: 경계 팽창, soften_sigma: 경계 부드럽기(가우시안)
    """
    if not masks_bin:
        return np.zeros((h, w), dtype=np.uint8)

    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks_bin:
        union = cv2.bitwise_or(union, m)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        union = cv2.dilate(union, k, iterations=1)

    if soften_sigma > 0:
        # 가우시안으로 부드럽게 → 0..255 범위 재정규화
        blurred = cv2.GaussianBlur(union.astype(np.float32), (0, 0), soften_sigma)
        maxi = blurred.max()
        if maxi > 0:
            blurred = blurred * (255.0 / maxi)
        union = np.clip(blurred, 0, 255).astype(np.uint8)

    else:
        union = (union > 0).astype(np.uint8) * 255

    return union


def select_masks(
    result, 
    keep_classes: Optional[List[str]] = None
) -> List[np.ndarray]:
    """
    Ultralytics Results에서 의류 마스크만 선택하여 원본 크기(H,W)의 이진 마스크 목록 반환.
    keep_classes: 남길 클래스 이름 또는 id 문자열 목록. None이면 전부 사용.
    """
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None or len(masks_obj.data) == 0:
        return []

    # 원본 크기
    H, W = result.orig_img.shape[:2]

    # 클래스 id/name
    names = result.names  # dict: {id: class_name}
    cls_ids = result.boxes.cls.int().cpu().numpy() if result.boxes is not None else np.array([], dtype=int)

    # keep 목록 파싱(id 또는 name)
    keep_ids = None
    if keep_classes:
        parsed = []
        for token in keep_classes:
            token = token.strip()
            if token == "":
                continue
            if token.isdigit():
                parsed.append(int(token))
            else:
                # 이름 → id
                # names는 dict 형태이므로 역탐색
                matched = [k for k, v in names.items() if v == token]
                if matched:
                    parsed.extend(matched)
        keep_ids = set(parsed)

    masks_tf = masks_obj.data  # (N, h, w) torch.bool/float
    masks_bin_list = []
    for i in range(masks_tf.shape[0]):
        # 클래스 필터링
        if keep_ids is not None and len(cls_ids) > i:
            if int(cls_ids[i]) not in keep_ids:
                continue

        m_small = masks_tf[i].cpu().numpy()  # (h, w) in model scale
        m_small = (m_small > 0.5).astype(np.uint8)  # binarize

        # 원본 크기로 리사이즈
        m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        masks_bin_list.append(m)

    return masks_bin_list


def process_image(
    model: YOLO,
    img_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
    keep_classes: Optional[List[str]],
    save_individual: bool,
    alpha_soften: float,
    alpha_dilate: int,
):
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for ri, res in enumerate(results):
        orig_bgr = res.orig_img.copy()  # HxWx3 (BGR, uint8)
        H, W = orig_bgr.shape[:2]

        # 마스크 선택
        masks = select_masks(res, keep_classes)
        if not masks:
            # 마스크 없으면 투명 배경만 저장
            alpha = np.zeros((H, W), dtype=np.uint8)
            cutout = to_rgba_with_alpha(orig_bgr, alpha)
            cutout.save(out_dir / f"{img_path.stem}_cutout_none.png")
            continue

        # 전체(합집합) 알파
        alpha = build_union_alpha(masks, H, W, dilate_px=alpha_dilate, soften_sigma=alpha_soften)
        cutout = to_rgba_with_alpha(orig_bgr, alpha)
        cutout_path = out_dir / f"{img_path.stem}_cutout.png"
        cutout.save(cutout_path)

        # 인스턴스별 저장 옵션
        if save_individual:
            for k, m in enumerate(masks):
                a = build_union_alpha([m], H, W, dilate_px=alpha_dilate, soften_sigma=alpha_soften)
                item = to_rgba_with_alpha(orig_bgr, a)
                item.save(out_dir / f"{img_path.stem}_item{k+1}.png")

        # 시각화(선택): 마스크 외곽선 오버레이 PNG
        overlay = orig_bgr.copy()
        for m in masks:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (255, 200, 60), 2)
        vis = cv2.addWeighted(orig_bgr, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_overlay.jpg"), vis)


def collect_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
    elif p.is_file():
        return [p]
    else:
        return []


def main():
    args = parse_args()

    keep_classes: Optional[List[str]] = None
    if args.keep_classes.strip():
        keep_classes = [s.strip() for s in args.keep_classes.split(",") if s.strip()]

    model = YOLO(args.model)

    images = collect_images(args.source)
    if not images:
        raise FileNotFoundError(f"입력 이미지가 없습니다: {args.source}")

    out_dir = Path(args.out)
    for img_path in images:
        process_image(
            model=model,
            img_path=img_path,
            out_dir=out_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            keep_classes=keep_classes,
            save_individual=args.save_individual,
            alpha_soften=args.alpha_soften,
            alpha_dilate=args.alpha_dilate,
        )


if __name__ == "__main__":
    main()
