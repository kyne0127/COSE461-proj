#!/usr/bin/env python3
"""
scripts/visualize_detections.py

GT bbox(초록) + 베이스라인 예측(파랑) + 파인튜닝 예측(빨강)을 한 이미지에 그려
세 패널을 가로로 나란히 저장합니다.

GPU 서버 실행:
    python /workspace/COSE461-proj/scripts/visualize_detections.py \
        --coco-json /tmp/cvat_upload/annotations.json \
        --image-dir /tmp/cvat_upload \
        --ft-model  /workspace/COSE461-proj/checkpoints/gdino-ft/best \
        --out-dir   /tmp/det_vis
"""

from __future__ import annotations
import argparse, json, textwrap
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────
COLORS = {
    "cup":     "#FF6B6B",
    "red box": "#FF0000",
    "cube":    "#FFA500",
    "box":     "#FFD700",
}
DEFAULT_COLOR = "#FFFFFF"

def _color(label: str) -> str:
    return COLORS.get(label.lower(), DEFAULT_COLOR)

# ── 그리기 헬퍼 ──────────────────────────────────────────────────────────────

def draw_boxes(img: Image.Image, boxes: list, label_prefix: str = "") -> Image.Image:
    """boxes: [{"box":[x1,y1,x2,y2], "label":str, "score":float, "style":"gt"|"pred"}]"""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        lbl   = b["label"]
        score = b.get("score", None)
        style = b.get("style", "pred")

        col   = _color(lbl)
        width = 3 if style == "gt" else 2
        dash  = style == "gt"

        if dash:
            # GT: 점선 효과 (실선 + 흰 테두리)
            draw.rectangle([x1-1,y1-1,x2+1,y2+1], outline="white", width=1)
            draw.rectangle([x1,y1,x2,y2], outline=col, width=width)
        else:
            draw.rectangle([x1,y1,x2,y2], outline=col, width=width)

        tag = f"{label_prefix}{lbl}"
        if score is not None:
            tag += f" {score:.2f}"
        tw = draw.textlength(tag, font=font) if hasattr(draw, "textlength") else len(tag)*7
        draw.rectangle([x1, y1-18, x1+tw+4, y1], fill=col)
        draw.text((x1+2, y1-17), tag, fill="black", font=font)

    return out


def make_panel(gt_img, base_img, ft_img, title: str, W: int) -> Image.Image:
    """세 이미지를 가로로 붙이고 상단에 타이틀을 추가."""
    H = gt_img.height
    BAR = 28
    panel = Image.new("RGB", (W*3, H+BAR), (30, 30, 30))

    panel.paste(gt_img,   (0,     BAR))
    panel.paste(base_img, (W,     BAR))
    panel.paste(ft_img,   (W*2,   BAR))

    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    labels = ["GT (초록)", "Baseline", "Fine-tuned"]
    for i, lbl in enumerate(labels):
        draw.text((i*W + 6, 6), lbl, fill="white", font=font)
    draw.text((W, 6), f"  ← {title}", fill="#AAAAAA", font=font)

    return panel


# ── NMS 헬퍼 ─────────────────────────────────────────────────────────────────

def _iou(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/max(a1+a2-inter,1e-6)

def _nms(dets, iou_thr=0.5):
    keep=[]
    for d in dets:
        if not any(_iou(d["box"],k["box"])>iou_thr for k in keep):
            keep.append(d)
    return keep


# ── 탐지기 ───────────────────────────────────────────────────────────────────

_ENRICHMENT = {
    "cup":      "paper cup or disposable cup viewed from above",
    "box":      "cardboard box or plastic storage box",
    "red box":  "red cardboard box or red plastic box",
    "cube":     "small colored cube block",
}
def _enrich(l): return _ENRICHMENT.get(l.lower().strip(), l.lower().strip())

def _match(detected, objects):
    dl = detected.lower()
    for o in objects:
        if dl == o.lower(): return o
    for o in objects:
        ol = o.lower()
        if dl in ol or ol in dl: return o
    dlw = set(dl.split())
    bs, bo = 0, None
    for o in objects:
        ov = len(dlw & set(o.lower().split()))
        if ov > bs: bs, bo = ov, o
    return bo if bs > 0 else None


class Detector:
    def __init__(self, model_id, box_thr=0.30, text_thr=0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"  loading {model_id} …", flush=True)
        self.proc  = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()
        self.model.eval()
        self.box_thr = box_thr; self.text_thr = text_thr

    def detect(self, pil_img, objects):
        qmap = {o: _enrich(o) for o in objects}
        enr  = list(qmap.values())
        rev  = {v:k for k,v in qmap.items()}
        text = ". ".join(enr) + "."
        W, H = pil_img.size
        inp = self.proc(images=pil_img, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids,
            box_threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(H,W)]
        )[0]
        buckets = {o:[] for o in objects}
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            lbl = label.lower().strip() if isinstance(label,str) else str(label)
            m   = _match(lbl, enr)
            if m is None: continue
            orig = rev.get(m, m)
            buckets[orig].append({"box": box.tolist(), "score": float(score), "label": orig})
        return {o: _nms(sorted(v, key=lambda d:d["score"], reverse=True))
                for o, v in buckets.items()}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json",  required=True)
    parser.add_argument("--image-dir",  required=True)
    parser.add_argument("--ft-model",   required=True)
    parser.add_argument("--base-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--out-dir",    default="/tmp/det_vis")
    parser.add_argument("--box-thr",    type=float, default=0.30)
    parser.add_argument("--text-thr",   type=float, default=0.25)
    parser.add_argument("--max-images", type=int,   default=59)
    args = parser.parse_args()

    with open(args.coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id2meta = {img["id"]: img for img in coco["images"]}
    ann_by_img: Dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.image_dir)

    base_det = Detector(args.base_model, args.box_thr, args.text_thr)
    ft_det   = Detector(args.ft_model,   args.box_thr, args.text_thr)

    img_ids = sorted(img_id2meta.keys())[:args.max_images]
    print(f"\n{len(img_ids)}장 처리 시작 →  {out_dir}\n")

    for img_id in img_ids:
        meta = img_id2meta[img_id]
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        pil = Image.open(img_dir / meta["file_name"]).convert("RGB")
        W, H = pil.size

        # GT 박스 구성
        gt_boxes_draw = []
        query_classes = []
        for ann in anns:
            name = cat_id2name[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            gt_boxes_draw.append({"box":[x,y,x+bw,y+bh], "label":name, "style":"gt"})
            if name not in query_classes:
                query_classes.append(name)

        # 탐지
        base_preds = base_det.detect(pil, query_classes)
        ft_preds   = ft_det.detect(pil, query_classes)

        def flatten(preds):
            out = []
            for cls, dets in preds.items():
                for d in dets:
                    out.append({**d, "style":"pred"})
            return out

        # 패널 생성
        gt_panel   = draw_boxes(pil, gt_boxes_draw)
        base_panel = draw_boxes(pil, flatten(base_preds))
        ft_panel   = draw_boxes(pil, flatten(ft_preds))

        title = meta["file_name"].replace("__t0.png","")
        panel = make_panel(gt_panel, base_panel, ft_panel, title, W)

        save_name = meta["file_name"].replace(".png","_det.png")
        panel.save(out_dir / save_name)
        print(f"  {save_name}")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()
