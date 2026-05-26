#!/usr/bin/env python3
"""
scripts/eval_gdino_quantitative.py

instances_default.json (GT) vs 모델 예측 비교.
베이스라인과 파인튜닝 모델을 동일한 59장 이미지에 대해 평가하여
class별 Precision / Recall / F1을 출력합니다.

GPU 서버 실행:
    python /workspace/COSE461-proj/scripts/eval_gdino_quantitative.py \
        --coco-json  /tmp/cvat_upload/annotations.json \
        --image-dir  /tmp/cvat_upload \
        --ft-model   /workspace/COSE461-proj/checkpoints/gdino-ft/best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

# ── NMS 헬퍼 ─────────────────────────────────────────────────────────────────

def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / max(a1+a2-inter, 1e-6)


def _nms(dets, iou_thr=0.5):
    keep = []
    for d in dets:
        if not any(_iou(d["box"], k["box"]) > iou_thr for k in keep):
            keep.append(d)
    return keep


# ── 탐지 ─────────────────────────────────────────────────────────────────────

_ENRICHMENT = {
    "cup":      "paper cup or disposable cup viewed from above",
    "box":      "cardboard box or plastic storage box",
    "red box":  "red cardboard box or red plastic box",
    "cube":     "small colored cube block",
}

def _enrich(label: str) -> str:
    return _ENRICHMENT.get(label.lower().strip(), label.lower().strip())


def _match_label(detected: str, objects: List[str]):
    dl = detected.lower()
    for o in objects:
        if dl == o.lower(): return o
    for o in objects:
        ol = o.lower()
        if dl in ol or ol in dl: return o
    dl_w = set(dl.split())
    best_s, best_o = 0, None
    for o in objects:
        ov = len(dl_w & set(o.lower().split()))
        if ov > best_s:
            best_s, best_o = ov, o
    return best_o if best_s > 0 else None


class Detector:
    def __init__(self, model_id: str, box_thr=0.30, text_thr=0.25, enrich=True):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc  = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()
        self.model.eval()
        self.box_thr  = box_thr
        self.text_thr = text_thr
        self.enrich   = enrich

    def detect(self, pil_img: Image.Image, objects: List[str]) -> Dict[str, list]:
        query_map   = {o: (_enrich(o) if self.enrich else o.lower()) for o in objects}
        enriched    = list(query_map.values())
        reverse_map = {v: k for k, v in query_map.items()}
        text = ". ".join(enriched) + "."
        W, H = pil_img.size
        inputs = self.proc(images=pil_img, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids,
            box_threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(H, W)],
        )[0]
        buckets = {o: [] for o in objects}
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            lbl     = label.lower().strip() if isinstance(label, str) else str(label)
            matched = _match_label(lbl, enriched)
            if matched is None: continue
            original = reverse_map.get(matched, matched)
            x1, y1, x2, y2 = box.tolist()
            buckets[original].append({"box": [x1,y1,x2,y2], "score": float(score)})
        return {o: _nms(sorted(v, key=lambda d: d["score"], reverse=True))
                for o, v in buckets.items()}


# ── 평가 ─────────────────────────────────────────────────────────────────────

def evaluate(detector: Detector, coco_path: str, image_dir: str, iou_thr: float = 0.5):
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id2meta = {img["id"]: img for img in coco["images"]}
    ann_by_img: Dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # {class_name: {tp, fp, fn}}
    stats: Dict[str, Dict[str, int]] = {}
    for name in cat_id2name.values():
        stats[name] = {"tp": 0, "fp": 0, "fn": 0}

    img_dir = Path(image_dir)

    for img_id, meta in img_id2meta.items():
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        pil = Image.open(img_dir / meta["file_name"]).convert("RGB")
        W, H = pil.size

        # GT: {class_name: [[x1,y1,x2,y2], ...]}
        gt: Dict[str, list] = {}
        for ann in anns:
            name = cat_id2name[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            gt.setdefault(name, []).append([x, y, x+bw, y+bh])

        # 이 이미지에 등장하는 클래스만 쿼리
        query_classes = [c for c in gt if c in stats]
        if not query_classes:
            continue

        preds = detector.detect(pil, query_classes)

        # 클래스별 TP/FP/FN
        for cls in query_classes:
            gt_boxes   = gt.get(cls, [])
            pred_boxes = [d["box"] for d in preds.get(cls, [])]

            matched_gt  = set()
            matched_pred = set()

            for pi, pb in enumerate(pred_boxes):
                best_iou, best_gi = 0.0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = _iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_thr:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                    stats[cls]["tp"] += 1
                else:
                    stats[cls]["fp"] += 1

            stats[cls]["fn"] += len(gt_boxes) - len(matched_gt)

    return stats


def print_results(label: str, stats: Dict[str, Dict[str, int]]):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"{'Class':<15} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"{'-'*55}")

    total_tp = total_fp = total_fn = 0
    for cls, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        if tp+fp == 0: continue   # 쿼리되지 않은 클래스 생략
        prec = tp / (tp+fp) if tp+fp else 0
        rec  = tp / (tp+fn) if tp+fn else 0
        f1   = 2*prec*rec / (prec+rec) if prec+rec else 0
        print(f"{cls:<15} {tp:>4} {fp:>4} {fn:>4}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")
        total_tp += tp; total_fp += fp; total_fn += fn

    prec = total_tp/(total_tp+total_fp) if total_tp+total_fp else 0
    rec  = total_tp/(total_tp+total_fn) if total_tp+total_fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    print(f"{'-'*55}")
    print(f"{'[Overall]':<15} {total_tp:>4} {total_fp:>4} {total_fn:>4}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json",  required=True)
    parser.add_argument("--image-dir",  required=True)
    parser.add_argument("--ft-model",   required=True,
                        help="파인튜닝 모델 경로")
    parser.add_argument("--base-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-thr",    type=float, default=0.30)
    parser.add_argument("--text-thr",   type=float, default=0.25)
    parser.add_argument("--iou-thr",    type=float, default=0.50)
    args = parser.parse_args()

    print(f"IoU threshold: {args.iou_thr}  box_thr: {args.box_thr}  text_thr: {args.text_thr}")

    print("\n[1/2] 베이스라인 모델 평가 중...")
    base_det   = Detector(args.base_model, args.box_thr, args.text_thr)
    base_stats = evaluate(base_det, args.coco_json, args.image_dir, args.iou_thr)
    del base_det
    torch.cuda.empty_cache()

    print("[2/2] 파인튜닝 모델 평가 중...")
    ft_det   = Detector(args.ft_model, args.box_thr, args.text_thr)
    ft_stats = evaluate(ft_det, args.coco_json, args.image_dir, args.iou_thr)

    print_results("Baseline (grounding-dino-base)", base_stats)
    print_results(f"Fine-tuned ({args.ft_model.split('/')[-2]})", ft_stats)


if __name__ == "__main__":
    main()
