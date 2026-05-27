#!/usr/bin/env python3
"""
scripts/detect_all_gdino.py

kyne0127/vla-evaluation의 모든 PNG 281장에 대해 GroundingDINO로 탐지 후
바운딩박스 시각화를 저장합니다.

구조:
  data-evaluation/S{1-6}/trial_{001-010}/  t0.png, c1.png, c2.png  (180장)
  data-training/{combo}/                   img_{001-011}.png        (101장)
  합계: 281장

실행 (GPU 서버):
    python /workspace/COSE461-proj/scripts/detect_all_gdino.py \
        --out-dir /tmp/gdino_all_vis

옵션:
    --model-id  HuggingFace 모델 ID (기본: IDEA-Research/grounding-dino-base)
    --box-thr   box confidence 임계값 (기본: 0.30)
    --text-thr  text matching 임계값 (기본: 0.25)
    --out-dir   결과 저장 디렉토리 (기본: /tmp/gdino_all_vis)
    --cache-dir 이미지 캐시 디렉토리 (기본: /tmp/gdino_all_imgs)
    --no-enrich label enrichment 비활성화
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ID = "kyne0127/vla-evaluation"

EVAL_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]
EVAL_TRIALS    = [f"trial_{i:03d}" for i in range(1, 11)]
EVAL_FRAMES    = ["t0", "c1", "c2"]

TRAIN_COMBOS = [
    "cube1_redbox1", "cube1_twobox", "cube1_yellowbox1",
    "cube2_redbox1", "cube2_yellowbox1",
    "cup1_redbox1",  "cup1_twobox",  "cup1_yellowbox1",
    "cup2_redbox1",  "cup2_yellowbox1",
]

COLORS: Dict[str, str] = {
    "cup":         "#4EC9B0",
    "red box":     "#F44747",
    "box":         "#FFD700",
    "cube":        "#FF8C00",
    "yellow box":  "#ADFF2F",
}
DEFAULT_COLOR = "#FFFFFF"

_LABEL_ENRICHMENT = {
    "cup":        "paper cup or disposable cup viewed from above",
    "box":        "cardboard box or plastic storage box",
    "red box":    "red cardboard box or red plastic box",
    "cube":       "small colored cube block",
    "yellow box": "yellow cardboard box or yellow plastic box",
}


def _color(label: str) -> str:
    return COLORS.get(label.lower().strip(), DEFAULT_COLOR)


# ── HuggingFace 다운로드 ──────────────────────────────────────────────────────

def _hf_download(hf_path: str, dst: Path) -> bool:
    from huggingface_hub import hf_hub_download
    if dst.exists():
        return True
    try:
        cached = hf_hub_download(REPO_ID, repo_type="dataset", filename=hf_path)
        shutil.copy2(cached, dst)
        return True
    except Exception as e:
        print(f"  [fail] {hf_path}: {e}", flush=True)
        return False


def build_entries(cache_dir: Path) -> List[Dict]:
    """
    HF에서 모든 이미지와 메타를 다운로드하고 entry 목록을 반환합니다.
    entry: {sid, split, img_path, target_label, destination_label, task}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    total_eval  = len(EVAL_SCENARIOS) * len(EVAL_TRIALS) * len(EVAL_FRAMES)
    total_train = sum(1 for _ in range(101))  # 실제는 아래서 카운트

    # ── evaluation ──────────────────────────────────────────────────────────
    print(f"  [eval] {len(EVAL_SCENARIOS)*len(EVAL_TRIALS)} trials × {len(EVAL_FRAMES)} frames = {total_eval}장", flush=True)
    meta_cache: Dict[str, Dict] = {}  # (scenario, trial) → meta

    for scenario in EVAL_SCENARIOS:
        for trial in EVAL_TRIALS:
            prefix = f"data-evaluation/{scenario}/{trial}"
            meta_key = f"{scenario}/{trial}"

            # meta.json (trial당 1회만 다운)
            meta_dst = cache_dir / f"eval_{scenario.lower()}_{trial}__meta.json"
            meta = {}
            if _hf_download(f"{prefix}/meta.json", meta_dst):
                try:
                    meta = json.load(open(meta_dst, encoding="utf-8"))
                except Exception:
                    pass
            meta_cache[meta_key] = meta

            for frame in EVAL_FRAMES:
                sid      = f"eval_{scenario.lower()}_{trial}_{frame}"
                img_dst  = cache_dir / f"{sid}.png"
                ok = _hf_download(f"{prefix}/{frame}.png", img_dst)
                if not ok:
                    continue

                entries.append({
                    "sid":               sid,
                    "split":             "eval",
                    "scenario":          scenario,
                    "trial":             trial,
                    "frame":             frame,
                    "img_path":          img_dst,
                    "task":              meta.get("task", ""),
                    "target_label":      meta.get("target_label", ""),
                    "destination_label": meta.get("destination_label", ""),
                })

    # ── training ─────────────────────────────────────────────────────────────
    print(f"  [train] {len(TRAIN_COMBOS)} combos", flush=True)

    for combo in TRAIN_COMBOS:
        prefix = f"data-training/{combo}"
        meta_dst = cache_dir / f"train_{combo}__meta.json"
        meta = {}
        if _hf_download(f"{prefix}/meta.json", meta_dst):
            try:
                meta = json.load(open(meta_dst, encoding="utf-8"))
            except Exception:
                pass

        img_count = meta.get("image_count", 10)
        for idx in range(1, img_count + 1):
            fname = f"img_{idx:03d}.png"
            sid   = f"train_{combo}_{fname.replace('.png','')}"
            img_dst = cache_dir / f"{sid}.png"
            ok = _hf_download(f"{prefix}/{fname}", img_dst)
            if not ok:
                continue
            entries.append({
                "sid":               sid,
                "split":             "train",
                "combo":             combo,
                "img_path":          img_dst,
                "task":              meta.get("task", ""),
                "target_label":      meta.get("target_label", ""),
                "destination_label": meta.get("destination_label", ""),
            })

    return entries


# ── GroundingDINO ─────────────────────────────────────────────────────────────

class GDino:
    def __init__(self, model_id: str, box_thr: float, text_thr: float, enrich: bool):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"  loading {model_id} …", flush=True)
        self.proc  = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()
        self.model.eval()
        self.box_thr  = box_thr
        self.text_thr = text_thr
        self.enrich   = enrich

    def _enrich(self, label: str) -> str:
        if self.enrich:
            return _LABEL_ENRICHMENT.get(label.lower().strip(), label.lower().strip())
        return label.lower().strip()

    def _match(self, detected: str, objects: List[str]):
        dl = detected.lower()
        for o in objects:
            if dl == o.lower():
                return o
        for o in objects:
            ol = o.lower()
            if dl in ol or ol in dl:
                return o
        dlw = set(dl.split())
        bs, bo = 0, None
        for o in objects:
            ov = len(dlw & set(o.lower().split()))
            if ov > bs:
                bs, bo = ov, o
        return bo if bs > 0 else None

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)

    def _nms(self, dets: list, iou_thr: float = 0.5) -> list:
        keep = []
        for d in dets:
            if not any(self._iou(d["box"], k["box"]) > iou_thr for k in keep):
                keep.append(d)
        return keep

    def _cross_label_nms(self, buckets: Dict[str, list], iou_thr: float = 0.5) -> Dict[str, list]:
        all_dets = [(lbl, det) for lbl, dets in buckets.items() for det in dets]
        all_dets.sort(key=lambda x: x[1]["score"], reverse=True)
        kept = []
        for lbl, det in all_dets:
            if not any(kl != lbl and self._iou(det["box"], kd["box"]) > iou_thr
                       for kl, kd in kept):
                kept.append((lbl, det))
        result = {lbl: [] for lbl in buckets}
        for lbl, det in kept:
            result[lbl].append(det)
        return result

    def detect(self, pil_img: Image.Image, objects: List[str]) -> Dict[str, list]:
        qmap    = {o: self._enrich(o) for o in objects}
        enriched = list(qmap.values())
        rev     = {v: k for k, v in qmap.items()}
        text    = ". ".join(enriched) + "."
        W, H    = pil_img.size
        inp     = self.proc(images=pil_img, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids,
            box_threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(H, W)],
        )[0]
        buckets = {o: [] for o in objects}
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            lbl  = label.lower().strip() if isinstance(label, str) else str(label)
            m    = self._match(lbl, enriched)
            if m is None:
                continue
            orig = rev.get(m, m)
            x1, y1, x2, y2 = box.tolist()
            buckets[orig].append({"box": [x1, y1, x2, y2], "score": float(score), "label": orig})
        buckets = {o: self._nms(sorted(v, key=lambda d: d["score"], reverse=True))
                   for o, v in buckets.items()}
        return self._cross_label_nms(buckets)


# ── 시각화 ────────────────────────────────────────────────────────────────────

def _font(size: int = 14):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def visualize(pil_img: Image.Image, detections: Dict[str, list], entry: Dict) -> Image.Image:
    out  = pil_img.copy()
    draw = ImageDraw.Draw(out)
    font = _font(14)

    for obj, dets in detections.items():
        col = _color(obj)
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            tag = f"{obj} {d['score']:.2f}"
            tw  = draw.textlength(tag, font=font) if hasattr(draw, "textlength") else len(tag) * 7
            draw.rectangle([x1, y1 - 20, x1 + tw + 6, y1], fill=col)
            draw.text((x1 + 3, y1 - 19), tag, fill="black", font=font)

    W, H = pil_img.size
    BAR  = 30
    canvas = Image.new("RGB", (W, H + BAR), (20, 20, 20))
    canvas.paste(out, (0, BAR))
    d2   = ImageDraw.Draw(canvas)
    det_summary = ", ".join(f"{o}×{len(v)}" for o, v in detections.items() if v) or "none"
    info = f"{entry['sid']} | {entry.get('task','')} | {det_summary}"
    d2.text((6, 8), info, fill="#CCCCCC", font=_font(12))
    return canvas


# ── 요약 출력 ─────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    total = len(results)
    miss_t = sum(1 for r in results if not r["detections"].get(r["target_label"]))
    miss_d = sum(1 for r in results if not r["detections"].get(r["destination_label"]))

    # split별
    from collections import Counter
    split_stats: Dict[str, Dict] = {}
    for r in results:
        sp = r.get("split", "?")
        if sp not in split_stats:
            split_stats[sp] = {"total": 0, "miss_t": 0, "miss_d": 0}
        split_stats[sp]["total"] += 1
        if not r["detections"].get(r["target_label"]):
            split_stats[sp]["miss_t"] += 1
        if not r["detections"].get(r["destination_label"]):
            split_stats[sp]["miss_d"] += 1

    print(f"\n{'═'*60}")
    print("  Detection Summary")
    print(f"{'═'*60}")
    print(f"  {'split':<10} {'images':>7} {'tgt miss':>9} {'dst miss':>9} {'tgt rec':>9} {'dst rec':>9}")
    print(f"  {'─'*56}")
    for sp, s in sorted(split_stats.items()):
        n = s["total"]
        mt, md = s["miss_t"], s["miss_d"]
        print(f"  {sp:<10} {n:>7} {mt:>9} {md:>9} {(n-mt)/n:>9.1%} {(n-md)/n:>9.1%}")
    print(f"  {'─'*56}")
    print(f"  {'[total]':<10} {total:>7} {miss_t:>9} {miss_d:>9} {(total-miss_t)/total:>9.1%} {(total-miss_d)/total:>9.1%}")
    print(f"{'═'*60}\n")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",  default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-thr",   type=float, default=0.30)
    parser.add_argument("--text-thr",  type=float, default=0.25)
    parser.add_argument("--out-dir",   default="/tmp/gdino_all_vis")
    parser.add_argument("--cache-dir", default="/tmp/gdino_all_imgs")
    parser.add_argument("--no-enrich", action="store_true")
    args = parser.parse_args()

    out_dir   = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 다운로드
    print(f"\n[1/3] HuggingFace 다운로드 → {cache_dir}")
    entries = build_entries(cache_dir)
    print(f"  총 {len(entries)}장 준비됨\n")

    # 2. 탐지
    print(f"[2/3] GroundingDINO 탐지  model={args.model_id}  box={args.box_thr}  text={args.text_thr}")
    detector = GDino(args.model_id, args.box_thr, args.text_thr, not args.no_enrich)

    results = []
    for i, entry in enumerate(entries, 1):
        img_path = Path(entry["img_path"])
        if not img_path.exists():
            continue

        pil     = Image.open(img_path).convert("RGB")
        objects = []
        for lbl in [entry["target_label"], entry["destination_label"]]:
            if lbl and lbl not in objects:
                objects.append(lbl)
        if not objects:
            continue

        dets = detector.detect(pil, objects)

        vis       = visualize(pil, dets, entry)
        save_path = out_dir / f"{entry['sid']}.png"
        vis.save(save_path)

        tgt_ok = "✓" if dets.get(entry["target_label"]) else "✗"
        dst_ok = "✓" if dets.get(entry["destination_label"]) else "✗"
        print(f"  [{i:03d}/281] {entry['sid']:<40} tgt{tgt_ok} dst{dst_ok}", flush=True)

        results.append({**entry, "detections": dets, "img_path": str(img_path)})

    # 3. 요약 및 JSON 저장
    print_summary(results)

    summary_path = out_dir / "detection_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            [{
                "sid":               r["sid"],
                "split":             r.get("split"),
                "task":              r.get("task", ""),
                "target_label":      r["target_label"],
                "destination_label": r["destination_label"],
                "detections": {
                    obj: [{"box": d["box"], "score": d["score"]} for d in dets]
                    for obj, dets in r["detections"].items()
                },
            } for r in results],
            f, ensure_ascii=False, indent=2,
        )

    print(f"시각화: {out_dir}/*.png  ({len(results)}장)")
    print(f"JSON:   {summary_path}")


if __name__ == "__main__":
    main()
