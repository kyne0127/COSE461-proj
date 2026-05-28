#!/usr/bin/env python3
"""
scripts/visualize_detector_sequence.py

각 eval trial에 대해 t0 → C1 → C2 프레임에서
GDino(bbox)와 Molmo(point)가 동일한 쿼리로 객체를 어떻게 감지하는지 비교합니다.

핵심 확인 항목:
  - 객체가 사라진 STOP/ASK 케이스에서 각 탐지기가 어떻게 반응하는지
  - GDino가 없는 객체를 억지로 매핑하는지 여부
  - Molmo가 없는 객체를 hallucinate하는지 여부

GPU 서버 실행:
    python /workspace/COSE461-proj/scripts/visualize_detector_sequence.py \
        --manifest   /tmp/manifest_gpu.jsonl \
        --out-dir    /tmp/det_sequence_vis \
        --gold-focus all          # all | stop | ask | continue

옵션:
    --gold-focus   시각화할 gold_decision 필터 (기본 all)
    --max-trials   최대 처리 trial 수 (기본 60)
    --no-molmo     Molmo 탐지 생략 (빠른 GDino 전용 모드)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ── 색상 ─────────────────────────────────────────────────────────────────────
COLORS = {
    "cup":         "#4EC9B0",
    "red box":     "#F44747",
    "box":         "#FFD700",
    "cube":        "#FF8C00",
    "yellow box":  "#ADFF2F",
}
DEFAULT_COLOR = "#DDDDDD"

GOLD_BADGE = {
    "CONTINUE": ("#4CAF50", "CONTINUE"),
    "ASK":      ("#FF9800", "ASK"),
    "STOP":     ("#F44336", "STOP"),
}

_LABEL_ENRICHMENT = {
    "cup":        "paper cup or disposable cup viewed from above",
    "box":        "cardboard box or plastic storage box",
    "red box":    "red cardboard box or red plastic box",
    "cube":       "small colored cube block",
    "yellow box": "yellow cardboard box or yellow plastic box",
}


def _col(label: str) -> str:
    return COLORS.get(label.lower().strip(), DEFAULT_COLOR)


def _font(size: int = 13):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── GDino 탐지기 ──────────────────────────────────────────────────────────────

class GDino:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base",
                 box_thr=0.30, text_thr=0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"  [GDino] loading {model_id} …", flush=True)
        self.proc  = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()
        self.model.eval()
        self.box_thr = box_thr
        self.text_thr = text_thr

    def _enrich(self, label):
        return _LABEL_ENRICHMENT.get(label.lower().strip(), label.lower().strip())

    def _match(self, detected, objects):
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

    @staticmethod
    def _iou(b1, b2):
        x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
        x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
        inter = max(0,x2-x1)*max(0,y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/max(a1+a2-inter,1e-6)

    def _nms(self, dets, iou_thr=0.5):
        keep=[]
        for d in dets:
            if not any(self._iou(d["box"],k["box"])>iou_thr for k in keep):
                keep.append(d)
        return keep

    def _cross_nms(self, buckets, iou_thr=0.5):
        all_dets = [(l,d) for l,ds in buckets.items() for d in ds]
        all_dets.sort(key=lambda x: x[1]["score"], reverse=True)
        kept=[]
        for l,d in all_dets:
            if not any(kl!=l and self._iou(d["box"],kd["box"])>iou_thr
                       for kl,kd in kept):
                kept.append((l,d))
        result={l:[] for l in buckets}
        for l,d in kept: result[l].append(d)
        return result

    def detect(self, pil_img, objects):
        qmap = {o: self._enrich(o) for o in objects}
        enriched = list(qmap.values())
        rev = {v:k for k,v in qmap.items()}
        text = ". ".join(enriched) + "."
        W,H = pil_img.size
        inp = self.proc(images=pil_img, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids,
            box_threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(H,W)]
        )[0]
        buckets = {o:[] for o in objects}
        for box,score,label in zip(res["boxes"],res["scores"],res["labels"]):
            lbl = label.lower().strip() if isinstance(label,str) else str(label)
            m = self._match(lbl, enriched)
            if m is None: continue
            orig = rev.get(m, m)
            x1,y1,x2,y2 = box.tolist()
            buckets[orig].append({"box":[x1,y1,x2,y2],"score":float(score),"label":orig})
        buckets = {o: self._nms(sorted(v,key=lambda d:d["score"],reverse=True))
                   for o,v in buckets.items()}
        return self._cross_nms(buckets)


# ── Molmo 탐지기 ──────────────────────────────────────────────────────────────

class MolmoDetector:
    def __init__(self, handler):
        self._handler = handler

    def detect(self, pil_img: Image.Image, objects: List[str], session_id: str) -> Dict[str, list]:
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        self._handler.handle("reset", {}, [], session_id)
        result = self._handler.handle("detect", {"objects": objects}, [arr], session_id)
        if not result.get("success"):
            return {o: [] for o in objects}
        return result.get("detections", {o: [] for o in objects})


# ── 한 프레임 그리기 ──────────────────────────────────────────────────────────

def draw_gdino_frame(pil_img: Image.Image, dets: Dict[str, list], objects: List[str],
                     frame_label: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _font(13)

    for obj in objects:
        col = _col(obj)
        for d in dets.get(obj, []):
            x1,y1,x2,y2 = d["box"]
            draw.rectangle([x1,y1,x2,y2], outline=col, width=3)
            tag = f"{obj} {d['score']:.2f}"
            tw = draw.textlength(tag, font=font) if hasattr(draw,"textlength") else len(tag)*7
            draw.rectangle([x1, y1-20, x1+tw+6, y1], fill=col)
            draw.text((x1+3, y1-19), tag, fill="black", font=font)

    # 탐지 못한 객체 표시
    missing = [o for o in objects if not dets.get(o)]
    if missing:
        miss_txt = "NOT FOUND: " + ", ".join(missing)
        draw.rectangle([4, 4, len(miss_txt)*7+8, 24], fill="#CC0000")
        draw.text((6, 5), miss_txt, fill="white", font=_font(11))

    # 프레임 레이블
    draw.rectangle([0, img.height-22, img.width, img.height], fill=(30,30,30))
    draw.text((6, img.height-20), f"GDino | {frame_label}", fill="white", font=_font(11))
    return img


def draw_molmo_frame(pil_img: Image.Image, dets: Dict[str, list], objects: List[str],
                     frame_label: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _font(13)
    W, H = img.size
    r = max(6, W // 60)

    for obj in objects:
        col = _col(obj)
        pts = dets.get(obj, [])
        for pt in pts:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                continue
            x, y = int(pt[0]), int(pt[1])
            # 원 + 십자
            draw.ellipse([x-r*2, y-r*2, x+r*2, y+r*2], outline=col, width=3)
            draw.ellipse([x-r//2, y-r//2, x+r//2, y+r//2], fill=col)
            draw.line([x-r*3, y, x+r*3, y], fill=col, width=2)
            draw.line([x, y-r*3, x, y+r*3], fill=col, width=2)
            tag = f"{obj}"
            draw.text((x+r*2+3, y-r), tag, fill=col, font=font)

    missing = [o for o in objects if not dets.get(o)]
    if missing:
        miss_txt = "NOT FOUND: " + ", ".join(missing)
        draw.rectangle([4, 4, len(miss_txt)*7+8, 24], fill="#CC0000")
        draw.text((6, 5), miss_txt, fill="white", font=_font(11))

    draw.rectangle([0, img.height-22, img.width, img.height], fill=(30,30,30))
    draw.text((6, img.height-20), f"Molmo | {frame_label}", fill="white", font=_font(11))
    return img


# ── 6-패널 조합 ──────────────────────────────────────────────────────────────

def make_trial_panel(
    trial_id: str,
    task: str,
    gold_decision: str,
    gold_state: str,
    checkpoint: str,
    frames: List[Tuple[str, Image.Image]],         # [(label, pil), ...]
    gdino_dets: List[Dict[str, list]],              # per frame
    molmo_dets: Optional[List[Dict[str, list]]],    # None if --no-molmo
    objects: List[str],
) -> Image.Image:
    """
    레이아웃 (6칸 또는 3칸):
      상단: GDino row  → t0 | C1 | C2
      하단: Molmo row  → t0 | C1 | C2  (no-molmo 시 생략)
    """
    W, H = frames[0][1].size
    n_cols = len(frames)
    n_rows = 2 if molmo_dets else 1
    HEADER = 40

    canvas = Image.new("RGB", (W * n_cols, H * n_rows + HEADER), (20, 20, 20))
    d = ImageDraw.Draw(canvas)

    # 헤더
    badge_col, badge_txt = GOLD_BADGE.get(gold_decision, ("#888888", gold_decision))
    header = f"[{trial_id}]  {task}  |  ckpt:{checkpoint}  |  gold:{gold_state}"
    d.rectangle([0, 0, W*n_cols, HEADER], fill=(30, 30, 30))
    d.rectangle([4, 6, 90, 34], fill=badge_col)
    d.text((8, 10), badge_txt, fill="white", font=_font(14))
    d.text((96, 10), header, fill="#CCCCCC", font=_font(12))

    # GDino 행
    for ci, (frame_label, pil) in enumerate(frames):
        img = draw_gdino_frame(pil, gdino_dets[ci], objects, frame_label)
        canvas.paste(img, (ci * W, HEADER))

    # Molmo 행
    if molmo_dets:
        for ci, (frame_label, pil) in enumerate(frames):
            img = draw_molmo_frame(pil, molmo_dets[ci], objects, frame_label)
            canvas.paste(img, (ci * W, HEADER + H))

    return canvas


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    default="/tmp/manifest_gpu.jsonl")
    parser.add_argument("--out-dir",     default="/tmp/det_sequence_vis")
    parser.add_argument("--gold-focus",  default="all",
                        choices=["all", "stop", "ask", "continue"])
    parser.add_argument("--max-trials",  type=int, default=60)
    parser.add_argument("--no-molmo",    action="store_true")
    parser.add_argument("--model-type",  default="fs_prompt")
    parser.add_argument("--adapter-ckpt",default="")
    parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-thr",     type=float, default=0.30)
    parser.add_argument("--text-thr",    type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.gold_focus != "all":
        samples = [s for s in samples if s["gold_decision"].upper() == args.gold_focus.upper()]
    samples = samples[:args.max_trials]
    print(f"\n총 {len(samples)}개 trial  (filter={args.gold_focus})\n")

    # 모델 로드
    gdino = GDino(args.gdino_model, args.box_thr, args.text_thr)

    molmo_detector = None
    if not args.no_molmo:
        print("  [Molmo] AmbResHandler 로드 중 …", flush=True)
        import os, sys
        repo_root   = Path("/workspace/COSE461-proj")
        ambres_root = Path("/workspace/AmbRes")
        for p in [str(repo_root/"src"), str(repo_root), str(ambres_root)]:
            if p not in sys.path: sys.path.insert(0, p)
        from module.models.ambres.handler import AmbResHandler
        handler = AmbResHandler()
        handler.setup({
            "model_type":  args.model_type,
            "adapter_ckpt": args.adapter_ckpt,
            "use_detection": False,
            "use_sam": False,
            "use_gdino": False,   # Molmo만 사용
        })
        molmo_detector = MolmoDetector(handler)
        print("  [Molmo] 준비 완료\n", flush=True)

    summary_rows = []

    for idx, sample in enumerate(samples, 1):
        sid      = sample["id"]
        task     = sample["task"]
        gold_dec = sample["gold_decision"]
        gold_st  = sample["gold_state"]
        ckpt     = sample["checkpoint"]
        objects  = list(dict.fromkeys([sample["target_label"], sample["destination_label"]]))

        t0_path = Path(sample["initial_img"])
        c1_path = Path(sample["c1_img"])
        c2_path = Path(sample["c2_img"])

        for p in [t0_path, c1_path, c2_path]:
            if not p.exists():
                print(f"  [skip] {sid}: 이미지 없음 {p}")
                continue

        pil_t0 = Image.open(t0_path).convert("RGB")
        pil_c1 = Image.open(c1_path).convert("RGB")
        pil_c2 = Image.open(c2_path).convert("RGB")
        frames = [("t0", pil_t0), ("C1", pil_c1), ("C2", pil_c2)]

        print(f"  [{idx:02d}/{len(samples)}] {sid}  gold={gold_dec}  objects={objects}", flush=True)

        # GDino 탐지 (t0, C1, C2)
        gdino_dets = []
        for frame_label, pil in frames:
            det = gdino.detect(pil, objects)
            found = {o: len(det.get(o, [])) for o in objects}
            print(f"         GDino {frame_label}: {found}", flush=True)
            gdino_dets.append(det)

        # Molmo 탐지 (t0, C1, C2)
        molmo_dets = None
        if molmo_detector:
            molmo_dets = []
            for fi, (frame_label, pil) in enumerate(frames):
                det = molmo_detector.detect(pil, objects, session_id=f"{sid}_molmo_{frame_label}")
                found = {o: len(det.get(o, [])) for o in objects}
                print(f"         Molmo {frame_label}: {found}", flush=True)
                molmo_dets.append(det)

        # 패널 생성 및 저장
        panel = make_trial_panel(
            sid, task, gold_dec, gold_st, ckpt,
            frames, gdino_dets, molmo_dets, objects
        )
        save_path = out_dir / f"{sid}__{gold_dec}.png"
        panel.save(save_path)

        # 요약 수집
        row = {"id": sid, "gold_decision": gold_dec, "gold_state": gold_st, "objects": objects}
        for fi, (fl, _) in enumerate(frames):
            for o in objects:
                row[f"gdino_{fl}_{o}"] = len(gdino_dets[fi].get(o, []))
                if molmo_dets:
                    row[f"molmo_{fl}_{o}"] = len(molmo_dets[fi].get(o, []))
        summary_rows.append(row)

    # 요약 JSON 저장
    summary_path = out_dir / "sequence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    # 핵심 문제 케이스 출력: STOP/ASK인데 탐지기가 없는 물체를 검출한 경우
    print(f"\n{'═'*65}")
    print("  Force-mapping 의심 케이스 (gold=STOP or ASK, 탐지기가 C1/C2에서 검출)")
    print(f"{'═'*65}")
    for r in summary_rows:
        if r["gold_decision"] not in ("STOP", "ASK"):
            continue
        for o in r["objects"]:
            t0_g = r.get(f"gdino_t0_{o}", 0)
            c1_g = r.get(f"gdino_C1_{o}", 0)
            c2_g = r.get(f"gdino_C2_{o}", 0)
            # t0에 있다가 checkpoint에도 있으면 문제 아님
            # t0에 없다가 checkpoint에 나타나거나 / t0에 있다가 사라지면 주목
            if t0_g > 0 and (c1_g == 0 or c2_g == 0):
                print(f"  {r['id']:22s}  [{r['gold_decision']}]  '{o}'  t0:{t0_g} C1:{c1_g} C2:{c2_g}  → 사라짐")
            elif t0_g == 0 and (c1_g > 0 or c2_g > 0):
                print(f"  {r['id']:22s}  [{r['gold_decision']}]  '{o}'  t0:{t0_g} C1:{c1_g} C2:{c2_g}  ⚠ force-map 의심")

    print(f"\n시각화: {out_dir}/*.png")
    print(f"요약:   {summary_path}")


if __name__ == "__main__":
    main()
