#!/usr/bin/env python3
"""
scripts/test_gdino_eval.py

manifest_eval.jsonl 샘플에 GroundingDINO detection을 수행하고
bounding box + center point 시각화 이미지를 저장합니다.

GPU 서버에서 실행:
    python /workspace/COSE461-proj/scripts/test_gdino_eval.py
    python /workspace/COSE461-proj/scripts/test_gdino_eval.py --out-dir /tmp/gdino_eval --samples S1,S2,S3,S4,S5

기본 동작:
    - S1~S5 시나리오 각 1개씩 (trial_001) 선택
    - 각 샘플: t0(초기) + checkpoint 이미지 나란히 시각화
    - bbox (실선) + center dot + label/score 표시
    - 결과 이미지: {out_dir}/{sample_id}_vis.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw

# ── GroundingDINO (standalone, 모듈 import 불필요) ────────────────────────────

# top-down 로봇 장면 enrichment 테이블 (클래스 정의보다 먼저 선언)
_ENRICHMENT = {
    "cup":      "paper cup or disposable cup viewed from above",
    "box":      "cardboard box or plastic storage box",
    "red box":  "red cardboard box or red plastic box",
    "cube":     "small colored cube block",
    "bottle":   "bottle viewed from above",
    "mug":      "coffee mug or ceramic mug",
    "red mug":  "red coffee mug or red ceramic mug",
}

def _enrich(label: str) -> str:
    return _ENRICHMENT.get(label.lower().strip(), label.lower().strip())


class GDino:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_thr: float = 0.30,
        text_thr: float = 0.25,
        device: str = "cuda",
        enrich: bool = True,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"[GDino] loading {model_id} on {device} …", flush=True)
        self.proc  = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device   = device
        self.box_thr  = box_thr
        self.text_thr = text_thr
        self.enrich   = enrich
        print(f"[GDino] ready  enrich={enrich}", flush=True)

    def detect(
        self,
        pil_img: Image.Image,
        objects: List[str],
    ) -> Dict[str, List[Dict]]:
        """
        Returns:
            {label: [{"box": [x1,y1,x2,y2], "center": [cx,cy], "score": float}, ...]}
            confidence 내림차순 정렬, 미탐지 label은 빈 리스트
        """
        if not objects:
            return {}

        if self.enrich:
            query_map = {o: _enrich(o) for o in objects}
        else:
            query_map = {o: o.lower().strip() for o in objects}
        enriched    = list(query_map.values())
        reverse_map = {v: k for k, v in query_map.items()}

        text = ". ".join(enriched) + "."
        W, H = pil_img.size
        inputs = self.proc(images=pil_img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=[(H, W)],
        )[0]

        buckets: Dict[str, list] = {obj: [] for obj in objects}
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            lbl     = label.lower().strip() if isinstance(label, str) else str(label)
            matched = _match_label(lbl, enriched)
            if matched is None:
                continue
            original = reverse_map.get(matched, matched)
            x1, y1, x2, y2 = box.tolist()
            buckets[original].append({
                "box":    [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "score":  float(score),
            })
        return {
            k: _nms(sorted(v, key=lambda d: d["score"], reverse=True))
            for k, v in buckets.items()
        }


def _iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list, iou_thr: float = 0.5) -> list:
    """confidence 내림차순 기준 greedy NMS. 같은 레이블 내 겹치는 박스 제거."""
    keep: list = []
    for d in detections:  # 이미 score 내림차순 정렬 상태
        if not any(_iou(d["box"], k["box"]) > iou_thr for k in keep):
            keep.append(d)
    return keep


def _match_label(detected: str, objects: List[str]) -> Optional[str]:
    dl = detected.lower()
    for o in objects:
        if dl == o.lower():
            return o
    for o in objects:
        ol = o.lower()
        if dl in ol or ol in dl:
            return o
    dl_w = set(dl.split())
    best_s, best_o = 0, None
    for o in objects:
        ov = len(dl_w & set(o.lower().split()))
        if ov > best_s:
            best_s, best_o = ov, o
    return best_o if best_s > 0 else None


# ── 이미지 로드 ───────────────────────────────────────────────────────────────

def _hf_relpath(manifest_path: str) -> Optional[str]:
    """
    /home/hands/.../data-evaluation/S1/trial_001/t0.png
    → data-evaluation/S1/trial_001/t0.png
    """
    m = re.search(r'(data[-_]evaluation/.+|data[-_]training/.+|images/.+)', manifest_path)
    if m:
        return m.group(1).replace("data_evaluation", "data-evaluation")
    return None


def load_image(path: str, repo_id: str = "kyne0127/vla-evaluation") -> Optional[Image.Image]:
    """로컬 경로 우선, 없으면 HuggingFace에서 다운로드."""
    if os.path.exists(path):
        return Image.open(path).convert("RGB")

    rel = _hf_relpath(path)
    if rel:
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel)
            return Image.open(local).convert("RGB")
        except Exception as e:
            print(f"  [warn] HF download 실패 ({rel}): {e}", flush=True)
    return None


# ── 시각화 ─────────────────────────────────────────────────────────────────────

# target=파란색, destination=초록색, 기타=노란색
_COLORS = {
    "target":      ((59,  130, 246), (255, 255, 255)),
    "destination": ((34,  197,  94), (255, 255, 255)),
    "other":       ((251, 191,  36), (30,  30,  30)),
}


def _role(label: str, target: str, dest: str) -> str:
    if label == target:
        return "target"
    if label == dest:
        return "destination"
    return "other"


def annotate(
    img: Image.Image,
    detections: Dict[str, List[Dict]],
    target_label: str,
    dest_label: str,
) -> Image.Image:
    """이미지에 bbox + center dot + label/score 주석을 그립니다."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    dot_r = max(7, W // 70)    # center dot 반지름 (이미지 크기 비례)

    for label, items in detections.items():
        if not items:
            continue
        role = _role(label, target_label, dest_label)
        box_color, txt_color = _COLORS[role]

        for idx, det in enumerate(items):
            x1, y1, x2, y2 = det["box"]
            cx, cy          = det["center"]
            score           = det["score"]

            # ─ bounding box ─
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # ─ center dot ─
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=box_color, outline=(255, 255, 255), width=2,
            )

            # ─ label + score 태그 ─
            tag    = f"[{role[0].upper()}] {label}  {score:.2f}"
            char_w = 7
            tw     = len(tag) * char_w + 4
            th     = 16
            tx     = max(0, min(int(x1), W - tw))
            ty     = max(0, int(y1) - th - 2)
            draw.rectangle([tx, ty, tx + tw, ty + th], fill=box_color)
            draw.text((tx + 2, ty + 1), tag, fill=txt_color)

            # 복수 인스턴스 번호
            if len(items) > 1:
                draw.text((int(cx) + dot_r + 3, int(cy) - dot_r), f"#{idx+1}", fill=box_color)

    return out


def make_panel(sample: dict, img_t0: Image.Image, img_c: Image.Image,
               dets_t0: dict, dets_c: dict) -> Image.Image:
    """
    t0 / checkpoint 이미지를 나란히 배치하고 상단에 메타 정보를 표시합니다.
    반환: 단일 PIL Image
    """
    target  = sample["target_label"]
    dest    = sample["destination_label"]
    cp_name = sample["checkpoint"]

    ann_t0 = annotate(img_t0, dets_t0, target, dest)
    ann_c  = annotate(img_c,  dets_c,  target, dest)

    # 높이 통일 (t0 기준)
    ref_h = ann_t0.height
    if ann_c.height != ref_h:
        scale = ref_h / ann_c.height
        ann_c = ann_c.resize((int(ann_c.width * scale), ref_h), Image.LANCZOS)

    # ─ 이미지 하단 캡션 바 ─
    cap_h  = 26
    pad    = 8

    def with_caption(img: Image.Image, caption: str) -> Image.Image:
        w, h = img.size
        panel = Image.new("RGB", (w, h + cap_h), (25, 25, 25))
        panel.paste(img, (0, 0))
        d = ImageDraw.Draw(panel)
        d.text((pad, h + 5), caption, fill=(210, 210, 210))
        return panel

    t0_det_summary = _det_summary(dets_t0)
    c_det_summary  = _det_summary(dets_c)

    frame_t0 = with_caption(ann_t0, f"t0 (initial)  {t0_det_summary}")
    frame_c  = with_caption(ann_c,  f"{cp_name} (checkpoint)  {c_det_summary}")

    # ─ 두 이미지 수평 이어붙이기 ─
    gap        = 6
    total_w    = frame_t0.width + gap + frame_c.width
    frame_h    = max(frame_t0.height, frame_c.height)
    title_h    = 52
    canvas     = Image.new("RGB", (total_w, title_h + frame_h), (18, 18, 18))

    # 제목 바
    d = ImageDraw.Draw(canvas)
    state_color = {
        "CLEAR":                 (74,  222, 128),
        "AMBIGUOUS_TARGET":      (251, 191,  36),
        "AMBIGUOUS_DESTINATION": (251, 191,  36),
        "INVALID_TARGET":        (248,  113, 113),
        "INVALID_DESTINATION":   (248,  113, 113),
    }.get(sample["gold_state"], (200, 200, 200))

    d.text((pad, 6),
           f"[{sample['id']}]  {sample['task']}",
           fill=(255, 220, 80))
    d.text((pad, 26),
           f"target: {target}  |  dest: {dest}  "
           f"|  gold: {sample['gold_state']} → {sample['gold_decision']}",
           fill=state_color)

    canvas.paste(frame_t0, (0, title_h))
    canvas.paste(frame_c,  (frame_t0.width + gap, title_h))
    return canvas


def _det_summary(dets: dict) -> str:
    """'cup×1, red box×2' 형태 요약."""
    parts = []
    for label, items in dets.items():
        if items:
            parts.append(f"{label}×{len(items)}")
    return ", ".join(parts) if parts else "(미탐지)"


# ── 샘플 선택 ─────────────────────────────────────────────────────────────────

def select_samples(manifest_path: str, scenarios: List[str]) -> List[dict]:
    """
    각 시나리오별로 trial_001을 선택합니다.
    없으면 해당 시나리오의 첫 번째 항목을 선택합니다.
    """
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    chosen: Dict[str, dict] = {}
    for entry in entries:
        s = entry.get("scenario", "")
        if s not in scenarios:
            continue
        if s not in chosen:
            chosen[s] = entry
        elif entry["id"].endswith("trial_001"):
            chosen[s] = entry

    return [chosen[s] for s in scenarios if s in chosen]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GroundingDINO eval visualizer")
    parser.add_argument("--manifest",  default=None,
                        help="manifest_eval.jsonl 경로 (기본: 자동 탐색)")
    parser.add_argument("--out-dir",   default=None,
                        help="결과 저장 디렉토리 (기본: <프로젝트>/eval_outputs/gdino_viz)")
    parser.add_argument("--samples",   default="S1,S2,S3,S4,S5",
                        help="시나리오 목록 (쉼표 구분, 기본: S1,S2,S3,S4,S5)")
    parser.add_argument("--model",     default="IDEA-Research/grounding-dino-base",
                        help="GroundingDINO 모델 ID")
    parser.add_argument("--box-thr",   type=float, default=0.30)
    parser.add_argument("--text-thr",  type=float, default=0.25)
    parser.add_argument("--no-enrich", action="store_true",
                        help="label enrichment 비활성화 (raw label로 쿼리)")
    parser.add_argument("--repo-id",   default="kyne0127/vla-evaluation",
                        help="HuggingFace dataset repo ID")
    args = parser.parse_args()

    # ─ 경로 결정 ─
    proj_root = Path(__file__).resolve().parent.parent
    manifest  = args.manifest or str(proj_root / "dataset" / "manifest_eval.jsonl")
    out_dir   = Path(args.out_dir) if args.out_dir else proj_root / "eval_outputs" / "gdino_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [s.strip() for s in args.samples.split(",")]

    # ─ 모델 로드 ─
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdino  = GDino(model_id=args.model, box_thr=args.box_thr,
                   text_thr=args.text_thr, device=device,
                   enrich=not args.no_enrich)

    # ─ 샘플 선택 ─
    samples = select_samples(manifest, scenarios)
    if not samples:
        print("[error] 선택된 샘플이 없습니다. manifest 경로를 확인하세요:", manifest)
        sys.exit(1)
    print(f"\n총 {len(samples)}개 샘플 처리 시작\n{'─'*60}", flush=True)

    # ─ 각 샘플 처리 ─
    results_log = []
    for sample in samples:
        sid  = sample["id"]
        task = sample["task"]
        tgt  = sample["target_label"]
        dst  = sample["destination_label"]
        chk  = sample["checkpoint"]  # "C1" or "C2"

        cp_img_key = "c1_img" if chk == "C1" else "c2_img"

        print(f"[{sid}]  {task}", flush=True)
        print(f"  target={tgt}  dest={dst}  checkpoint={chk}", flush=True)
        print(f"  gold: {sample['gold_state']} → {sample['gold_decision']}", flush=True)

        img_t0 = load_image(sample["initial_img"], args.repo_id)
        img_c  = load_image(sample[cp_img_key],    args.repo_id)

        if img_t0 is None or img_c is None:
            print(f"  [skip] 이미지 로드 실패", flush=True)
            continue

        objects = [tgt, dst]

        print(f"  detecting on t0 … ", end="", flush=True)
        dets_t0 = gdino.detect(img_t0, objects)
        _print_dets(dets_t0)

        print(f"  detecting on {chk} … ", end="", flush=True)
        dets_c = gdino.detect(img_c, objects)
        _print_dets(dets_c)

        panel = make_panel(sample, img_t0, img_c, dets_t0, dets_c)
        out_path = out_dir / f"{sid}_vis.png"
        panel.save(str(out_path))
        print(f"  saved → {out_path}", flush=True)

        results_log.append({
            "id":         sid,
            "scenario":   sample["scenario"],
            "gold_state": sample["gold_state"],
            "t0_dets":    {k: len(v) for k, v in dets_t0.items()},
            "cp_dets":    {k: len(v) for k, v in dets_c.items()},
        })
        print(flush=True)

    # ─ 요약 로그 ─
    log_path = out_dir / "detection_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    print(f"{'─'*60}")
    print(f"완료: {len(results_log)}/{len(samples)}개 성공")
    print(f"결과 디렉토리: {out_dir}")
    print(f"요약 로그: {log_path}")

    # 각 샘플별 탐지 요약 출력
    print(f"\n{'─'*60}")
    print(f"{'ID':<25} {'Scenario':<6} {'Gold State':<28} {'t0 dets':<25} {'CP dets'}")
    print(f"{'─'*60}")
    for r in results_log:
        t0_s = ", ".join(f"{k}:{v}" for k, v in r["t0_dets"].items())
        cp_s = ", ".join(f"{k}:{v}" for k, v in r["cp_dets"].items())
        print(f"{r['id']:<25} {r['scenario']:<6} {r['gold_state']:<28} {t0_s:<25} {cp_s}")


def _print_dets(dets: Dict[str, List[Dict]]) -> None:
    parts = []
    for label, items in dets.items():
        if items:
            scores = ", ".join(f"{d['score']:.2f}" for d in items)
            parts.append(f"{label}×{len(items)}({scores})")
        else:
            parts.append(f"{label}×0")
    print(", ".join(parts), flush=True)


if __name__ == "__main__":
    main()
