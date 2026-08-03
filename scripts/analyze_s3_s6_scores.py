#!/usr/bin/env python3
"""
Analyze GDino detection confidence scores for S3 vs S6 c1 images.

S3: cup removed → GDino false-positive, expected LOW confidence
S6: cup moved   → GDino genuine detection, expected HIGH confidence

Run on the server:
  python scripts/analyze_s3_s6_scores.py --manifest dataset/manifest_eval.jsonl

Output shows per-trial: dist_from_g0, best_gdino_score, verdict
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_REPO = _SRC.parent
_AMBRES = _REPO.parent / "AmbRes"
for _p in (_SRC, _REPO, _AMBRES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_g0_with_score(initial_img: str, target_label: str, dst_label: str, gdino) -> tuple[dict, float | None]:
    """Returns (g0_dict, t0_best_score_for_target)."""
    from PIL import Image

    pil = Image.open(initial_img).convert("RGB")
    W, H = pil.size

    # detect_with_scores to capture t0 confidence
    scored = gdino.detect_with_scores(pil, [target_label, dst_label])
    tgt_scored = scored.get(target_label, [])
    t0_score = max((d["score"] for d in tgt_scored), default=None)

    def _first(label):
        entries = scored.get(label, [])
        valid = [d for d in entries if isinstance(d.get("coord"), (list, tuple)) and len(d["coord"]) == 2]
        if not valid:
            return None
        best = max(valid, key=lambda d: d["score"])
        c = best["coord"]
        return [int(c[0]), int(c[1])]

    g0 = {
        "target":      {"label": target_label, "coord": _first(target_label)},
        "destination": {"label": dst_label,     "coord": _first(dst_label)},
        "image_shape": [H, W],
    }
    return g0, t0_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset/manifest_eval.jsonl")
    parser.add_argument("--scenarios", nargs="+", default=["S3", "S6"])
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--path-old", default="", help="Path prefix to replace in manifest image paths")
    parser.add_argument("--path-new", default="", help="Replacement path prefix")
    args = parser.parse_args()

    def _remap(p: str) -> str:
        if args.path_old and p.startswith(args.path_old):
            return args.path_new + p[len(args.path_old):]
        return p

    from module.models.ambres.gdino import GroundingDINO

    print("Loading GDino …")
    gdino = GroundingDINO(model_id=args.model_id, device="cuda")

    samples = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("scenario") in args.scenarios:
                samples.append(d)

    print(f"\nAnalyzing {len(samples)} samples from scenarios {args.scenarios}\n")
    print(f"{'ID':<20} {'S':<3} {'t0_sc':>6} {'c1_sc':>6} {'delta':>6} {'dist':>6}  note")
    print("-" * 75)

    delta_by_scenario: dict[str, list[float]] = {}
    t0_score_by_scenario: dict[str, list[float]] = {}
    c1_score_by_scenario: dict[str, list[float]] = {}

    from PIL import Image

    for sample in samples:
        sid     = sample["id"]
        scen    = sample["scenario"]
        ck      = sample["checkpoint"]
        tgt     = sample["target_label"]
        dst     = sample["destination_label"]
        t0_img  = _remap(sample["initial_img"])
        ck_img  = _remap(sample["c1_img"] if ck == "C1" else sample["c2_img"])

        try:
            # Build g0 + t0 score
            g0, t0_score = _load_g0_with_score(t0_img, tgt, dst, gdino)
            g0_coord = g0["target"]["coord"] if ck == "C1" else g0["destination"]["coord"]

            # Detect at checkpoint with scores
            pil_ck = Image.open(ck_img).convert("RGB")
            role_label = tgt if ck == "C1" else dst
            ck_scored = gdino.detect_with_scores(pil_ck, [role_label]).get(role_label, [])

            if not ck_scored:
                t0_str = f"{t0_score:.3f}" if t0_score is not None else "—"
                print(f"{sid:<20} {scen:<3} {t0_str:>6} {'—':>6} {'—':>6} {'—':>6}  no C1 detection")
                continue

            best_ck = max(ck_scored, key=lambda d: d["score"])
            coord_ck = best_ck["coord"]
            c1_score = best_ck["score"]

            dist = (
                math.sqrt(
                    (g0_coord[0] - coord_ck[0]) ** 2
                    + (g0_coord[1] - coord_ck[1]) ** 2
                )
                if g0_coord else float("nan")
            )

            delta = (t0_score - c1_score) if t0_score is not None else float("nan")
            n_det = len(ck_scored)
            note = f"n_det={n_det}"

            t0_str = f"{t0_score:.3f}" if t0_score is not None else "N/A"
            print(
                f"{sid:<20} {scen:<3} {t0_str:>6} {c1_score:>6.3f} {delta:>6.3f} {dist:>6.0f}  {note}"
            )

            delta_by_scenario.setdefault(scen, []).append(delta)
            t0_score_by_scenario.setdefault(scen, []).append(t0_score or 0)
            c1_score_by_scenario.setdefault(scen, []).append(c1_score)

        except Exception as exc:
            print(f"{sid:<20} ERROR: {exc}")

    print("\n--- Summary by scenario ---")
    for scen in args.scenarios:
        t0s   = t0_score_by_scenario.get(scen, [])
        c1s   = c1_score_by_scenario.get(scen, [])
        deltas = delta_by_scenario.get(scen, [])
        if deltas:
            print(
                f"  {scen}: n={len(deltas)}"
                f"  t0_score mean={sum(t0s)/len(t0s):.3f}"
                f"  c1_score mean={sum(c1s)/len(c1s):.3f}"
                f"  delta mean={sum(deltas)/len(deltas):.3f}"
                f"  delta range=[{min(deltas):.3f}, {max(deltas):.3f}]"
            )


if __name__ == "__main__":
    main()
