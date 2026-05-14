"""Pilot experiment to determine the THRESHOLD for consistency_monitor.py.

Goal:
  Measure how much AmbRes detect output varies for the same object across
  repeated captures of the same scene (intra-scene noise), then set
  THRESHOLD to a safe multiple of that noise level.

Two measurement modes:
  1. Intra-scene noise  — detect the same object N times in the same image
                          (simulates real-world measurement noise / quantisation).
  2. Inter-instance distance — detect an image that contains multiple instances
                                of the same label; report minimum pairwise distance
                                so we can confirm THRESHOLD << that gap.

Output:
  JSON report with per-object statistics and a final recommended threshold.
  The recommended value should be written to consistency_monitor.py's default.

Usage examples:
  # Mock mode (no GPU): verify statistics logic
  python pilot_threshold.py --mock --n-trials 10 --objects "cup" "box"

  # Real mode: single image, repeat detection 5 times
  python pilot_threshold.py --images real_0.png --objects "blue cup" \\
      --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm --n-trials 5

  # Real mode: multiple images of same scene
  python pilot_threshold.py --images c1_a.png c1_b.png c1_c.png \\
      --objects "red block" "tray" --model-type finetune \\
      --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT   = Path(__file__).resolve().parent.parent  # COSE461-proj/src
REPO_ROOT   = _SRC_ROOT.parent                        # COSE461-proj
AMBRES_ROOT = REPO_ROOT.parent / "AmbRes"             # workspace/AmbRes
for _p in (_SRC_ROOT, REPO_ROOT, AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Statistics helpers (pure — no AmbRes dependency, fully testable)
# ---------------------------------------------------------------------------

def centroid(coords: list[list[int]]) -> list[float]:
    """Compute mean [x, y] of a list of coordinates."""
    if not coords:
        raise ValueError("coords must be non-empty")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of values using nearest-rank."""
    if not values:
        raise ValueError("values must be non-empty")
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def compute_intra_stats(
    coords_per_label: dict[str, list[list[int]]],
    p: float = 95.0,
) -> dict[str, dict[str, float]]:
    """Compute spread statistics for each label across multiple detections.

    Args:
        coords_per_label: {"label": [[x,y], [x,y], ...]} — all detections
                          across trials/images for that label.
        p: Percentile for spread summary (default 95).

    Returns:
        {"label": {"n": int, "centroid": [x, y],
                   "distances": [d, ...], "p95": float, "max": float}}
    """
    stats: dict[str, dict[str, Any]] = {}
    for label, coords in coords_per_label.items():
        if not coords:
            continue
        c = centroid(coords)
        distances = [euclidean(coord, c) for coord in coords]
        stats[label] = {
            "n": len(coords),
            "centroid": [round(c[0], 1), round(c[1], 1)],
            "distances": [round(d, 2) for d in distances],
            f"p{int(p)}": round(percentile(distances, p), 2),
            "max": round(max(distances), 2),
        }
    return stats


def compute_inter_instance_distances(
    multi_coords: dict[str, list[list[int]]],
) -> dict[str, float | None]:
    """Compute minimum pairwise distance between instances of the same label.

    Args:
        multi_coords: {"label": [[x,y], [x,y], ...]} where len > 1 indicates
                      multiple instances were detected.

    Returns:
        {"label": min_distance or None if only one instance}
    """
    result: dict[str, float | None] = {}
    for label, coords in multi_coords.items():
        if len(coords) < 2:
            result[label] = None
            continue
        min_dist = float("inf")
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = euclidean(coords[i], coords[j])
                min_dist = min(min_dist, d)
        result[label] = round(min_dist, 2)
    return result


def recommend_threshold(
    intra_stats: dict[str, dict[str, Any]],
    safety_factor: float = 3.0,
    p_key: str = "p95",
) -> float:
    """Return recommended THRESHOLD = max(p95 across all labels) × safety_factor.

    The safety_factor (default 3×) ensures the threshold is well above
    normal detection noise while remaining << inter-instance distance.

    Args:
        intra_stats: Output of compute_intra_stats().
        safety_factor: Multiplier applied to the worst-case p95.
        p_key: Which percentile key to use from intra_stats.

    Returns:
        Recommended threshold in pixels (rounded to nearest integer).
    """
    if not intra_stats:
        raise ValueError("intra_stats is empty — no data to derive threshold from")
    p_values = [s[p_key] for s in intra_stats.values() if p_key in s]
    if not p_values:
        raise ValueError(f"No '{p_key}' key found in intra_stats")
    return round(max(p_values) * safety_factor)


# ---------------------------------------------------------------------------
# AmbRes detect integration
# ---------------------------------------------------------------------------

def _collect_detections_real(
    image_paths: list[Path],
    objects: list[str],
    handler: Any,
    n_trials: int,
    session_prefix: str = "pilot",
) -> dict[str, list[list[int]]]:
    """Run AmbRes detect on each image (× n_trials for single-image case).

    For a single image, repeats detection n_trials times to measure noise.
    For multiple images, detects once per image.

    Returns:
        {"label": [[x,y], ...]} — all first-instance coords across runs.
    """
    from extraction.ambres_g0_extractor import _load_image, _unwrap_handler_result

    coords_per_label: dict[str, list[list[int]]] = {obj: [] for obj in objects}
    images_to_run = image_paths if len(image_paths) > 1 else image_paths * n_trials

    for i, img_path in enumerate(images_to_run):
        session_id = f"{session_prefix}_{i}"
        image_arr, _ = _load_image(img_path)
        _unwrap_handler_result(
            handler.handle("reset", {}, [], session_id), "reset"
        )
        result = _unwrap_handler_result(
            handler.handle("detect", {"objects": objects}, [image_arr], session_id),
            "detect",
        )
        detections = result["detections"]
        for label in objects:
            all_coords = detections.get(label) or []
            if all_coords:
                coords_per_label[label].append(all_coords[0])  # first instance only

    return coords_per_label


def _collect_detections_mock(
    objects: list[str],
    n_trials: int,
    noise_std: float = 3.0,
    seed: int = 42,
) -> dict[str, list[list[int]]]:
    """Generate synthetic detections with Gaussian noise for testing / dry-runs.

    Simulates n_trials detections per object with configurable noise level.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    coords_per_label: dict[str, list[list[int]]] = {}
    for label in objects:
        base_x = rng.randint(100, 500)
        base_y = rng.randint(100, 400)
        coords = [
            [
                int(base_x + np_rng.normal(0, noise_std)),
                int(base_y + np_rng.normal(0, noise_std)),
            ]
            for _ in range(n_trials)
        ]
        coords_per_label[label] = coords
    return coords_per_label


def _make_handler(model_type: str, adapter_ckpt: str) -> Any:
    from module.models.ambres.handler import AmbResHandler
    handler = AmbResHandler()
    handler.setup(
        {
            "model_type": model_type,
            "adapter_ckpt": adapter_ckpt,
            "use_detection": False,
            "use_sam": False,
        }
    )
    return handler


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    coords_per_label: dict[str, list[list[int]]],
    safety_factor: float = 3.0,
    p: float = 95.0,
) -> dict[str, Any]:
    """Compute full pilot report from raw detections."""
    intra = compute_intra_stats(coords_per_label, p=p)
    inter = compute_inter_instance_distances(coords_per_label)
    threshold = recommend_threshold(intra, safety_factor=safety_factor)

    return {
        "intra_scene": intra,
        "inter_instance_min_dist": inter,
        "safety_factor": safety_factor,
        "recommended_threshold_px": threshold,
        "note": (
            f"Set consistency_monitor.py threshold={threshold}. "
            f"Verify this is << inter-instance distances above."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot experiment: measure AmbRes coord variation → determine THRESHOLD."
    )
    parser.add_argument(
        "--images", nargs="+", default=[],
        help="Image paths. Single image → repeated n-trials times. Multiple → once each.",
    )
    parser.add_argument(
        "--objects", nargs="+", required=True,
        help="Object labels to detect (e.g. 'red block' 'tray').",
    )
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Repeat count when using a single image (default 5).")
    parser.add_argument("--safety-factor", type=float, default=3.0,
                        help="Multiplier on p95 to get threshold (default 3.0).")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Spread percentile to summarise (default 95).")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic detections — no GPU required.")
    parser.add_argument("--mock-noise-std", type=float, default=3.0,
                        help="Gaussian noise std for mock mode (default 3.0 px).")
    parser.add_argument("--model-type", default="fs_prompt",
                        choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this path (default: stdout).")
    args = parser.parse_args()

    if args.mock:
        coords = _collect_detections_mock(
            args.objects, args.n_trials, noise_std=args.mock_noise_std
        )
    else:
        if not args.images:
            parser.error("--images required unless --mock is set")
        if args.model_type == "finetune" and not args.adapter_ckpt:
            parser.error("--model-type finetune requires --adapter-ckpt")
        handler = _make_handler(args.model_type, args.adapter_ckpt)
        coords = _collect_detections_real(
            [Path(p) for p in args.images], args.objects, handler, args.n_trials
        )

    report = build_report(coords, safety_factor=args.safety_factor, p=args.percentile)
    out = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out)
        print(f"Report written to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
