"""B4 baseline: binary grounding anomaly detector.

B4 compares stored G0 coordinates against checkpoint detections, but it does
not classify grounding states. Any anomaly is collapsed to Decision.ASK.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.common import BaselineResult
from monitoring.consistency_monitor import Decision, get_checkpoint_detections


METHOD_NAME = "B4_BINARY_ANOMALY"


def _valid_coords(raw: Any) -> list[list[float]]:
    return [
        [float(c[0]), float(c[1])]
        for c in (raw or [])
        if isinstance(c, (list, tuple)) and len(c) == 2
    ]


def _euclidean(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _role_valid(
    g0: dict[str, Any],
    detections_t: dict[str, list[list[int]]],
    role: str,
    threshold: float,
) -> tuple[bool, str, dict[str, Any]]:
    label = g0[role]["label"]
    g0_coord = g0[role]["coord"]
    coords = _valid_coords(detections_t.get(label))

    detail = {
        "role": role,
        "label": label,
        "g0_coord": g0_coord,
        "coords": coords,
        "n_coords": len(coords),
        "min_distance": None,
    }

    if len(coords) == 0:
        return False, f"{role} '{label}' not found", detail
    if len(coords) > 1:
        distances = [_euclidean(g0_coord, c) for c in coords]
        detail["min_distance"] = min(distances)
        return False, f"multiple {role} candidates for '{label}'", detail

    dist = _euclidean(g0_coord, coords[0])
    detail["min_distance"] = dist
    if dist > threshold:
        return False, f"{role} '{label}' moved beyond threshold", detail

    return True, "", detail


def run_b4_binary_anomaly(
    g0: dict[str, Any],
    checkpoint_img: str | Path,
    *,
    checkpoint: str = "C1",
    handler: Any,
    threshold: float = 50.0,
    session_id: str = "b4_checkpoint",
) -> BaselineResult:
    """Run B4 at one checkpoint using G0 coordinate validity only.

    Args:
        g0: Stored initial grounding with target/destination label+coord.
        checkpoint_img: Checkpoint image, e.g. C1 or C2 snapshot.
        checkpoint: Checkpoint label stored in metadata ("C1" or "C2").
        handler: Pre-built AmbResHandler. Required to avoid hidden model loads.
        threshold: Pixel distance threshold for binary validity.
        session_id: AmbRes session identifier.

    Returns:
        BaselineResult. Any target/destination anomaly returns Decision.ASK.
        Otherwise returns Decision.CONTINUE.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")
    if handler is None:
        raise ValueError("handler is required for B4 checkpoint detection")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    detections_t = get_checkpoint_detections(
        checkpoint_img,
        g0,
        handler,
        session_id=session_id,
    )

    target_valid, target_reason, target_detail = _role_valid(
        g0, detections_t, "target", threshold
    )
    dest_valid, dest_reason, dest_detail = _role_valid(
        g0, detections_t, "destination", threshold
    )

    reasons = [r for r in (target_reason, dest_reason) if r]
    metadata = {
        "checkpoint": checkpoint,
        "session_id": session_id,
        "uses_checkpoint": True,
        "stores_g0": True,
        "uses_coord": True,
        "uses_taxonomy": False,
        "threshold": threshold,
        "target_valid": target_valid,
        "destination_valid": dest_valid,
        "target_detail": target_detail,
        "destination_detail": dest_detail,
    }

    if reasons:
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason="Grounding anomaly detected: " + "; ".join(reasons),
            question="Grounding anomaly detected. Please clarify the task grounding.",
            raw_output={"detections": detections_t},
            metadata=metadata,
        )

    return BaselineResult(
        method=METHOD_NAME,
        decision=Decision.CONTINUE,
        reason="No binary grounding anomaly detected",
        raw_output={"detections": detections_t},
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run B4 binary anomaly baseline using precomputed detections JSON. "
            "This CLI avoids loading AmbRes; programmatic API uses handler+image."
        )
    )
    parser.add_argument("g0_json")
    parser.add_argument("detections_json")
    parser.add_argument("--checkpoint", default="C1", choices=["C1", "C2"])
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()

    with open(args.g0_json) as f:
        g0 = json.load(f)
    with open(args.detections_json) as f:
        detections_t = json.load(f)

    target_valid, target_reason, target_detail = _role_valid(
        g0, detections_t, "target", args.threshold
    )
    dest_valid, dest_reason, dest_detail = _role_valid(
        g0, detections_t, "destination", args.threshold
    )
    reasons = [r for r in (target_reason, dest_reason) if r]
    result = BaselineResult(
        method=METHOD_NAME,
        decision=Decision.ASK if reasons else Decision.CONTINUE,
        reason=(
            "Grounding anomaly detected: " + "; ".join(reasons)
            if reasons
            else "No binary grounding anomaly detected"
        ),
        question=(
            "Grounding anomaly detected. Please clarify the task grounding."
            if reasons
            else ""
        ),
        raw_output={"detections": detections_t},
        metadata={
            "checkpoint": args.checkpoint,
            "uses_checkpoint": True,
            "stores_g0": True,
            "uses_coord": True,
            "uses_taxonomy": False,
            "threshold": args.threshold,
            "target_valid": target_valid,
            "destination_valid": dest_valid,
            "target_detail": target_detail,
            "destination_detail": dest_detail,
        },
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
