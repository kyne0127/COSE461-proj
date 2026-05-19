from __future__ import annotations

import argparse
import json
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GroundingState(str, Enum):
    CLEAR = "CLEAR"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    INVALID_TARGET = "INVALID_TARGET"
    AMBIGUOUS_DESTINATION = "AMBIGUOUS_DESTINATION"
    INVALID_DESTINATION = "INVALID_DESTINATION"
    UNSAFE_OR_BLOCKED = "UNSAFE_OR_BLOCKED"
    # OCCUPIED_DESTINATION: out of current scope (Phase 2+)


class Decision(str, Enum):
    CONTINUE = "CONTINUE"
    ASK = "ASK"
    STOP = "STOP"


# proposal §6.1: "Conservative (INVALID만 STOP), 나머지는 ASK → 사용자 판단 위임"
# INVALID_DESTINATION → ASK: destination 소실은 user input으로 회복 가능
_STATE_TO_DECISION: dict[GroundingState, Decision] = {
    GroundingState.CLEAR: Decision.CONTINUE,
    GroundingState.AMBIGUOUS_TARGET: Decision.ASK,
    GroundingState.INVALID_TARGET: Decision.STOP,
    GroundingState.AMBIGUOUS_DESTINATION: Decision.ASK,
    GroundingState.INVALID_DESTINATION: Decision.ASK,
    GroundingState.UNSAFE_OR_BLOCKED: Decision.STOP,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _euclidean(a: list[int], b: list[int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _relative_dist(
    g0_primary: list[int],
    g0_anchor: list[int],
    gt_primary: list[int],
    gt_anchor: list[int],
) -> float:
    """Change in the (primary − anchor) relative vector between t₀ and Gₜ.

    By comparing how the primary object is positioned relative to a stable
    anchor (destination at C1, target at C2), camera-induced coordinate shifts
    are cancelled out, leaving only genuine object motion.

    Caller is responsible for supplying a valid (non-empty) gt_anchor.
    """
    rel_g0 = [g0_primary[0] - g0_anchor[0], g0_primary[1] - g0_anchor[1]]
    rel_gt = [gt_primary[0]  - gt_anchor[0],  gt_primary[1]  - gt_anchor[1]]
    return math.sqrt((rel_g0[0] - rel_gt[0]) ** 2 + (rel_g0[1] - rel_gt[1]) ** 2)


def _check_target(
    g0: dict[str, Any],
    detections_t: dict[str, list[list[int]]],
    threshold: float,
    rel_threshold: float | None,
) -> GroundingState:
    label    = g0["target"]["label"]
    g0_coord = g0["target"]["coord"]
    # Molmo가 객체를 찾지 못할 때 [[]] 같은 빈 좌표를 반환할 수 있으므로
    # len==2 인 유효한 좌표만 남긴다
    raw    = detections_t.get(label) or []
    coords = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]

    if not coords:
        return GroundingState.INVALID_TARGET

    # Multiple instances → cannot tell which is the original even with coord memory.
    # (This is distinct from CLEAR(distractor): when Molmo labels TWO objects with the
    # same name, the scene is genuinely ambiguous from the robot's perspective.)
    if len(coords) > 1:
        return GroundingState.AMBIGUOUS_TARGET

    # Single instance: check absolute distance first (fast path for stationary cameras).
    if _euclidean(g0_coord, coords[0]) <= threshold:
        return GroundingState.CLEAR

    # Absolute distance exceeds threshold.  This may reflect genuine object motion OR
    # camera motion (the robot moves between t₀ and C1, shifting all pixel coords).
    #
    # Heuristic: if the destination ALSO moved beyond `threshold` we infer camera motion
    # and re-evaluate using the camera-normalized relative distance
    #   Δ = ‖(target_t − dest_t) − (target_0 − dest_0)‖
    # which cancels the shared camera-induced shift.  If the destination is still near
    # its G₀ position the camera has not moved, so any target displacement is genuine.
    dest_raw = detections_t.get(g0["destination"]["label"]) or []
    valid_anchors = [c for c in dest_raw if isinstance(c, (list, tuple)) and len(c) == 2]
    if valid_anchors:
        gt_anchor = min(valid_anchors, key=lambda c: _euclidean(g0["destination"]["coord"], c))
        dest_abs_dist = _euclidean(g0["destination"]["coord"], gt_anchor)
        if dest_abs_dist > threshold:
            # Destination also displaced → camera moved → camera-normalized check
            _rel = rel_threshold if rel_threshold is not None else threshold * 5.0
            rel = _relative_dist(g0_coord, g0["destination"]["coord"], coords[0], gt_anchor)
            if rel <= _rel:
                return GroundingState.CLEAR

    # Target displaced and camera-motion explanation does not apply → different instance
    return GroundingState.AMBIGUOUS_TARGET


def _check_destination(
    g0: dict[str, Any],
    detections_t: dict[str, list[list[int]]],
    threshold: float,
    rel_threshold: float | None,
) -> GroundingState:
    label    = g0["destination"]["label"]
    g0_coord = g0["destination"]["coord"]
    raw    = detections_t.get(label) or []
    coords = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]

    if not coords:
        return GroundingState.INVALID_DESTINATION

    # Multiple candidates → ambiguous which to place into
    if len(coords) > 1:
        return GroundingState.AMBIGUOUS_DESTINATION

    # Single candidate: absolute distance fast path
    if _euclidean(g0_coord, coords[0]) <= threshold:
        return GroundingState.CLEAR

    # Symmetric camera-motion check using target as anchor (same logic as C1 above).
    tgt_raw = detections_t.get(g0["target"]["label"]) or []
    valid_anchors = [c for c in tgt_raw if isinstance(c, (list, tuple)) and len(c) == 2]
    if valid_anchors:
        gt_anchor = min(valid_anchors, key=lambda c: _euclidean(g0["target"]["coord"], c))
        tgt_abs_dist = _euclidean(g0["target"]["coord"], gt_anchor)
        if tgt_abs_dist > threshold:
            _rel = rel_threshold if rel_threshold is not None else threshold * 5.0
            rel = _relative_dist(g0_coord, g0["target"]["coord"], coords[0], gt_anchor)
            if rel <= _rel:
                return GroundingState.CLEAR

    return GroundingState.AMBIGUOUS_DESTINATION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_grounding(
    g0: dict[str, Any],
    detections_t: dict[str, list[list[int]]],
    checkpoint: str,
    threshold: float = 50.0,
    rel_threshold: float | None = None,
) -> tuple[GroundingState, Decision]:
    """Compare G₀ against Gₜ detections and return (GroundingState, Decision).

    Args:
        g0: Output of extract_g0 —
            {"target": {"label": str, "coord": [x,y]},
             "destination": {"label": str, "coord": [x,y]},
             "image_shape": [h, w]}
        detections_t: Raw AmbRes detect output at checkpoint —
            {"label": [[x,y], [x,y], ...], ...}
            (all detected instances per label, not just the first)
        checkpoint: "C1" (pre-pick, checks target) or "C2" (pre-place, checks destination)
        threshold: Max Euclidean pixel distance (absolute) to consider the same instance.
                   Placeholder default 50 px; set by pilot_threshold.py.
        rel_threshold: Max change in the (object − anchor) relative vector to still be
                       considered CLEAR.  Used when the absolute distance exceeds
                       `threshold` but the camera may have moved.  Defaults to
                       ``threshold * 5`` when None, which empirically separates camera
                       motion (≈ threshold × 3–4) from genuine object displacement.

    Returns:
        (GroundingState, Decision) where Decision follows _STATE_TO_DECISION policy.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

    if checkpoint == "C1":
        state = _check_target(g0, detections_t, threshold, rel_threshold)
    else:
        state = _check_destination(g0, detections_t, threshold, rel_threshold)

    return state, _STATE_TO_DECISION[state]


def get_checkpoint_detections(
    image_path: str | Path,
    g0: dict[str, Any],
    handler: Any,
    session_id: str = "checkpoint",
) -> dict[str, list[list[int]]]:
    """Acquire Gₜ raw detections at a checkpoint using a pre-initialized handler.

    Calls reset → detect directly (query/respond not needed because labels are
    already known from G₀). Returns ALL coordinates per label so the Consistency
    Monitor can detect multiple instances.

    Args:
        image_path: Path to the checkpoint image.
        g0: Stored G₀ (provides target/destination labels to detect).
        handler: Pre-initialized AmbResHandler (already set up with model weights).
        session_id: AmbRes session identifier for this checkpoint call.

    Returns:
        {"label": [[x, y], ...], ...} — all detections per label.
    """
    _src = Path(__file__).resolve().parent.parent   # COSE461-proj/src
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from extraction.ambres_g0_extractor import _load_image, _unwrap_handler_result

    image_arr, _ = _load_image(image_path)
    # Deduplicate labels in case target == destination
    labels = list(dict.fromkeys([g0["target"]["label"], g0["destination"]["label"]]))

    _unwrap_handler_result(handler.handle("reset", {}, [], session_id), "reset")
    result = _unwrap_handler_result(
        handler.handle("detect", {"objects": labels}, [image_arr], session_id),
        "detect",
    )
    return result["detections"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check grounding consistency: G₀ vs checkpoint detections."
    )
    parser.add_argument("g0_json", help="Path to JSON file containing G₀")
    parser.add_argument(
        "detections_json",
        help='Path to JSON file containing detections_t ({"label": [[x,y],...], ...})',
    )
    parser.add_argument("checkpoint", choices=["C1", "C2"])
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument(
        "--rel-threshold",
        type=float,
        default=None,
        help="Camera-normalized relative threshold (default: threshold * 5).",
    )
    args = parser.parse_args()

    with open(args.g0_json) as f:
        g0 = json.load(f)
    with open(args.detections_json) as f:
        detections_t = json.load(f)

    state, decision = check_grounding(
        g0, detections_t, args.checkpoint, args.threshold, args.rel_threshold
    )
    print(json.dumps({"state": state.value, "decision": decision.value}, indent=2))


if __name__ == "__main__":
    main()
