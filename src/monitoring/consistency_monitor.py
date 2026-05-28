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

    # Multiple instances: check whether the count was already the same at t₀.
    # If count is stable AND all instances remain within threshold of their G₀
    # positions, the scene is unchanged (CLEAR).  A count change or position
    # shift means a new/moved instance → AMBIGUOUS_TARGET.
    if len(coords) > 1:
        g0_coords = g0["target"].get("coords")
        if g0_coords and len(g0_coords) == len(coords):
            unmatched = list(coords)
            all_close = True
            for gc in g0_coords:
                if not unmatched:
                    all_close = False
                    break
                best = min(unmatched, key=lambda c: _euclidean(gc, c))
                if _euclidean(gc, best) > threshold:
                    all_close = False
                    break
                unmatched.remove(best)
            if all_close:
                return GroundingState.CLEAR

        # Distinguish AMBIGUOUS from INVALID when count increased:
        #   AMBIGUOUS: original instance is still present + new ones appeared → ASK
        #   INVALID:   original instance is GONE, only distractors remain → STOP
        # When t0 had a single instance, check whether any current detection is
        # near the original g0 coordinate.  If none are → the original target
        # has been removed (INVALID), not merely accompanied by new objects.
        if g0_coord is not None:
            t0_had_single = not g0_coords or len(g0_coords) == 1
            nearest = min(coords, key=lambda c: _euclidean(g0_coord, c))
            if t0_had_single and _euclidean(g0_coord, nearest) > threshold:
                return GroundingState.INVALID_TARGET

        return GroundingState.AMBIGUOUS_TARGET

    # g0_coord가 None이면 t₀에서 감지 실패 — 위치 비교 불가, 단일 인스턴스 존재 → CLEAR
    if g0_coord is None:
        return GroundingState.CLEAR

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
    dest_coord = g0["destination"]["coord"]
    dest_raw = detections_t.get(g0["destination"]["label"]) or []
    valid_anchors = [c for c in dest_raw if isinstance(c, (list, tuple)) and len(c) == 2]
    if valid_anchors and dest_coord is not None:
        gt_anchor = min(valid_anchors, key=lambda c: _euclidean(dest_coord, c))
        dest_abs_dist = _euclidean(dest_coord, gt_anchor)
        if dest_abs_dist > threshold:
            # Destination also displaced → camera moved → camera-normalized check
            _rel = rel_threshold if rel_threshold is not None else threshold * 5.0
            rel = _relative_dist(g0_coord, dest_coord, coords[0], gt_anchor)
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

    # Multiple candidates: count-stable check (mirrors _check_target logic).
    if len(coords) > 1:
        g0_coords = g0["destination"].get("coords")
        if g0_coords and len(g0_coords) == len(coords):
            unmatched = list(coords)
            all_close = True
            for gc in g0_coords:
                if not unmatched:
                    all_close = False
                    break
                best = min(unmatched, key=lambda c: _euclidean(gc, c))
                if _euclidean(gc, best) > threshold:
                    all_close = False
                    break
                unmatched.remove(best)
            if all_close:
                return GroundingState.CLEAR

        # Same INVALID vs AMBIGUOUS distinction as _check_target:
        # if the original destination is gone and only distractors remain → INVALID.
        if g0_coord is not None:
            t0_had_single = not g0_coords or len(g0_coords) == 1
            nearest = min(coords, key=lambda c: _euclidean(g0_coord, c))
            if t0_had_single and _euclidean(g0_coord, nearest) > threshold:
                return GroundingState.INVALID_DESTINATION

        return GroundingState.AMBIGUOUS_DESTINATION

    # g0_coord가 None이면 t₀에서 감지 실패 — 단일 인스턴스 존재 → CLEAR
    if g0_coord is None:
        return GroundingState.CLEAR

    # Single candidate: absolute distance fast path
    if _euclidean(g0_coord, coords[0]) <= threshold:
        return GroundingState.CLEAR

    # Symmetric camera-motion check using target as anchor (same logic as C1 above).
    tgt_coord = g0["target"]["coord"]
    tgt_raw = detections_t.get(g0["target"]["label"]) or []
    valid_anchors = [c for c in tgt_raw if isinstance(c, (list, tuple)) and len(c) == 2]
    if valid_anchors and tgt_coord is not None:
        gt_anchor = min(valid_anchors, key=lambda c: _euclidean(tgt_coord, c))
        tgt_abs_dist = _euclidean(tgt_coord, gt_anchor)
        if tgt_abs_dist > threshold:
            _rel = rel_threshold if rel_threshold is not None else threshold * 5.0
            rel = _relative_dist(g0_coord, tgt_coord, coords[0], gt_anchor)
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
# Ensemble API  (GDino + Molmo B+A hybrid)
# ---------------------------------------------------------------------------

def check_grounding_ensemble(
    g0: dict[str, Any],
    molmo_detections_ck: dict[str, list[list[int]]],
    gdino_detections_t0: dict[str, list[list[float]]],
    gdino_detections_ck: dict[str, list[list[float]]],
    checkpoint: str,
    threshold: float = 50.0,
    rel_threshold: float | None = None,
) -> tuple[GroundingState, Decision]:
    """B+A hybrid ensemble: GDino(역할 분담) + delta 기반 Molmo 보정.

    알고리즘:
      1. GDino t0 탐지 실패(count=0) → Molmo 단독 판단으로 fallback (force_map 방지)
      2. GDino Ck 탐지 0 + Molmo Ck 탐지 0 → INVALID (high-confidence disappear)
      3. GDino Ck 탐지 0 + Molmo Ck 탐지 >0 → AMBIGUOUS (conflict → ASK)
      4. GDino Ck count=1 + Molmo AMBIGUOUS → GDino 좌표로 재판단 (과잉탐지 보정)
      5. 그 외 → Molmo 좌표 기반 판단 그대로 사용

    Args:
        g0: extract_g0 결과.
        molmo_detections_ck: Molmo checkpoint 탐지 결과 {label: [[x,y], ...]}.
        gdino_detections_t0: GDino t0 탐지 결과 {label: [[cx,cy], ...]}.
        gdino_detections_ck: GDino checkpoint 탐지 결과 {label: [[cx,cy], ...]}.
        checkpoint: "C1" or "C2".
        threshold: 절대 픽셀 거리 임계값.
        rel_threshold: 카메라 이동 보정 상대 임계값.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

    role = "target" if checkpoint == "C1" else "destination"
    label = g0[role]["label"]

    gdino_t0_count = len(gdino_detections_t0.get(label) or [])
    gdino_ck_coords = [
        c for c in (gdino_detections_ck.get(label) or [])
        if isinstance(c, (list, tuple)) and len(c) == 2
    ]
    gdino_ck_count = len(gdino_ck_coords)
    molmo_ck_count = len([
        c for c in (molmo_detections_ck.get(label) or [])
        if isinstance(c, (list, tuple)) and len(c) == 2
    ])

    # ── 1. GDino t0 실패 → Molmo fallback ─────────────────────────────────
    if gdino_t0_count == 0:
        return check_grounding(g0, molmo_detections_ck, checkpoint, threshold, rel_threshold)

    # ── 2 & 3. GDino disappear 판단 ────────────────────────────────────────
    if gdino_ck_count == 0:
        if molmo_ck_count == 0:
            # 두 탐지기 모두 사라짐 → INVALID (high confidence)
            state = (
                GroundingState.INVALID_TARGET
                if checkpoint == "C1"
                else GroundingState.INVALID_DESTINATION
            )
        else:
            # GDino 사라졌지만 Molmo는 탐지 → 불일치 → AMBIGUOUS
            state = (
                GroundingState.AMBIGUOUS_TARGET
                if checkpoint == "C1"
                else GroundingState.AMBIGUOUS_DESTINATION
            )
        return state, _STATE_TO_DECISION[state]

    # ── 4 & 5. GDino 존재 확인됨 → Molmo 좌표 기반 판단 ─────────────────────
    molmo_state, molmo_decision = check_grounding(
        g0, molmo_detections_ck, checkpoint, threshold, rel_threshold
    )

    ambiguous_states = (GroundingState.AMBIGUOUS_TARGET, GroundingState.AMBIGUOUS_DESTINATION)
    invalid_states = (GroundingState.INVALID_TARGET, GroundingState.INVALID_DESTINATION)

    # Molmo가 INVALID를 반환했지만 GDino가 객체 존재를 확인한 경우:
    # Molmo의 다중 포인트 탐지(격자형 좌표)가 이동된 객체에 대해 모두 g0에서 멀어
    # 잘못된 INVALID를 유발할 수 있음. GDino 좌표로 재판단.
    if molmo_state in invalid_states and gdino_ck_count >= 1:
        g0_coord = g0[role]["coord"]
        nearest = min(gdino_ck_coords, key=lambda c: _euclidean(g0_coord, c))
        if _euclidean(g0_coord, nearest) <= threshold:
            return GroundingState.CLEAR, Decision.CONTINUE
        # GDino가 객체를 확인했으나 변위됨 → AMBIGUOUS → ASK
        ambig_state = (
            GroundingState.AMBIGUOUS_TARGET
            if checkpoint == "C1"
            else GroundingState.AMBIGUOUS_DESTINATION
        )
        return ambig_state, _STATE_TO_DECISION[ambig_state]

    if molmo_state in ambiguous_states and gdino_ck_count == 1:
        # Molmo 과잉탐지 보정: GDino가 정확히 1개만 탐지 → GDino 좌표로 CLEAR 판단
        g0_coord = g0[role]["coord"]
        nearest = min(gdino_ck_coords, key=lambda c: _euclidean(g0_coord, c))
        if _euclidean(g0_coord, nearest) <= threshold:
            return GroundingState.CLEAR, Decision.CONTINUE
        # 거리 초과 시에도 GDino count=1이므로 AMBIGUOUS보다 완화된 판단
        # anchor 기반 카메라 보정 시도
        other_role = "destination" if role == "target" else "target"
        other_label = g0[other_role]["label"]
        gdino_other = [
            c for c in (gdino_detections_ck.get(other_label) or [])
            if isinstance(c, (list, tuple)) and len(c) == 2
        ]
        if gdino_other:
            gt_anchor = min(gdino_other, key=lambda c: _euclidean(g0[other_role]["coord"], c))
            anchor_dist = _euclidean(g0[other_role]["coord"], gt_anchor)
            if anchor_dist > threshold:
                _rel = rel_threshold if rel_threshold is not None else threshold * 5.0
                rel = _relative_dist(g0_coord, g0[other_role]["coord"], nearest, gt_anchor)
                if rel <= _rel:
                    return GroundingState.CLEAR, Decision.CONTINUE

    return molmo_state, molmo_decision


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
