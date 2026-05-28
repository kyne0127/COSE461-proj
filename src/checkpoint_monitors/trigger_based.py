"""
src.checkpoint_monitors.trigger_based
======================================
Offline evaluation wrapper for the DINOv2 trigger-based monitor.

Simulates what the real-robot trigger loop would do for a single checkpoint:
  1. Compute DINOv2 scene-change score between t₀ and the checkpoint frame.
  2. If score > θ  → trigger fires → run AmbRes consistency check.
  3. If score ≤ θ  → no change detected → return CONTINUE.

This gives a `BaselineResult` with the same interface as B1–B5, so it can be
registered in evaluate.py and compared against other methods.

Limitation: offline evaluation uses only two frames (t₀ and checkpoint),
whereas the real monitor runs at 5–10 Hz and can fire at any frame between
checkpoints.  The score computed here is a lower bound on what the live monitor
would see — the peak change score mid-episode may be higher.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from checkpoint_monitors.common import BaselineResult
from monitoring.consistency_monitor import Decision, check_grounding, get_checkpoint_detections
from trigger.dinov2_monitor import DINOv2Monitor

logger = logging.getLogger(__name__)


def run_trigger_based(
    g0: dict[str, Any],
    initial_img: str | Path,
    checkpoint_img: str | Path,
    checkpoint: str,
    *,
    handler: Any,
    dinov2_encoder: Any,
    threshold: float = 50.0,
    dinov2_threshold: float = 0.15,
    bbox_size: int = 80,
    session_id: str = "trigger",
) -> BaselineResult:
    """
    Args:
        g0:               Pre-extracted G₀ dict (target / destination / image_shape).
        initial_img:      Path to t₀ frame — DINOv2 reference features are extracted here.
        checkpoint_img:   Path to C1 or C2 frame.
        checkpoint:       "C1" or "C2".
        handler:          AmbResHandler for consistency-check detections.
        dinov2_encoder:   Object with `encode(crop: np.ndarray) → np.ndarray`.
                          Use `trigger.dinov2_monitor.DINOv2Encoder` or `_RandomEmbeddingModel`.
        threshold:        Pixel distance threshold for check_grounding().
        dinov2_threshold: Scene-change score threshold θ (spec §2.4 default 0.15).
        bbox_size:        Square crop size around G₀ coord (pixels).
        session_id:       AmbRes session ID prefix.

    Returns:
        BaselineResult with method="TRIGGER".
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

    # ── Load t₀ reference frame ──────────────────────────────────────────────
    frame_0 = np.asarray(Image.open(initial_img).convert("RGB"), dtype=np.uint8)

    # ── Build DINOv2Monitor (stores reference features) ─────────────────────
    monitor = DINOv2Monitor(
        frame_0=frame_0,
        g0=g0,
        model=dinov2_encoder,
        threshold=dinov2_threshold,
        bbox_size=bbox_size,
    )

    # ── Load checkpoint frame and compute score ──────────────────────────────
    frame_ck = np.asarray(Image.open(checkpoint_img).convert("RGB"), dtype=np.uint8)
    phase = "pre_pick" if checkpoint == "C1" else "pre_place"
    score = monitor.compute_score(frame_ck, phase)

    logger.debug(
        "TRIGGER %s: phase=%s score=%.4f θ=%.4f → %s",
        checkpoint, phase, score, dinov2_threshold,
        "FIRE" if score > dinov2_threshold else "PASS",
    )

    # ── No trigger: scene has not changed enough ─────────────────────────────
    if score <= dinov2_threshold:
        return BaselineResult(
            method="TRIGGER",
            decision=Decision.CONTINUE,
            reason=(
                f"DINOv2 score {score:.4f} ≤ θ={dinov2_threshold:.4f} → "
                "trigger did not fire; assuming CONTINUE"
            ),
            metadata={
                "dinov2_score": score,
                "dinov2_scores_detail": monitor.last_scores,
                "triggered": False,
                "phase": phase,
            },
        )

    # ── Trigger fired: run AmbRes consistency check ──────────────────────────
    detections = get_checkpoint_detections(
        checkpoint_img, g0, handler, session_id=session_id
    )
    state, decision = check_grounding(g0, detections, checkpoint, threshold)

    logger.debug(
        "TRIGGER %s after fire: state=%s decision=%s",
        checkpoint, state.value, decision.value,
    )

    return BaselineResult(
        method="TRIGGER",
        decision=decision,
        reason=(
            f"DINOv2 score {score:.4f} > θ={dinov2_threshold:.4f} → "
            f"triggered; AmbRes state={state.value}"
        ),
        metadata={
            "dinov2_score": score,
            "dinov2_scores_detail": monitor.last_scores,
            "triggered": True,
            "phase": phase,
            "grounding_state": state.value,
        },
    )
