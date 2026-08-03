"""
src.memory_pipeline.static_pipeline
======================================
Control condition: static iterative AmbRes without episode memory.

At each checkpoint, AmbRes is called with ONLY the original task description
— no history of prior detections, decisions, or scene changes.  This mirrors
what happens when a non-adaptive system re-applies the same grounding logic
at every checkpoint independently.

Key contrast with pipeline.py:
  - No EpisodeMemory accumulation
  - AmbRes task_description never changes between checkpoints
  - Each checkpoint is evaluated as if the episode just started
  - GDino detector is identical (isolates the memory variable)

This is the direct "ablation off" condition for the memory contribution.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
_AMBRES_ROOT = _REPO_ROOT.parent / "AmbRes"
for _p in (_SRC_ROOT, _REPO_ROOT, _AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from monitoring.consistency_monitor import Decision, GroundingState, check_grounding


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StaticCheckpointOutcome:
    checkpoint:  str
    state:       GroundingState
    decision:    Decision
    detections:  dict[str, Any]
    question:    str = ""
    user_response: str = ""


@dataclass
class StaticPipelineResult:
    status:      Literal["complete", "stopped", "error"]
    g0:          dict[str, Any] | None
    c1:          StaticCheckpointOutcome | None = None
    c2:          StaticCheckpointOutcome | None = None
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        def _fmt(co: StaticCheckpointOutcome | None) -> dict | None:
            if co is None:
                return None
            return {
                "checkpoint":    co.checkpoint,
                "state":         co.state.value,
                "decision":      co.decision.value,
                "question":      co.question,
                "user_response": co.user_response,
            }
        return {
            "status":      self.status,
            "stop_reason": self.stop_reason,
            "g0":          self.g0,
            "c1":          _fmt(self.c1),
            "c2":          _fmt(self.c2),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_arr(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _first_coord(detections: dict, label: str) -> list[int] | None:
    raw = detections.get(label) or []
    valid = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]
    return [int(valid[0][0]), int(valid[0][1])] if valid else None


def _all_coords(detections: dict, label: str) -> list[list[int]]:
    raw = detections.get(label) or []
    return [[int(c[0]), int(c[1])] for c in raw
            if isinstance(c, (list, tuple)) and len(c) == 2]


def _build_g0(
    detections: dict,
    target_label: str,
    dest_label: str,
    image_shape: list[int],
) -> dict[str, Any]:
    return {
        "target": {
            "label":  target_label,
            "coord":  _first_coord(detections, target_label),
            "coords": _all_coords(detections, target_label),
        },
        "destination": {
            "label":  dest_label,
            "coord":  _first_coord(detections, dest_label),
            "coords": _all_coords(detections, dest_label),
        },
        "image_shape": image_shape,
    }


def _ambres_labels(
    image_path: str | Path,
    task_description: str,    # original task only — no memory context
    handler: Any,
    session_id: str,
) -> tuple[str, str]:
    """Run AmbRes query→respond to get target/destination labels.

    task_description is always the raw user task; no memory context is injected.
    """
    from extraction.role_parser import parse_roles

    arr = _load_arr(image_path)
    tensors = [arr]

    def _unwrap(r: dict, step: str) -> dict:
        if not r.get("success"):
            raise RuntimeError(f"AmbRes {step} failed: {r.get('error')}")
        return {k: v for k, v in r.items() if k != "success"}

    _unwrap(handler.handle("reset", {}, [], session_id), "reset")
    step1 = _unwrap(
        handler.handle("query", {"task_description": task_description}, tensors, session_id),
        "query",
    )
    step2 = _unwrap(
        handler.handle("respond", {"response": ""}, [], session_id),
        "respond",
    )

    obj_list = (
        step2.get("object_list") or step2.get("task_objects") or
        step1.get("object_list") or step1.get("task_objects") or []
    )
    roles = parse_roles(obj_list, task_description)
    return roles["target"], roles["destination"]


def _clarifying_question(g0: dict, role: str) -> str:
    label = g0[role]["label"]
    if role == "target":
        return f"[C1] 어떤 '{label}'을(를) 집어야 하나요?"
    return f"[C2] 어떤 '{label}'에 놓아야 하나요?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_static_pipeline(
    image_t0: str | Path,
    image_c1: str | Path,
    image_c2: str | Path,
    task_description: str,
    target_label: str,
    destination_label: str,
    *,
    handler: Any,
    gdino: Any,
    threshold: float = 50.0,
    user_response_fn: Callable[[str, dict], str] | None = None,
    session_prefix: str = "static",
) -> StaticPipelineResult:
    """
    Run the static (no-memory) pipeline.

    At every checkpoint:
      1. GDino detects objects.
      2. AmbRes is called with the original task_description only.
      3. check_grounding makes the CONTINUE/ASK/STOP decision.

    No information flows between checkpoints.  Each is evaluated
    as if it were the first and only observation.
    """
    if user_response_fn is None:
        user_response_fn = lambda q, g: ""

    # ── t0: build reference G₀ ───────────────────────────────────────────────
    pil_t0 = Image.open(image_t0).convert("RGB")
    W, H = pil_t0.size
    image_shape = [H, W]

    t0_det = gdino.detect(pil_t0, [target_label, destination_label])
    g0 = _build_g0(t0_det, target_label, destination_label, image_shape)

    logger.info(
        "Static t0: target=%s@%s  dest=%s@%s",
        target_label, g0["target"]["coord"],
        destination_label, g0["destination"]["coord"],
    )

    # ── C1 ───────────────────────────────────────────────────────────────────
    c1_out, g0_c1 = _run_static_checkpoint(
        "C1", image_c1, task_description,
        target_label, destination_label,
        g0=g0,
        handler=handler, gdino=gdino,
        threshold=threshold,
        user_response_fn=user_response_fn,
        session_prefix=session_prefix,
        image_shape=image_shape,
    )

    if c1_out.decision == Decision.STOP:
        return StaticPipelineResult(
            status="stopped", g0=g0, c1=c1_out,
            stop_reason=f"C1 {c1_out.state.value} → STOP",
        )

    # ── C2 ───────────────────────────────────────────────────────────────────
    c2_out, _ = _run_static_checkpoint(
        "C2", image_c2, task_description,
        target_label, destination_label,
        g0=g0_c1,
        handler=handler, gdino=gdino,
        threshold=threshold,
        user_response_fn=user_response_fn,
        session_prefix=session_prefix,
        image_shape=image_shape,
    )

    if c2_out.decision == Decision.STOP:
        return StaticPipelineResult(
            status="stopped", g0=g0, c1=c1_out, c2=c2_out,
            stop_reason=f"C2 {c2_out.state.value} → STOP",
        )

    return StaticPipelineResult(status="complete", g0=g0, c1=c1_out, c2=c2_out)


# ---------------------------------------------------------------------------
# Checkpoint execution
# ---------------------------------------------------------------------------

def _run_static_checkpoint(
    checkpoint: str,
    image_path: str | Path,
    task_description: str,       # always original — never augmented
    target_label: str,
    destination_label: str,
    *,
    g0: dict[str, Any],
    handler: Any,
    gdino: Any,
    threshold: float,
    user_response_fn: Callable,
    session_prefix: str,
    image_shape: list[int],
) -> tuple[StaticCheckpointOutcome, dict[str, Any]]:
    pil_img = Image.open(image_path).convert("RGB")

    # 1. AmbRes — called with original task only, no memory context
    try:
        tgt, dst = _ambres_labels(
            image_path, task_description,
            handler, f"{session_prefix}_{checkpoint.lower()}",
        )
    except RuntimeError as exc:
        logger.warning("AmbRes failed at %s: %s — using original labels", checkpoint, exc)
        tgt, dst = target_label, destination_label

    # 2. GDino detection with (potentially updated) labels
    ck_det = gdino.detect(pil_img, [tgt, dst])
    g0_ck = _build_g0(ck_det, tgt, dst, image_shape)

    # 3. Grounding check
    state, decision = check_grounding(g0, ck_det, checkpoint, threshold)

    logger.info("Static %s: state=%s decision=%s", checkpoint, state.value, decision.value)

    # 4. ASK path
    question = ""
    user_response = ""
    g0_after = g0_ck

    if decision == Decision.ASK:
        role = "target" if checkpoint == "C1" else "destination"
        question = _clarifying_question(g0, role)
        user_response = user_response_fn(question, g0)
        if user_response:
            # Re-run AmbRes with user response — still no memory context
            try:
                tgt2, dst2 = _ambres_labels(
                    image_path,
                    f"{task_description}\nUser clarification: \"{user_response}\"",
                    handler, f"{session_prefix}_{checkpoint.lower()}_ask",
                )
                det2 = gdino.detect(pil_img, [tgt2, dst2])
                g0_after = _build_g0(det2, tgt2, dst2, image_shape)
            except RuntimeError as exc:
                logger.warning("ASK clarification failed: %s", exc)

    outcome = StaticCheckpointOutcome(
        checkpoint=checkpoint,
        state=state,
        decision=decision,
        detections=ck_det,
        question=question,
        user_response=user_response,
    )
    return outcome, g0_after
