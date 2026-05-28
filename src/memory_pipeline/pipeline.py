"""
src.memory_pipeline.pipeline
==============================
Memory-augmented AmbRes pipeline.

Core difference from src/pipeline.py (OURS):
  - GDino tracks object states across all checkpoints (not just t0)
  - EpisodeMemory accumulates: detections, decisions, user responses
  - At each checkpoint, AmbRes receives the full memory context in its
    task_description — not just the original task string
  - Decision is still made by check_grounding (coordinate-based), but the
    AmbRes reasoning step is memory-conditioned

This enables the system to reason about dynamic state changes:
  "At t0 there was one cup.  At C1 two cups appeared — which is the target?"
  rather than treating each checkpoint as if the scene has never changed.

Ablation note:
  Set memory_context=False to run GDino-only (no memory injection) for the
  "+GDino, -Memory" ablation condition.
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

# Path setup — resolve src/ root so internal imports work when run standalone
_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
_AMBRES_ROOT = _REPO_ROOT.parent / "AmbRes"
for _p in (_SRC_ROOT, _REPO_ROOT, _AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from memory_pipeline.context_builder import build_memory_context, inject_context
from memory_pipeline.episode_memory import CheckpointRecord, EpisodeMemory, ObjectState
from memory_pipeline.scene_monitor import SceneMonitor
from monitoring.consistency_monitor import (
    Decision, GroundingState,
    check_grounding, check_grounding_ensemble, get_checkpoint_detections,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckpointOutcome:
    checkpoint:      str
    state:           GroundingState
    decision:        Decision
    detections:      dict[str, Any]
    changes:         list[str]
    memory_context:  str = ""
    user_response:   str = ""
    question:        str = ""


@dataclass
class MemoryPipelineResult:
    status:      Literal["complete", "stopped", "error"]
    memory:      EpisodeMemory
    c1:          CheckpointOutcome | None = None
    c2:          CheckpointOutcome | None = None
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        def _fmt(co: CheckpointOutcome | None) -> dict | None:
            if co is None:
                return None
            return {
                "checkpoint":     co.checkpoint,
                "state":          co.state.value,
                "decision":       co.decision.value,
                "changes":        co.changes,
                "question":       co.question,
                "user_response":  co.user_response,
                "memory_context": co.memory_context,
            }
        return {
            "status":      self.status,
            "stop_reason": self.stop_reason,
            "c1":          _fmt(self.c1),
            "c2":          _fmt(self.c2),
            "memory":      self.memory.to_dict(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image_arr(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _first_coord(detections: dict[str, Any], label: str) -> list[int] | None:
    raw = detections.get(label) or []
    valid = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]
    return [int(valid[0][0]), int(valid[0][1])] if valid else None


def _all_valid_coords(detections: dict[str, Any], label: str) -> list[list[int]]:
    raw = detections.get(label) or []
    return [[int(c[0]), int(c[1])] for c in raw
            if isinstance(c, (list, tuple)) and len(c) == 2]


def _build_g0_from_detections(
    detections: dict[str, Any],
    target_label: str,
    dest_label: str,
    image_shape: list[int],
) -> dict[str, Any]:
    """Build a G₀-compatible dict from GDino detections."""
    return {
        "target": {
            "label":  target_label,
            "coord":  _first_coord(detections, target_label),
            "coords": _all_valid_coords(detections, target_label),
        },
        "destination": {
            "label":  dest_label,
            "coord":  _first_coord(detections, dest_label),
            "coords": _all_valid_coords(detections, dest_label),
        },
        "image_shape": image_shape,
    }


def _extract_labels_from_ambres(
    image_path: str | Path,
    task_with_context: str,
    handler: Any,
    session_id: str,
) -> tuple[str, str, str]:
    """
    Run AmbRes query+respond to get target/destination labels and VLM question.

    The task string already contains the memory context, so AmbRes
    reasons about the current situation in light of prior observations.
    Returns (target_label, destination_label, clarifying_question).
    clarifying_question is the VLM-generated question from the query step;
    it is "" if the VLM judged the task unambiguous.
    """
    from extraction.role_parser import parse_roles

    image_arr = _load_image_arr(image_path)
    tensors = [image_arr]

    def _unwrap(result: dict, step: str) -> dict:
        if not result.get("success"):
            raise RuntimeError(f"AmbRes {step} failed: {result.get('error')}")
        return {k: v for k, v in result.items() if k != "success"}

    _unwrap(handler.handle("reset", {}, [], session_id), "reset")

    step1 = _unwrap(
        handler.handle("query", {"task_description": task_with_context}, tensors, session_id),
        "query",
    )
    # Capture the VLM-generated clarifying question before calling respond.
    # When memory context is injected, the VLM naturally asks about scene changes
    # (e.g. "원래 1개였던 컵이 현재 2개입니다. 어떤 컵을 집어야 하나요?")
    vlm_question: str = step1.get("clarifying_question", "") or ""

    step2 = _unwrap(
        handler.handle("respond", {"response": ""}, [], session_id),
        "respond",
    )

    obj_list = step2.get("object_list") or step2.get("task_objects") or \
               step1.get("object_list") or step1.get("task_objects") or []
    roles = parse_roles(obj_list, task_with_context)
    return roles["target"], roles["destination"], vlm_question


def _fallback_question(g0: dict, role: str, changes: list[str] | None = None) -> str:
    """Template question used when the VLM returns no clarifying_question."""
    label = g0[role]["label"]
    if changes:
        change_desc = "; ".join(changes[:2])
        if role == "target":
            return f"씬 변화({change_desc})가 감지됐습니다. 어떤 '{label}'을(를) 집어야 하나요?"
        return f"씬 변화({change_desc})가 감지됐습니다. 어떤 '{label}'에 놓아야 하나요?"
    if role == "target":
        return f"어떤 '{label}'을(를) 집어야 하나요?"
    return f"어떤 '{label}'에 놓아야 하나요?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_memory_pipeline(
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
    use_memory_context: bool = True,
    user_response_fn: Callable[[str, dict], str] | None = None,
    session_prefix: str = "mem",
) -> MemoryPipelineResult:
    """
    Run the memory-augmented AmbRes pipeline.

    Args:
        image_t0 / c1 / c2:   Image paths for each phase.
        task_description:      Original user task (never modified).
        target_label:          GDino query label for the target object.
        destination_label:     GDino query label for the destination object.
        handler:               AmbResHandler instance (for VLM reasoning).
        gdino:                 GroundingDINO instance (for continuous tracking).
        threshold:             Pixel distance threshold for check_grounding.
        use_memory_context:    False → GDino-only, no memory injection
                               (ablation: isolates memory contribution).
        user_response_fn:      Called on ASK. Signature: (question, g0) → str.
        session_prefix:        AmbRes session ID prefix.

    Returns:
        MemoryPipelineResult with full memory trace.
    """
    monitor = SceneMonitor(gdino, move_threshold_px=threshold)
    memory = EpisodeMemory(task_description)
    if user_response_fn is None:
        user_response_fn = lambda q, g: ""

    # ── t₀: initial snapshot ─────────────────────────────────────────────────
    pil_t0 = Image.open(image_t0).convert("RGB")
    W, H = pil_t0.size
    image_shape = [H, W]

    t0_detections = gdino.detect(pil_t0, [target_label, destination_label])
    t0_record = monitor.snapshot_from_detections(
        t0_detections, target_label, destination_label, "t0"
    )
    memory.record(t0_record)

    # G₀ built from GDino detections at t₀
    g0 = _build_g0_from_detections(t0_detections, target_label, destination_label, image_shape)

    logger.info(
        "Memory t0: target=%s@%s  dest=%s@%s",
        target_label, g0["target"]["coord"],
        destination_label, g0["destination"]["coord"],
    )

    # ── C1: pre-pick checkpoint ──────────────────────────────────────────────
    c1_outcome, g0_after_c1 = _run_checkpoint(
        "C1", image_c1, task_description,
        target_label, destination_label,
        g0=g0,
        prev_record=t0_record,
        memory=memory,
        monitor=monitor,
        handler=handler,
        gdino=gdino,
        threshold=threshold,
        use_memory_context=use_memory_context,
        user_response_fn=user_response_fn,
        session_prefix=session_prefix,
        image_shape=image_shape,
        gdino_t0_detections=t0_detections,
    )

    if c1_outcome.decision == Decision.STOP:
        return MemoryPipelineResult(
            status="stopped",
            memory=memory,
            c1=c1_outcome,
            stop_reason=f"C1 {c1_outcome.state.value} → STOP",
        )

    g0 = g0_after_c1

    # ── C2: pre-place checkpoint ─────────────────────────────────────────────
    c1_record = memory.get("C1")
    c2_outcome, _ = _run_checkpoint(
        "C2", image_c2, task_description,
        target_label, destination_label,
        g0=g0,
        prev_record=c1_record or t0_record,
        memory=memory,
        monitor=monitor,
        handler=handler,
        gdino=gdino,
        threshold=threshold,
        use_memory_context=use_memory_context,
        user_response_fn=user_response_fn,
        session_prefix=session_prefix,
        image_shape=image_shape,
        gdino_t0_detections=t0_detections,
    )

    if c2_outcome.decision == Decision.STOP:
        return MemoryPipelineResult(
            status="stopped",
            memory=memory,
            c1=c1_outcome,
            c2=c2_outcome,
            stop_reason=f"C2 {c2_outcome.state.value} → STOP",
        )

    return MemoryPipelineResult(
        status="complete",
        memory=memory,
        c1=c1_outcome,
        c2=c2_outcome,
    )


# ---------------------------------------------------------------------------
# Checkpoint execution (shared C1 / C2 logic)
# ---------------------------------------------------------------------------

def _run_checkpoint(
    checkpoint: str,
    image_path: str | Path,
    task_description: str,
    target_label: str,
    destination_label: str,
    *,
    g0: dict[str, Any],
    prev_record: CheckpointRecord,
    memory: EpisodeMemory,
    monitor: SceneMonitor,
    handler: Any,
    gdino: Any,
    threshold: float,
    use_memory_context: bool,
    user_response_fn: Callable,
    session_prefix: str,
    image_shape: list[int],
    gdino_t0_detections: dict[str, Any],
) -> tuple[CheckpointOutcome, dict[str, Any]]:
    """Execute one checkpoint: detect → memory context → AmbRes → grounding check."""

    # ── 1. GDino snapshot at checkpoint ─────────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    ck_detections = gdino.detect(pil_img, [target_label, destination_label])
    ck_record = monitor.snapshot_from_detections(
        ck_detections, target_label, destination_label, checkpoint
    )

    # ── 2. Detect changes from previous snapshot ─────────────────────────────
    changes = monitor.detect_changes(prev_record, ck_record)
    ck_record.changes = changes
    if changes:
        logger.info("Memory %s changes: %s", checkpoint, "; ".join(changes))

    # ── 3. Build memory context (or empty string for ablation) ───────────────
    context_str = build_memory_context(memory, checkpoint) if use_memory_context else ""
    task_with_context = inject_context(task_description, context_str)

    # ── 4. AmbRes reasoning with memory-conditioned task ─────────────────────
    # Re-extract labels using the memory-augmented task description.
    # Capture the VLM's clarifying_question for use in the ASK path.
    try:
        tgt_label_ck, dst_label_ck, vlm_question = _extract_labels_from_ambres(
            image_path, task_with_context,
            handler, f"{session_prefix}_{checkpoint.lower()}",
        )
        # Guard against VLM hallucination: when memory context is injected, the
        # VLM sometimes returns entirely different objects ('banana', 'middle drawer')
        # instead of the original task objects.  If the returned label shares no
        # words with the original label, fall back to prevent GDino from failing
        # to detect the original target (which would cause a false INVALID/STOP).
        def _has_word_overlap(a: str, b: str) -> bool:
            return bool(set(a.lower().split()) & set(b.lower().split()))

        if not _has_word_overlap(tgt_label_ck, target_label):
            logger.warning(
                "Memory %s: VLM returned label '%s' has no overlap with original '%s' "
                "— falling back to original to prevent hallucination",
                checkpoint, tgt_label_ck, target_label,
            )
            tgt_label_ck = target_label
            vlm_question = ""  # question based on hallucinated label is unreliable
        if not _has_word_overlap(dst_label_ck, destination_label):
            logger.warning(
                "Memory %s: VLM returned dest label '%s' has no overlap with original '%s' "
                "— falling back to original",
                checkpoint, dst_label_ck, destination_label,
            )
            dst_label_ck = destination_label
    except RuntimeError as exc:
        logger.warning("AmbRes failed at %s: %s — using original labels", checkpoint, exc)
        tgt_label_ck, dst_label_ck, vlm_question = target_label, destination_label, ""

    # ── 5. Build updated G₀ for this checkpoint ──────────────────────────────
    # Re-detect with potentially relabelled queries
    if tgt_label_ck != target_label or dst_label_ck != destination_label:
        ck_detections_relabelled = gdino.detect(pil_img, [tgt_label_ck, dst_label_ck])
    else:
        ck_detections_relabelled = ck_detections

    g0_ck = _build_g0_from_detections(
        ck_detections_relabelled, tgt_label_ck, dst_label_ck, image_shape
    )

    # ── 6. Grounding check: GDino+Molmo ensemble ────────────────────────────
    # Molmo detections at checkpoint: used as primary presence signal.
    # GDino t0/ck detections: used to validate absence (if both miss → INVALID).
    # This lets Molmo's joint image+text understanding judge genuine absence (S3)
    # while GDino catches false-positive Molmo detections (S6 moved object).
    molmo_detections_ck = get_checkpoint_detections(
        image_path, g0, handler,
        f"{session_prefix}_{checkpoint.lower()}_det",
    )

    if use_memory_context:
        state, decision = check_grounding_ensemble(
            g0,
            molmo_detections_ck,
            gdino_t0_detections,
            ck_detections,   # original-label GDino C1 detections (consistent with g0)
            checkpoint,
            threshold,
        )
    else:
        # ablation (STATIC): vanilla Molmo-based check, no GDino ensemble
        state, decision = check_grounding(g0, molmo_detections_ck, checkpoint, threshold)

    logger.info(
        "Memory %s: state=%s decision=%s  (context_len=%d)",
        checkpoint, state.value, decision.value, len(context_str),
    )

    # ── 7. ASK: user clarification + G₀ update ───────────────────────────────
    question = ""
    user_response = ""
    g0_after = g0

    if decision == Decision.ASK:
        # Use the VLM's own clarifying question when available.
        # The VLM receives the full episode memory context via task_with_context,
        # so its question naturally references scene changes (e.g. count increase,
        # position shift). Fall back to a change-aware template only if the VLM
        # returned no question.
        role = "target" if checkpoint == "C1" else "destination"
        if vlm_question:
            question = f"[{checkpoint}] {vlm_question}"
            logger.info("Memory %s: using VLM question: %s", checkpoint, vlm_question)
        else:
            question = f"[{checkpoint} {state.value}] {_fallback_question(g0, role, changes)}"
            logger.info("Memory %s: VLM returned no question; using fallback", checkpoint)
        user_response = user_response_fn(question, g0)
        if user_response:
            # Rebuild G₀ with user clarification as memory context
            clarified_context = inject_context(
                task_description,
                build_memory_context(memory, f"{checkpoint}_clarify") if use_memory_context else "",
            )
            try:
                tgt_cl, dst_cl, _ = _extract_labels_from_ambres(
                    image_path,
                    inject_context(clarified_context, f'User clarification: "{user_response}"'),
                    handler, f"{session_prefix}_{checkpoint.lower()}_clarify",
                )
                det_cl = gdino.detect(pil_img, [tgt_cl, dst_cl])
                g0_after = _build_g0_from_detections(det_cl, tgt_cl, dst_cl, image_shape)
            except RuntimeError as exc:
                logger.warning("Clarification AmbRes failed: %s", exc)

    # ── 8. Record in memory ───────────────────────────────────────────────────
    ck_record.decision = decision.value
    ck_record.grounding_state = state.value
    ck_record.user_response = user_response
    memory.record(ck_record)

    outcome = CheckpointOutcome(
        checkpoint=checkpoint,
        state=state,
        decision=decision,
        detections=ck_detections_relabelled,
        changes=changes,
        memory_context=context_str,
        user_response=user_response,
        question=question,
    )
    return outcome, g0_after
