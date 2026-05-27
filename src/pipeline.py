"""End-to-end pipeline: t₀ → C1 → C2.

Orchestrates:
  1. G₀ extraction at t₀          (extraction.ambres_g0_extractor)
  2. C1 consistency check          (monitoring.consistency_monitor)
  3. Rolling G₀ update if ASK at C1
  4. C2 consistency check          (monitoring.consistency_monitor)
  5. Rolling G₀ update if ASK at C2

Rolling update (proposal §6.1):
  When ASK is returned, the pipeline calls user_response_fn to get the
  user's clarification, then re-runs AmbRes (reset → query → respond(answer)
  → detect) on the checkpoint image to produce an updated G₀. The updated
  G₀ is used for subsequent checkpoint checks.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

_SRC_ROOT   = Path(__file__).resolve().parent        # COSE461-proj/src
REPO_ROOT   = _SRC_ROOT.parent                       # COSE461-proj
AMBRES_ROOT = REPO_ROOT.parent / "AmbRes"            # workspace/AmbRes
for _p in (_SRC_ROOT, REPO_ROOT, AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extraction.ambres_g0_extractor import (
    _first_coord,
    _load_image,
    _make_handler,
    _object_list_from_step,
    _unwrap_handler_result,
    extract_g0,
)
from extraction.role_parser import parse_roles
from monitoring.consistency_monitor import (
    Decision,
    GroundingState,
    check_grounding,
    get_checkpoint_detections,
)

# Callable type: (clarifying_question, current_g0) → user_answer_str
UserResponseFn = Callable[[str, dict[str, Any]], str]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckpointOutcome:
    """Result of one checkpoint (C1 or C2)."""
    checkpoint:     str            # "C1" or "C2"
    state:          GroundingState
    decision:       Decision
    detections:     dict[str, list]   # raw detect output at this checkpoint
    g0_before:      dict[str, Any]    # G₀ used for comparison
    g0_after:       dict[str, Any]    # G₀ after rolling update (== g0_before if no update)
    user_response:  str = ""          # "" if no user interaction
    question:       str = ""          # clarifying question generated on ASK


@dataclass
class PipelineResult:
    """Full pipeline outcome."""
    # "complete"          — reached end of C2, PLACE can proceed
    # "stopped"           — STOP decision at C1 or C2
    # "initial_ambiguous" — t₀ Step 1 returned ambiguity=true
    status:      Literal["complete", "stopped", "initial_ambiguous"]
    g0_initial:  dict[str, Any] | None    # G₀ from t₀ (None if initial_ambiguous)
    c1:          CheckpointOutcome | None
    c2:          CheckpointOutcome | None
    stop_reason: str = ""                 # populated when status=="stopped"

    def to_dict(self) -> dict[str, Any]:
        def _fmt(co: CheckpointOutcome | None) -> dict | None:
            if co is None:
                return None
            return {
                "checkpoint":    co.checkpoint,
                "state":         co.state.value,
                "decision":      co.decision.value,
                "question":      co.question,
                "user_response": co.user_response,
                "g0_after":      co.g0_after,
            }
        return {
            "status":      self.status,
            "g0_initial":  self.g0_initial,
            "c1":          _fmt(self.c1),
            "c2":          _fmt(self.c2),
            "stop_reason": self.stop_reason,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _noop_response(question: str, g0: dict) -> str:
    """Default user-response function: returns '' (no clarification given)."""
    return ""


def _clarifying_question(g0: dict, role: str) -> str:
    """Generate a human-readable clarifying question from the current G₀."""
    label = g0[role]["label"]
    templates = {
        "target":      f"어떤 '{label}'을(를) 집어야 하나요?",
        "destination": f"어떤 '{label}'에 놓아야 하나요?",
    }
    return templates.get(role, f"'{label}'에 대해 명확히 해주세요.")


def _update_g0(
    image_path: str | Path,
    task_description: str,
    user_response: str,
    handler: Any,
    session_id: str,
) -> dict[str, Any]:
    """Rolling G₀ update after user clarification.

    Runs: reset → query(task) → respond(user_answer) → detect → new G₀.
    This is the AmbRes two-step dialogue with a real user answer instead
    of the forced empty-string used at t₀.
    """
    image_arr, image_shape = _load_image(image_path)
    tensors = [image_arr]

    _unwrap_handler_result(handler.handle("reset", {}, [], session_id), "reset")

    step1 = _unwrap_handler_result(
        handler.handle("query", {"task_description": task_description}, tensors, session_id),
        "query",
    )
    step2 = _unwrap_handler_result(
        handler.handle("respond", {"response": user_response}, [], session_id),
        "respond",
    )

    object_list = _object_list_from_step(step2) or _object_list_from_step(step1)
    roles = parse_roles(object_list, task_description)
    labels = [roles["target"], roles["destination"]]

    detections = _unwrap_handler_result(
        handler.handle("detect", {"objects": labels}, tensors, session_id),
        "detect",
    )["detections"]

    return {
        "target": {
            "label": roles["target"],
            "coord": _first_coord(detections, roles["target"]),
        },
        "destination": {
            "label": roles["destination"],
            "coord": _first_coord(detections, roles["destination"]),
        },
        "image_shape": image_shape,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    image_t0: str | Path,
    image_c1: str | Path,
    image_c2: str | Path,
    task_description: str,
    *,
    handler: Any = None,
    model_type: str = "fs_prompt",
    adapter_ckpt: str = "",
    threshold: float = 50.0,
    user_response_fn: UserResponseFn | None = None,
    session_prefix: str = "pipeline",
    allow_ambiguous: bool = False,
) -> PipelineResult:
    """Run the full t₀ → C1 → C2 pipeline.

    Args:
        image_t0:         Image at t₀ (for initial G₀ extraction).
        image_c1:         Image at C1 checkpoint (pre-pick).
        image_c2:         Image at C2 checkpoint (pre-place).
        task_description: Natural-language task instruction.
        handler:          Pre-built AmbResHandler. Created internally if None.
        model_type:       "fs_prompt" or "finetune" (ignored when handler given).
        adapter_ckpt:     Adapter checkpoint ID (ignored when handler given).
        threshold:        Euclidean pixel distance threshold for consistency check.
        user_response_fn: Called when decision==ASK.
                          Signature: (clarifying_question: str, g0: dict) → str.
                          Return "" to skip rolling update.
                          Defaults to no-op (always returns "").
        session_prefix:   Prefix for AmbRes session IDs.

    Returns:
        PipelineResult with status in {"complete", "stopped", "initial_ambiguous"}.
    """
    if handler is None:
        handler = _make_handler(model_type, adapter_ckpt, use_detection=False)
    if user_response_fn is None:
        user_response_fn = _noop_response

    # ── t₀: G₀ extraction ──────────────────────────────────────────────────
    try:
        g0 = extract_g0(
            image_t0, task_description,
            handler=handler,
            session_id=f"{session_prefix}_t0",
            allow_ambiguous=allow_ambiguous,
        )
    except RuntimeError as exc:
        if "ambiguity=true" in str(exc):
            return PipelineResult(
                status="initial_ambiguous",
                g0_initial=None,
                c1=None,
                c2=None,
                stop_reason=str(exc),
            )
        raise

    g0_current = g0

    # ── C1: pre-pick checkpoint ─────────────────────────────────────────────
    c1_det = get_checkpoint_detections(
        image_c1, g0_current, handler,
        session_id=f"{session_prefix}_c1",
    )
    c1_state, c1_decision = check_grounding(g0_current, c1_det, "C1", threshold)

    g0_after_c1 = g0_current
    c1_user_response = ""
    c1_question = ""

    if c1_decision == Decision.STOP:
        return PipelineResult(
            status="stopped",
            g0_initial=g0,
            c1=CheckpointOutcome("C1", c1_state, c1_decision, c1_det,
                                  g0_current, g0_current),
            c2=None,
            stop_reason=f"C1 {c1_state.value} → STOP",
        )

    if c1_decision == Decision.ASK:
        c1_question = f"[C1 {c1_state.value}] {_clarifying_question(g0_current, 'target')}"
        c1_user_response = user_response_fn(c1_question, g0_current)
        if c1_user_response:
            g0_after_c1 = _update_g0(
                image_c1, task_description, c1_user_response,
                handler, f"{session_prefix}_c1_update",
            )

    c1_outcome = CheckpointOutcome(
        "C1", c1_state, c1_decision, c1_det,
        g0_before=g0_current, g0_after=g0_after_c1,
        user_response=c1_user_response,
        question=c1_question,
    )
    g0_current = g0_after_c1

    # ── C2: pre-place checkpoint ────────────────────────────────────────────
    c2_det = get_checkpoint_detections(
        image_c2, g0_current, handler,
        session_id=f"{session_prefix}_c2",
    )
    c2_state, c2_decision = check_grounding(g0_current, c2_det, "C2", threshold)

    g0_after_c2 = g0_current
    c2_user_response = ""
    c2_question = ""

    if c2_decision == Decision.STOP:
        return PipelineResult(
            status="stopped",
            g0_initial=g0,
            c1=c1_outcome,
            c2=CheckpointOutcome("C2", c2_state, c2_decision, c2_det,
                                  g0_current, g0_current),
            stop_reason=f"C2 {c2_state.value} → STOP",
        )

    if c2_decision == Decision.ASK:
        c2_question = f"[C2 {c2_state.value}] {_clarifying_question(g0_current, 'destination')}"
        c2_user_response = user_response_fn(c2_question, g0_current)
        if c2_user_response:
            g0_after_c2 = _update_g0(
                image_c2, task_description, c2_user_response,
                handler, f"{session_prefix}_c2_update",
            )

    c2_outcome = CheckpointOutcome(
        "C2", c2_state, c2_decision, c2_det,
        g0_before=g0_current, g0_after=g0_after_c2,
        user_response=c2_user_response,
        question=c2_question,
    )

    return PipelineResult(
        status="complete",
        g0_initial=g0,
        c1=c1_outcome,
        c2=c2_outcome,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full t₀→C1→C2 pipeline (non-interactive, for testing)."
    )
    parser.add_argument("image_t0")
    parser.add_argument("image_c1")
    parser.add_argument("image_c2")
    parser.add_argument("task_description")
    parser.add_argument("--model-type",    default="fs_prompt",
                        choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt",  default="")
    parser.add_argument("--threshold",     type=float, default=50.0)
    parser.add_argument("--session-prefix", default="pipeline")
    args = parser.parse_args()

    result = run_pipeline(
        args.image_t0, args.image_c1, args.image_c2,
        args.task_description,
        model_type=args.model_type,
        adapter_ckpt=args.adapter_ckpt,
        threshold=args.threshold,
        session_prefix=args.session_prefix,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
