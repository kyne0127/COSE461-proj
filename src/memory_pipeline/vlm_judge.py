"""
src.memory_pipeline.vlm_judge
=================================
VLM-as-judge: custom-prompted structured decision using the same Molmo model.

AmbRes is just a wrapper: Molmo + outlines JSON constraints + prompt templates.
Its STEP2 prompt is simply "Is the task ambiguous?" — too vague to be reliable.

This module replaces that single vague question with an explicit judge prompt
that provides detection counts and decision rules, then forces the model to
output a structured {"decision": CONTINUE|ASK|STOP, "explanation": ...} via
the same outlines logit processor that AmbRes uses internally.

Key: handler._model is AmbresStructured (subclass of MolmoChat), which already
holds outlines_tokenizer. We reuse it directly — no new model loaded.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import transformers
import outlines
import outlines.processors
from PIL import Image
from pydantic import BaseModel

from monitoring.consistency_monitor import Decision, GroundingState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema (mirrors AmbiguityReasoning but with explicit decision field)
# ---------------------------------------------------------------------------

class JudgeDecision(BaseModel):
    decision:    Literal["CONTINUE", "ASK", "STOP"]
    explanation: str


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
<image>
TASK: {task}

DETECTION REPORT at {checkpoint}:
  '{target_label}': {t0_target_n} instance(s) at start  →  {ck_target_n} instance(s) now
  '{dest_label}':   {t0_dest_n} instance(s) at start  →  {ck_dest_n} instance(s) now
{changes_line}
DECISION RULES for '{role_label}' ({role_desc}):
  0 instances  →  STOP      (object is missing)
  1 instance   →  CONTINUE  (object is clearly identified)
  2+ instances →  ASK       (ambiguous — robot cannot determine which one)

Apply the rule strictly and provide your decision."""

_ROLE_DESC = {"C1": "target to pick", "C2": "destination to place into"}


# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------

def _decision_to_state(decision: Decision, checkpoint: str) -> GroundingState:
    if decision == Decision.CONTINUE:
        return GroundingState.CLEAR
    if decision == Decision.ASK:
        return (GroundingState.AMBIGUOUS_TARGET if checkpoint == "C1"
                else GroundingState.AMBIGUOUS_DESTINATION)
    return (GroundingState.INVALID_TARGET if checkpoint == "C1"
            else GroundingState.INVALID_DESTINATION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_vlm_judge(
    image_path: str | Path,
    task_description: str,
    g0: dict[str, Any],
    ck_detections: dict[str, Any],
    changes: list[str],
    checkpoint: str,
    handler: Any,
    session_id: str,
) -> tuple[Decision, GroundingState, str, str]:
    """
    Ask Molmo (via handler._model) for CONTINUE/ASK/STOP using a custom prompt
    and outlines-constrained JSON output.

    Reuses handler._model.outlines_tokenizer — no new model is loaded.

    Returns (decision, grounding_state, explanation, clarifying_question).
    """
    target_label = g0["target"]["label"]
    dest_label   = g0["destination"]["label"]

    t0_target_n = len(g0["target"].get("coords") or [])
    t0_dest_n   = len(g0["destination"].get("coords") or [])
    ck_target_n = len([c for c in (ck_detections.get(target_label) or [])
                       if isinstance(c, (list, tuple)) and len(c) == 2])
    ck_dest_n   = len([c for c in (ck_detections.get(dest_label) or [])
                       if isinstance(c, (list, tuple)) and len(c) == 2])

    role_label   = target_label if checkpoint == "C1" else dest_label
    role_desc    = _ROLE_DESC.get(checkpoint, checkpoint)
    changes_line = ("CHANGES: " + "; ".join(changes) + "\n") if changes else ""

    prompt = _JUDGE_PROMPT.format(
        task=task_description,
        checkpoint=checkpoint,
        target_label=target_label,
        dest_label=dest_label,
        t0_target_n=t0_target_n,
        t0_dest_n=t0_dest_n,
        ck_target_n=ck_target_n,
        ck_dest_n=ck_dest_n,
        changes_line=changes_line,
        role_label=role_label,
        role_desc=role_desc,
    )

    # Access Molmo + outlines directly via handler._model
    molmo = handler._model
    pil_img = Image.open(image_path).convert("RGB")
    molmo.set_image([pil_img], reset_chat=True)
    molmo.add_message("user", prompt)

    judge_processor = transformers.LogitsProcessorList([
        outlines.processors.JSONLogitsProcessor(JudgeDecision, molmo.outlines_tokenizer)
    ])

    raw = molmo.run_molmo_np(
        molmo.messages,
        generate_kwargs={"logits_processor": judge_processor},
        do_sample=True,
    )
    molmo.add_message("assistant", raw)

    logger.info("VLM judge %s raw: %s", checkpoint, raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        warnings.warn(f"VLM judge JSON parse failed: {raw!r} — defaulting CONTINUE")
        data = {"decision": "CONTINUE", "explanation": "parse error"}

    raw_decision = data.get("decision", "CONTINUE").upper()
    explanation  = data.get("explanation", "")

    decision = {"CONTINUE": Decision.CONTINUE,
                "ASK":      Decision.ASK,
                "STOP":     Decision.STOP}.get(raw_decision, Decision.CONTINUE)
    state    = _decision_to_state(decision, checkpoint)

    question = ""
    if decision == Decision.ASK and explanation:
        question = f"[{checkpoint}] {explanation}"

    logger.info(
        "VLM judge %s: decision=%s state=%s  "
        "(t0 %s:%d/%s:%d → ck %s:%d/%s:%d)",
        checkpoint, decision.value, state.value,
        target_label, t0_target_n, dest_label, t0_dest_n,
        target_label, ck_target_n, dest_label, ck_dest_n,
    )

    return decision, state, explanation, question
