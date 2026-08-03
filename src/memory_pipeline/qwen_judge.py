"""
src.memory_pipeline.qwen_judge
================================
Qwen2-VL-7B-Instruct as a memory-aware checkpoint judge/gate/detector.

Three modes:

1. ask_qwen_judge  — full judge (replaces check_grounding entirely)
   Receives task_with_context (memory included) + image + detection counts
   → outputs CONTINUE/ASK/STOP directly.

2. ask_qwen_gate   — gate on top of check_grounding
   Called ONLY when check_grounding returns ASK.
   Asks Qwen: "given task description + memory context, can you resolve
   this ambiguity without asking the human?"
   → YES: override ASK → CONTINUE
   → NO:  keep ASK
   STOP is never passed here — object disappearance is handled by check_grounding.

3. ask_qwen_detect — memory-conditioned detection (replaces Molmo detection)
   Called when movement is detected in SceneMonitor changes.
   Molmo cannot incorporate memory context; Qwen can.
   Provides: "object was at [A], memory shows movement — find it now."
   → Returns detection coords in Molmo format {label: [[x, y], ...]}
   → Fed into check_grounding_ensemble as molmo_detections_ck.
   This ablation isolates the contribution of memory-aware detection.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image

from monitoring.consistency_monitor import Decision, GroundingState

logger = logging.getLogger(__name__)


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
# Prompts — judge mode (full decision)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a robot pick-and-place task monitor. At each checkpoint you assess \
whether the robot can proceed.

Output ONLY a JSON object — no prose, no markdown, no code fences:
{"decision": "CONTINUE", "explanation": "..."}
{"decision": "ASK",      "explanation": "..."}
{"decision": "STOP",     "explanation": "..."}

Decision rules for the role being checked:
  0 detected instances → STOP   (object missing, cannot proceed)
  1 detected instance  → CONTINUE (object clearly identified)
  2+ detected instances → ASK  (ambiguous — UNLESS the task or memory context
                                 uniquely identifies one instance)

Important: if the memory context includes a user clarification that resolves
the current ambiguity, output CONTINUE even if count ≥ 2."""

_JUDGE_USER_PROMPT = """\
{task_with_context}

---
DETECTION REPORT at {checkpoint}:
  '{target_label}': {t0_target_n} instance(s) at start  →  {ck_target_n} instance(s) now
  '{dest_label}':   {t0_dest_n} instance(s) at start  →  {ck_dest_n} instance(s) now
{changes_line}
ROLE TO CHECK: '{role_label}' ({role_desc})

Look at the image and the memory context above. Apply the decision rules \
and output your JSON decision."""

_JUDGE_ROLE_DESC = {"C1": "target to pick", "C2": "destination to place into"}


# ---------------------------------------------------------------------------
# Prompts — gate mode (resolve ASK only)
# ---------------------------------------------------------------------------

_GATE_SYSTEM_PROMPT = """\
You are a robot task monitor. The automated detection system found multiple \
instances of an object and flagged it as ambiguous.

Your job: determine if the task description or memory context already \
uniquely identifies which instance to use — making human clarification unnecessary.

Output ONLY this JSON (no prose, no markdown):
{"can_resolve": true,  "explanation": "..."}
{"can_resolve": false, "explanation": "..."}

Guidelines:
- can_resolve = true  if the task names a distinguishing attribute (color,
  size, position) OR memory shows a prior clarification that applies here.
- can_resolve = false if multiple instances are truly interchangeable given
  the task and memory."""

_GATE_USER_PROMPT = """\
{task_with_context}

---
CHECKPOINT {checkpoint} — ambiguity detected:
  '{role_label}' ({role_desc}): {ck_role_n} instance(s) found now

QUESTION: Based on the task description and memory context above, can you \
uniquely identify which '{role_label}' to use WITHOUT asking the human?

Look at the image. Output your JSON decision."""

_GATE_ROLE_DESC = {"C1": "target to pick", "C2": "destination to place into"}


# ---------------------------------------------------------------------------
# Prompts — detect mode (memory-conditioned object localization)
# ---------------------------------------------------------------------------

_DETECT_SYSTEM_PROMPT = """\
You are a robot vision assistant. Locate specific objects in the image \
and return their current pixel coordinates.

Output ONLY this JSON (no prose, no markdown):
{"target": [x, y], "destination": [x, y]}

Use null if an object is not visible:
{"target": null, "destination": [x, y]}

Coordinates must be the center of the object as integer [x, y] pixels."""

_DETECT_USER_PROMPT = """\
{task_with_context}

---
OBJECT LOCATIONS AT START (t0):
  target '{target_label}': was at {t0_target_coord}
  destination '{dest_label}': was at {t0_dest_coord}

CHANGES DETECTED: {changes_str}

The object may have moved from its original position. \
Look carefully at the entire image — do not assume it is at the start position.
Return the current center [x, y] of each object, or null if not visible."""


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class QwenJudge:
    """Lazy-loaded Qwen2-VL-7B-Instruct judge/gate."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        logger.info("Loading Qwen2-VL from %s …", model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Qwen2-VL ready.")

    def _generate(self, messages: list, pil_img: Image.Image, max_new_tokens: int) -> str:
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def judge(
        self,
        image_path: str | Path,
        task_with_context: str,
        g0: dict[str, Any],
        ck_detections: dict[str, Any],
        changes: list[str],
        checkpoint: str,
    ) -> tuple[Decision, GroundingState, str, str]:
        """Full judge: returns CONTINUE/ASK/STOP from scratch."""
        target_label = g0["target"]["label"]
        dest_label   = g0["destination"]["label"]

        t0_target_n = len(g0["target"].get("coords") or [])
        t0_dest_n   = len(g0["destination"].get("coords") or [])
        ck_target_n = len([c for c in (ck_detections.get(target_label) or [])
                           if isinstance(c, (list, tuple)) and len(c) == 2])
        ck_dest_n   = len([c for c in (ck_detections.get(dest_label) or [])
                           if isinstance(c, (list, tuple)) and len(c) == 2])

        role_label   = target_label if checkpoint == "C1" else dest_label
        role_desc    = _JUDGE_ROLE_DESC.get(checkpoint, checkpoint)
        changes_line = ("CHANGES: " + "; ".join(changes) + "\n") if changes else ""

        user_text = _JUDGE_USER_PROMPT.format(
            task_with_context=task_with_context,
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

        pil_img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text":  user_text},
            ]},
        ]

        raw = self._generate(messages, pil_img, max_new_tokens=256)
        logger.info("Qwen judge %s raw: %s", checkpoint, raw)

        try:
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Qwen judge JSON parse failed: %r — defaulting CONTINUE", raw)
            data = {"decision": "CONTINUE", "explanation": "parse error"}

        raw_decision = data.get("decision", "CONTINUE").upper()
        explanation  = data.get("explanation", "")

        decision = {"CONTINUE": Decision.CONTINUE,
                    "ASK":      Decision.ASK,
                    "STOP":     Decision.STOP}.get(raw_decision, Decision.CONTINUE)
        state    = _decision_to_state(decision, checkpoint)
        question = f"[{checkpoint}] {explanation}" if decision == Decision.ASK and explanation else ""

        logger.info(
            "Qwen judge %s: decision=%s state=%s  "
            "(t0 %s:%d/%s:%d → ck %s:%d/%s:%d)",
            checkpoint, decision.value, state.value,
            target_label, t0_target_n, dest_label, t0_dest_n,
            target_label, ck_target_n, dest_label, ck_dest_n,
        )
        return decision, state, explanation, question

    def gate(
        self,
        image_path: str | Path,
        task_with_context: str,
        g0: dict[str, Any],
        ck_detections: dict[str, Any],
        checkpoint: str,
    ) -> tuple[bool, str]:
        """
        Gate: called only when check_grounding returned ASK.
        Returns (can_resolve, explanation).
        can_resolve=True → caller should override ASK → CONTINUE.
        """
        target_label = g0["target"]["label"]
        dest_label   = g0["destination"]["label"]
        role_label   = target_label if checkpoint == "C1" else dest_label
        role_desc    = _GATE_ROLE_DESC.get(checkpoint, checkpoint)

        ck_role_n = len([
            c for c in (ck_detections.get(role_label) or [])
            if isinstance(c, (list, tuple)) and len(c) == 2
        ])

        user_text = _GATE_USER_PROMPT.format(
            task_with_context=task_with_context,
            checkpoint=checkpoint,
            role_label=role_label,
            role_desc=role_desc,
            ck_role_n=ck_role_n,
        )

        pil_img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": _GATE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text":  user_text},
            ]},
        ]

        raw = self._generate(messages, pil_img, max_new_tokens=128)
        logger.info("Qwen gate %s raw: %s", checkpoint, raw)

        try:
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Qwen gate JSON parse failed: %r — keeping ASK", raw)
            return False, "parse error"

        can_resolve = bool(data.get("can_resolve", False))
        explanation = data.get("explanation", "")

        logger.info(
            "Qwen gate %s: can_resolve=%s  role=%s(%d instances)",
            checkpoint, can_resolve, role_label, ck_role_n,
        )
        return can_resolve, explanation

    def detect(
        self,
        image_path: str | Path,
        task_with_context: str,
        g0: dict[str, Any],
        changes: list[str],
    ) -> dict[str, list[list[int]]]:
        """
        Memory-conditioned detection: replaces Molmo when movement is detected.

        Molmo cannot incorporate memory context about where an object moved.
        Qwen reads task_with_context (which includes movement history) and
        searches the whole image — not just the t0 position.

        Returns detections in Molmo format: {label: [[x, y], ...]}
        Empty list means object not found (→ check_grounding will see count=0).
        """
        target_label = g0["target"]["label"]
        dest_label   = g0["destination"]["label"]

        t0_target_coord = g0["target"].get("coord")
        t0_dest_coord   = g0["destination"].get("coord")
        changes_str = "; ".join(changes) if changes else "none"

        user_text = _DETECT_USER_PROMPT.format(
            task_with_context=task_with_context,
            target_label=target_label,
            dest_label=dest_label,
            t0_target_coord=t0_target_coord,
            t0_dest_coord=t0_dest_coord,
            changes_str=changes_str,
        )

        pil_img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": _DETECT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text":  user_text},
            ]},
        ]

        raw = self._generate(messages, pil_img, max_new_tokens=64)
        logger.info("Qwen detect raw: %s", raw)

        try:
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Qwen detect JSON parse failed: %r — returning empty", raw)
            return {target_label: [], dest_label: []}

        def _parse_coord(val: Any) -> list[list[int]]:
            if val is None:
                return []
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    return [[int(val[0]), int(val[1])]]
                except (TypeError, ValueError):
                    return []
            return []

        target_coords = _parse_coord(data.get("target"))
        dest_coords   = _parse_coord(data.get("destination"))

        logger.info(
            "Qwen detect: target=%s@%s  dest=%s@%s",
            target_label, target_coords,
            dest_label, dest_coords,
        )

        return {target_label: target_coords, dest_label: dest_coords}


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_instance: QwenJudge | None = None


def _get_instance(model_name: str = "Qwen/Qwen2-VL-7B-Instruct") -> QwenJudge:
    global _instance
    if _instance is None:
        _instance = QwenJudge(model_name)
    return _instance


def ask_qwen_judge(
    image_path: str | Path,
    task_with_context: str,
    g0: dict[str, Any],
    ck_detections: dict[str, Any],
    changes: list[str],
    checkpoint: str,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
) -> tuple[Decision, GroundingState, str, str]:
    """Full judge — replaces check_grounding. Uses memory-conditioned task."""
    return _get_instance(model_name).judge(
        image_path, task_with_context, g0, ck_detections, changes, checkpoint
    )


def ask_qwen_gate(
    image_path: str | Path,
    task_with_context: str,
    g0: dict[str, Any],
    ck_detections: dict[str, Any],
    checkpoint: str,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
) -> tuple[bool, str]:
    """
    Gate — called only when check_grounding returned ASK.
    Returns (can_resolve, explanation).
    If can_resolve=True, caller overrides ASK → CONTINUE.
    STOP cases are never passed here.
    """
    return _get_instance(model_name).gate(
        image_path, task_with_context, g0, ck_detections, checkpoint
    )


def ask_qwen_detect(
    image_path: str | Path,
    task_with_context: str,
    g0: dict[str, Any],
    changes: list[str],
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
) -> dict[str, list[list[int]]]:
    """
    Memory-conditioned detection — replaces Molmo when movement is detected.

    Called when SceneMonitor changes contain movement (e.g. 'moved 157px').
    Qwen reads the memory context and searches the whole image for the object,
    not just the t0 position.

    Returns {label: [[x, y], ...]} in Molmo detection format.
    """
    return _get_instance(model_name).detect(
        image_path, task_with_context, g0, changes
    )
