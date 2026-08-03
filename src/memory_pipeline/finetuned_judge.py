"""
src.memory_pipeline.finetuned_judge
=====================================
Fine-tuned Molmo judge integrated with episode memory.

Combines:
  - GDino coordinates (t0 baseline + checkpoint detections)
  - Episode memory context (SceneMonitor changes, past decisions)
  - Fine-tuned Molmo decision (trained on CONTINUE/ASK/STOP taxonomy)

This isolates the contribution of memory context on top of a fine-tuned model,
enabling direct comparison:
  cond1 (eval_holdout): FT + GDino, no memory
  memory_finetuned:     FT + GDino + memory context
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from monitoring.consistency_monitor import Decision, GroundingState

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
_FT_CKPT_DEFAULT = str(_REPO / "COSE461-proj/ckpt/grounding_ft/BvFsfsf4/checkpoint-162")
_AMBRES_CKPT = "/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
_MODEL_ID = "allenai/Molmo-7B-D-0924"
_HF_CACHE = "/workspace/hf_cache"

_TAXONOMY = """\
Use the following grounding state taxonomy to guide your judgment:

  CLEAR             — target and destination are uniquely identifiable at expected
                      positions; the robot can safely continue.
  AMBIGUOUS_TARGET  — multiple target candidates exist or the target has moved to an
                      unexpected position; ask the user which one to pick.
  INVALID_TARGET    — the target object cannot be found in the scene; the robot must
                      stop immediately (safety-critical).
  AMBIGUOUS_DESTINATION — multiple destination candidates or destination is unclear;
                      ask the user for clarification.
  INVALID_DESTINATION — the destination is no longer visible; the robot must
                      stop immediately (safety-critical).

Decision mapping:
  CLEAR                    → respond not ambiguous (robot CONTINUES)
  INVALID_TARGET           → respond ambiguous, set stop_bool=True
  INVALID_DESTINATION      → respond ambiguous, set stop_bool=True
  Any other non-CLEAR      → respond ambiguous and ask for user clarification"""


def _fmt(c: list | None) -> str:
    return f"({int(c[0])},{int(c[1])})" if c else "unknown"


def _fmtl(cs: list) -> str:
    return ", ".join(f"({int(c[0])},{int(c[1])})" for c in cs[:5]) if cs else "none"


def build_prompt(
    task_with_context: str,
    target_label: str,
    dest_label: str,
    g0_target: list | None,
    g0_dest: list | None,
    ck_target_coords: list,
    ck_dest_coords: list,
    checkpoint: str,
) -> str:
    """Build fine-tuned model prompt with memory context included."""
    return "\n".join([
        f"Original robot task and history:",
        f"{task_with_context}",
        "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{target_label}" was at pixel {_fmt(g0_target)}',
        f'  - Destination "{dest_label}" was at pixel {_fmt(g0_dest)}',
        "",
        f"Checkpoint {checkpoint} — GroundingDINO detection results (authoritative):",
        f'  - "{target_label}": {len(ck_target_coords)} detection(s) at [{_fmtl(ck_target_coords)}]',
        f'  - "{dest_label}": {len(ck_dest_coords)} detection(s) at [{_fmtl(ck_dest_coords)}]',
        "",
        "Based on the detection results and memory context above, assess whether "
        "the task grounding is valid.",
        "",
        _TAXONOMY,
    ])


def _decision_to_state(decision: Decision, checkpoint: str) -> GroundingState:
    if decision == Decision.CONTINUE:
        return GroundingState.CLEAR
    if decision == Decision.ASK:
        return (GroundingState.AMBIGUOUS_TARGET if checkpoint == "C1"
                else GroundingState.AMBIGUOUS_DESTINATION)
    return (GroundingState.INVALID_TARGET if checkpoint == "C1"
            else GroundingState.INVALID_DESTINATION)


class FinetunedJudge:
    """Fine-tuned Molmo model for memory-aware grounding decisions."""

    def __init__(
        self,
        ft_ckpt: str = _FT_CKPT_DEFAULT,
        ambres_ckpt: str = _AMBRES_CKPT,
        hf_cache: str = _HF_CACHE,
    ):
        import os
        os.environ["HF_HOME"] = hf_cache

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        logger.info("Loading base Molmo...")
        self.processor = AutoProcessor.from_pretrained(
            _MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda"
        )
        logger.info("Merging AmbRes LoRA: %s", ambres_ckpt)
        model = PeftModel.from_pretrained(model, ambres_ckpt)
        model = model.merge_and_unload()
        logger.info("Applying fine-tuned LoRA: %s", ft_ckpt)
        model = PeftModel.from_pretrained(model, ft_ckpt)
        model.eval()
        self.model = model
        self._gen_cfg = GenerationConfig(
            do_sample=False, max_new_tokens=256, stop_strings="<|endoftext|>"
        )
        logger.info("FinetunedJudge ready.")

    def judge(
        self,
        image_path: str | Path,
        task_with_context: str,
        g0: dict[str, Any],
        gdino_t0_detections: dict[str, Any],
        gdino_ck_detections: dict[str, Any],
        checkpoint: str,
    ) -> tuple[Decision, GroundingState, str, str]:
        """
        Run fine-tuned Molmo with memory context for CONTINUE/ASK/STOP decision.

        Returns (decision, grounding_state, explanation, clarifying_question).
        """
        import os
        sys.path.insert(0, str(_REPO / "AmbRes"))
        from ambres.training.data import process_msg_list

        target_label = g0["target"]["label"]
        dest_label   = g0["destination"]["label"]

        g0_target = g0["target"].get("coord")
        g0_dest   = g0["destination"].get("coord")

        ck_target = [[int(c[0]), int(c[1])] for c in (gdino_ck_detections.get(target_label) or [])
                     if isinstance(c, (list, tuple)) and len(c) == 2]
        ck_dest   = [[int(c[0]), int(c[1])] for c in (gdino_ck_detections.get(dest_label) or [])
                     if isinstance(c, (list, tuple)) and len(c) == 2]

        prompt = build_prompt(
            task_with_context, target_label, dest_label,
            g0_target, g0_dest, ck_target, ck_dest, checkpoint,
        )

        pil_img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": f"<image>\nTASK DESCRIPTION: {prompt}"}]
        inputs = process_msg_list(messages, [pil_img], self.processor)
        inputs = {k: v.to(device=self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"]      = inputs["images"].to(self.model.dtype)
        inputs["image_masks"] = inputs["image_masks"].to(self.model.dtype)

        with torch.no_grad():
            out = self.model.generate_from_batch(
                inputs, self._gen_cfg,
                tokenizer=self.processor.tokenizer,
                return_dict_in_generate=True,
            )
        tokens = out.sequences[0, inputs["input_ids"].size(1):]
        raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        logger.info("FT judge %s raw: %s", checkpoint, raw)

        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("FT judge parse failed: %r — defaulting CONTINUE", raw)
            data = {"task_ambiguous": False, "stop_bool": False, "clarifying_question": ""}

        ambig = bool(data.get("task_ambiguous", data.get("ambiguity_bool", False)))
        stop  = bool(data.get("stop_bool", False))
        explanation = data.get("clarifying_question", "")

        if not ambig:
            decision = Decision.CONTINUE
        elif stop:
            decision = Decision.STOP
        else:
            decision = Decision.ASK

        state    = _decision_to_state(decision, checkpoint)
        question = f"[{checkpoint}] {explanation}" if decision == Decision.ASK and explanation else ""

        logger.info("FT judge %s: decision=%s state=%s", checkpoint, decision.value, state.value)
        return decision, state, explanation, question


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_instance: FinetunedJudge | None = None


def ask_finetuned_judge(
    image_path: str | Path,
    task_with_context: str,
    g0: dict[str, Any],
    gdino_t0_detections: dict[str, Any],
    gdino_ck_detections: dict[str, Any],
    checkpoint: str,
    ft_ckpt: str = _FT_CKPT_DEFAULT,
) -> tuple[Decision, GroundingState, str, str]:
    """
    Fine-tuned Molmo judge with episode memory context.

    Unlike cond1 (eval_holdout), this receives task_with_context which
    includes the episode memory — enabling memory-conditioned decisions.
    """
    global _instance
    if _instance is None:
        _instance = FinetunedJudge(ft_ckpt=ft_ckpt)
    return _instance.judge(
        image_path, task_with_context, g0,
        gdino_t0_detections, gdino_ck_detections, checkpoint,
    )
