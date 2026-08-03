"""VLM-as-judge ablation baselines.

Three conditions comparing visual input vs structured detection text:

  VLM_IMG_ONLY   — checkpoint image + taxonomy, NO detection context
                   Tests: can visual perception alone judge grounding?

  VLM_TAX        — checkpoint image + Molmo detections + taxonomy
                   Tests: does detection text help alongside visual?

  VLM_TAX_DINO   — checkpoint image + DINO detections + "trust DINO" header + taxonomy
                   Tests: does authoritative DINO text override visual bias?

  VLM_BLANK_DINO — blank image + DINO detections + taxonomy
                   Tests: can text-only detection context drive correct decisions?

Decision mapping from AmbRes query output:
  ambiguity=False                          → CONTINUE
  ambiguity=True + "STOP:" in question     → STOP
  ambiguity=True + target count=0          → STOP (fallback)
  ambiguity=True otherwise                 → ASK
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.bundle import DetectionBundle
from baselines.common import BaselineResult
from monitoring.consistency_monitor import Decision


METHOD_IMG_ONLY  = "VLM_IMG_ONLY"
METHOD_TAX       = "VLM_TAX"
METHOD_TAX_DINO  = "VLM_TAX_DINO"
METHOD_BLANK_DINO = "VLM_BLANK_DINO"

_TAXONOMY_TEXT = """\
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
  INVALID_TARGET           → respond ambiguous and begin your clarifying question with "STOP: "
  INVALID_DESTINATION      → respond ambiguous and begin your clarifying question with "STOP: "
  Any other non-CLEAR      → respond ambiguous and ask for user clarification"""

_DINO_TRUST_HEADER = """\
*** DETECTION RESULTS ARE AUTHORITATIVE ***
The object detection below was performed by GroundingDINO, a precise object detector.
These counts and positions are ground truth. Do NOT re-count objects from the image.
Use the image ONLY for spatial context. Your judgment must be based on the numbers
provided, not on your own visual counting."""


def _format_coords(coords: list[Any]) -> str:
    valid = [c for c in (coords or []) if isinstance(c, (list, tuple)) and len(c) == 2]
    if not valid:
        return "none"
    return ", ".join(f"({int(c[0])},{int(c[1])})" for c in valid[:5])


def _count_valid(coords: list[Any]) -> int:
    return sum(1 for c in (coords or []) if isinstance(c, (list, tuple)) and len(c) == 2)


def _build_prompt(
    task: str,
    g0: dict[str, Any],
    det_at_ck: dict[str, list] | None,
    checkpoint: str,
    use_taxonomy: bool,
    dino_grounded: bool = False,
    img_only: bool = False,
) -> str:
    target_label = g0["target"]["label"]
    dest_label   = g0["destination"]["label"]
    g0_target = g0["target"].get("coord")
    g0_dest   = g0["destination"].get("coord")
    g0_target_str = f"({g0_target[0]},{g0_target[1]})" if g0_target else "unknown"
    g0_dest_str   = f"({g0_dest[0]},{g0_dest[1]})"   if g0_dest   else "unknown"

    lines = []

    if dino_grounded:
        lines += [_DINO_TRUST_HEADER, ""]

    lines += [
        f"Original robot task: {task}",
        "",
        "Initial scene (t0) — stored reference positions:",
        f'  - Target "{target_label}" was at pixel {g0_target_str}',
        f'  - Destination "{dest_label}" was at pixel {g0_dest_str}',
        "",
    ]

    if img_only:
        lines.append(
            "Examine the current checkpoint scene image and assess whether the task "
            "grounding (target + destination) is still valid and unambiguous."
        )
    else:
        n_target = _count_valid(det_at_ck.get(target_label) if det_at_ck else [])
        n_dest   = _count_valid(det_at_ck.get(dest_label)   if det_at_ck else [])
        target_pos = _format_coords(det_at_ck.get(target_label) if det_at_ck else [])
        dest_pos   = _format_coords(det_at_ck.get(dest_label)   if det_at_ck else [])

        ck_label = (
            f"Checkpoint {checkpoint} — GroundingDINO detection results (authoritative):"
            if dino_grounded else
            f"Checkpoint {checkpoint} — current detection results:"
        )
        lines += [
            ck_label,
            f'  - "{target_label}": {n_target} detection(s) at [{target_pos}]',
            f'  - "{dest_label}": {n_dest} detection(s) at [{dest_pos}]',
            "",
        ]

        if dino_grounded:
            lines.append(
                "Based ONLY on the detection counts and positions above "
                "(not your own visual counting), assess whether the task grounding is valid."
            )
        else:
            lines.append(
                "Examine the current checkpoint scene image and assess whether the task "
                "grounding (target + destination) is still valid and unambiguous."
            )

    if use_taxonomy:
        lines += ["", _TAXONOMY_TEXT]

    return "\n".join(lines)


def _parse_output(
    step1: dict[str, Any],
    det_at_ck: dict[str, list] | None,
    target_label: str,
) -> tuple[Decision, str, str]:
    ambiguity = bool(step1.get("ambiguity", step1.get("task_ambiguous", False)))
    stop_bool = bool(step1.get("stop_bool", False))
    clarifying_q = str(step1.get("clarifying_question", "")).strip()

    if not ambiguity:
        return Decision.CONTINUE, "VLM judged grounding still valid", ""

    # Explicit stop_bool field (fine-tuned model output)
    if stop_bool:
        return Decision.STOP, "VLM classified grounding as invalid (stop_bool=True)", ""

    # Fallback: legacy STOP signal from clarifying_question text
    cq_upper = clarifying_q.upper()
    if cq_upper.startswith("STOP:") or "INVALID_TARGET" in cq_upper:
        return Decision.STOP, f"VLM detected invalid target: {clarifying_q}", ""

    # Fallback: target has zero detections
    if det_at_ck is not None and _count_valid(det_at_ck.get(target_label)) == 0:
        return Decision.STOP, f"Target not detected; VLM: {clarifying_q}", ""

    return Decision.ASK, f"VLM detected grounding anomaly: {clarifying_q}", clarifying_q


def run_vlm_judge(
    det_bundle: DetectionBundle,
    checkpoint_img: str | Path,
    task: str,
    handler: Any,
    *,
    checkpoint: str = "C1",
    use_taxonomy: bool = True,
    dino_grounded: bool = False,
    img_only: bool = False,
    use_blank_image: bool = False,
    session_id: str = "vlm_judge",
) -> BaselineResult:
    """Run one VLM-as-judge condition.

    Args:
        det_bundle:      Pre-computed detection bundle.
        checkpoint_img:  Checkpoint scene image path.
        task:            Original task description.
        handler:         Pre-initialized AmbResHandler.
        checkpoint:      "C1" or "C2".
        use_taxonomy:    Inject grounding state taxonomy into prompt.
        dino_grounded:   Use DINO trust header + DINO detections from bundle.
        img_only:        No detection context — image + taxonomy only.
        use_blank_image: Replace checkpoint image with black image.
        session_id:      AmbRes session identifier.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

    from extraction.ambres_g0_extractor import _load_image, _unwrap_handler_result

    g0 = det_bundle.g0
    det_at_ck = None if img_only else det_bundle.det_at(checkpoint)

    if img_only:
        method = METHOD_IMG_ONLY
    elif use_blank_image:
        method = METHOD_BLANK_DINO
    elif dino_grounded:
        method = METHOD_TAX_DINO
    else:
        method = METHOD_TAX

    prompt = _build_prompt(
        task, g0, det_at_ck, checkpoint,
        use_taxonomy=use_taxonomy,
        dino_grounded=dino_grounded,
        img_only=img_only,
    )

    if use_blank_image:
        real_arr, _ = _load_image(checkpoint_img)
        image_arr = np.zeros_like(real_arr)
    else:
        image_arr, _ = _load_image(checkpoint_img)

    _unwrap_handler_result(handler.handle("reset", {}, [], session_id), "reset")
    step1 = _unwrap_handler_result(
        handler.handle("query", {"task_description": prompt}, [image_arr], session_id),
        "query",
    )

    target_label = g0["target"]["label"]
    dest_label   = g0["destination"]["label"]
    decision, reason, question = _parse_output(step1, det_at_ck, target_label)

    n_target = _count_valid(det_at_ck.get(target_label)) if det_at_ck else None
    n_dest   = _count_valid(det_at_ck.get(dest_label))   if det_at_ck else None

    metadata = {
        "checkpoint":      checkpoint,
        "img_only":        img_only,
        "use_blank_image": use_blank_image,
        "dino_grounded":   dino_grounded,
        "use_taxonomy":    use_taxonomy,
        "uses_checkpoint": True,
        "stores_g0":       not img_only,
        "uses_coord":      not img_only,
        "uses_taxonomy":   use_taxonomy,
        "vlm_ambiguity":   step1.get("ambiguity", step1.get("task_ambiguous")),
        "vlm_clarifying_question": step1.get("clarifying_question", ""),
        "n_target_at_ck":  n_target,
        "n_dest_at_ck":    n_dest,
        "session_id":      session_id,
    }

    return BaselineResult(
        method=method,
        decision=decision,
        reason=reason,
        question=question,
        raw_output={"step1": step1, "detections": det_at_ck},
        metadata=metadata,
    )
