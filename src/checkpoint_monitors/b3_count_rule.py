"""B3 baseline: attribute-aware candidate count rule.

B3 calls AmbRes Step 1 to obtain the current object list, ignores the VLM's
own ambiguity flag, and applies exact full-label counts for target and
destination candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.common import BaselineResult
from extraction.ambres_g0_extractor import (
    _load_image,
    _make_handler,
    _object_list_from_step,
    _unwrap_handler_result,
)
from monitoring.consistency_monitor import Decision


METHOD_NAME = "B3_ATTRIBUTE_AWARE_COUNT"


def _count_exact(objects: list[str], label: str) -> int:
    return sum(1 for obj in objects if obj == label)


def run_b3_count_rule(
    checkpoint_img: str | Path,
    task_description: str,
    target_label: str,
    destination_label: str,
    *,
    checkpoint: str = "C1",
    handler: Any = None,
    model_type: str = "fs_prompt",
    adapter_ckpt: str = "",
    session_id: str = "b3_checkpoint",
) -> BaselineResult:
    """Run B3 at one checkpoint using attribute-aware exact label counts.

    Args:
        checkpoint_img: Checkpoint image, e.g. C1 or C2 snapshot.
        task_description: Natural-language task instruction.
        target_label: Full target label, e.g. "red block".
        destination_label: Full destination label, e.g. "gray tray".
        checkpoint: Checkpoint label stored in metadata ("C1" or "C2").
        handler: Pre-built AmbResHandler. Created internally if None.
        model_type: AmbRes model type, used only when handler is None.
        adapter_ckpt: AmbRes adapter checkpoint, used only when handler is None.
        session_id: AmbRes session identifier.

    Returns:
        BaselineResult following the count rule:
        target missing -> STOP, target multiple -> ASK,
        destination missing/multiple -> ASK, otherwise CONTINUE.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")
    if not target_label:
        raise ValueError("target_label must be non-empty")
    if not destination_label:
        raise ValueError("destination_label must be non-empty")

    image_arr, image_shape = _load_image(checkpoint_img)
    if handler is None:
        handler = _make_handler(model_type, adapter_ckpt, use_detection=False)

    _unwrap_handler_result(handler.handle("reset", {}, [], session_id), "reset")
    step1 = _unwrap_handler_result(
        handler.handle(
            "query",
            {"task_description": task_description},
            [image_arr],
            session_id,
        ),
        "query",
    )
    object_list = _object_list_from_step(step1)

    n_target = _count_exact(object_list, target_label)
    n_dest = _count_exact(object_list, destination_label)
    metadata = {
        "checkpoint": checkpoint,
        "image_shape": image_shape,
        "session_id": session_id,
        "uses_checkpoint": True,
        "stores_g0": False,
        "target_label": target_label,
        "destination_label": destination_label,
        "n_target": n_target,
        "n_destination": n_dest,
        "object_list": object_list,
        "ignores_vlm_ambiguity": True,
    }

    if n_target == 0:
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.STOP,
            reason=f"{target_label} not found in scene",
            raw_output=step1,
            metadata=metadata,
        )
    if n_target > 1:
        question = f"Multiple {target_label}s found. Which one?"
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason=f"Multiple {target_label}s found",
            question=question,
            raw_output=step1,
            metadata=metadata,
        )
    if n_dest == 0:
        question = f"{destination_label} not found. Where should I place it?"
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason=f"{destination_label} not found in scene",
            question=question,
            raw_output=step1,
            metadata=metadata,
        )
    if n_dest > 1:
        question = f"Multiple {destination_label}s found. Which one?"
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason=f"Multiple {destination_label}s found",
            question=question,
            raw_output=step1,
            metadata=metadata,
        )

    return BaselineResult(
        method=METHOD_NAME,
        decision=Decision.CONTINUE,
        reason="Exactly one target and one destination found",
        raw_output=step1,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B3 attribute-aware count baseline.")
    parser.add_argument("checkpoint_img")
    parser.add_argument("task_description")
    parser.add_argument("target_label")
    parser.add_argument("destination_label")
    parser.add_argument("--checkpoint", default="C1", choices=["C1", "C2"])
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--session-id", default="b3_checkpoint")
    args = parser.parse_args()

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    result = run_b3_count_rule(
        args.checkpoint_img,
        args.task_description,
        args.target_label,
        args.destination_label,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        adapter_ckpt=args.adapter_ckpt,
        session_id=args.session_id,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
