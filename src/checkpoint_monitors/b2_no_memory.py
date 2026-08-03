"""B2 baseline: checkpoint re-query without G0 memory.

B2 runs AmbRes Step 1 independently at each checkpoint. It does not store
or compare the initial grounding, so each checkpoint decision depends only
on the current image and task text.
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
    _ambiguity_from_step,
    _load_image,
    _make_handler,
    _unwrap_handler_result,
)
from monitoring.consistency_monitor import Decision


METHOD_NAME = "B2_NO_MEMORY"


def run_b2_no_memory(
    checkpoint_img: str | Path,
    task_description: str,
    *,
    checkpoint: str = "C1",
    handler: Any = None,
    model_type: str = "fs_prompt",
    adapter_ckpt: str = "",
    session_id: str = "b2_checkpoint",
) -> BaselineResult:
    """Run B2 at one checkpoint with no G0 memory.

    Args:
        checkpoint_img: Checkpoint image, e.g. C1 or C2 snapshot.
        task_description: Natural-language task instruction.
        checkpoint: Checkpoint label stored in metadata ("C1" or "C2").
        handler: Pre-built AmbResHandler. Created internally if None.
        model_type: AmbRes model type, used only when handler is None.
        adapter_ckpt: AmbRes adapter checkpoint, used only when handler is None.
        session_id: AmbRes session identifier.

    Returns:
        BaselineResult with Decision.ASK if the current checkpoint scene is
        ambiguous, otherwise Decision.CONTINUE.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

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

    metadata = {
        "checkpoint": checkpoint,
        "image_shape": image_shape,
        "session_id": session_id,
        "uses_checkpoint": True,
        "stores_g0": False,
    }

    if _ambiguity_from_step(step1):
        question = str(step1.get("clarifying_question", ""))
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason=f"{checkpoint} scene is ambiguous",
            question=question,
            raw_output=step1,
            metadata=metadata,
        )

    return BaselineResult(
        method=METHOD_NAME,
        decision=Decision.CONTINUE,
        reason=f"{checkpoint} scene is unambiguous without memory",
        raw_output=step1,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B2 No-memory checkpoint baseline.")
    parser.add_argument("checkpoint_img")
    parser.add_argument("task_description")
    parser.add_argument("--checkpoint", default="C1", choices=["C1", "C2"])
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--session-id", default="b2_checkpoint")
    args = parser.parse_args()

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    result = run_b2_no_memory(
        args.checkpoint_img,
        args.task_description,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        adapter_ckpt=args.adapter_ckpt,
        session_id=args.session_id,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
