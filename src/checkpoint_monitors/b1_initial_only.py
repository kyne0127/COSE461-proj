"""B1 baseline: AmbResVLM initial-only decision.

B1 runs AmbRes Step 1 once at t0. It does not store G0, inspect
checkpoints, or compare coordinates. A clear initial scene is normalized
to Decision.CONTINUE for experiment metrics.
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


METHOD_NAME = "B1_INITIAL_ONLY"


def run_b1_initial_only(
    image_path: str | Path,
    task_description: str,
    *,
    handler: Any = None,
    model_type: str = "fs_prompt",
    adapter_ckpt: str = "",
    session_id: str = "b1_t0",
) -> BaselineResult:
    """Run the B1 Initial-only baseline.

    Args:
        image_path: Initial t0 image.
        task_description: Natural-language task instruction.
        handler: Pre-built AmbResHandler. Created internally if None.
        model_type: AmbRes model type, used only when handler is None.
        adapter_ckpt: AmbRes adapter checkpoint, used only when handler is None.
        session_id: AmbRes session identifier.

    Returns:
        BaselineResult with Decision.ASK if t0 is ambiguous, otherwise
        Decision.CONTINUE.
    """
    image_arr, image_shape = _load_image(image_path)
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

    if _ambiguity_from_step(step1):
        question = str(step1.get("clarifying_question", ""))
        return BaselineResult(
            method=METHOD_NAME,
            decision=Decision.ASK,
            reason="Initial scene is ambiguous",
            question=question,
            raw_output=step1,
            metadata={
                "image_shape": image_shape,
                "session_id": session_id,
                "uses_checkpoint": False,
                "stores_g0": False,
            },
        )

    return BaselineResult(
        method=METHOD_NAME,
        decision=Decision.CONTINUE,
        reason="Initial scene is unambiguous; B1 proceeds without checkpoints",
        raw_output=step1,
        metadata={
            "image_shape": image_shape,
            "session_id": session_id,
            "uses_checkpoint": False,
            "stores_g0": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B1 Initial-only baseline.")
    parser.add_argument("image_path")
    parser.add_argument("task_description")
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--session-id", default="b1_t0")
    args = parser.parse_args()

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    result = run_b1_initial_only(
        args.image_path,
        args.task_description,
        model_type=args.model_type,
        adapter_ckpt=args.adapter_ckpt,
        session_id=args.session_id,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
