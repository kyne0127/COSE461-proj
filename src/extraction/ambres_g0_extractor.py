from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from extraction.role_parser import parse_roles


_SRC_ROOT   = Path(__file__).resolve().parent.parent  # COSE461-proj/src
REPO_ROOT   = _SRC_ROOT.parent                        # COSE461-proj
AMBRES_ROOT = REPO_ROOT.parent / "AmbRes"             # workspace/AmbRes
for path in (_SRC_ROOT, REPO_ROOT, AMBRES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_image(path: str | Path) -> tuple[np.ndarray, list[int]]:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    return arr, [h, w]


def _unwrap_handler_result(result: dict[str, Any], method: str) -> dict[str, Any]:
    if not result.get("success"):
        raise RuntimeError(f"AmbRes {method} failed: {result.get('error')}")
    return {k: v for k, v in result.items() if k != "success"}


def _object_list_from_step(step_output: dict[str, Any]) -> list[str]:
    objects = step_output.get("object_list")
    if objects is None:
        objects = step_output.get("task_objects")
    if not isinstance(objects, list) or not all(isinstance(obj, str) for obj in objects):
        raise ValueError(f"AmbRes output does not contain a valid object list: {step_output}")
    return objects


def _ambiguity_from_step(step_output: dict[str, Any]) -> bool:
    if "ambiguity" in step_output:
        return bool(step_output["ambiguity"])
    if "task_ambiguous" in step_output:
        return bool(step_output["task_ambiguous"])
    raise ValueError(f"AmbRes output does not contain ambiguity flag: {step_output}")


def _first_coord(detections: dict[str, Any], label: str) -> list[int] | None:
    raw = detections.get(label) or []
    coords = [c for c in raw if isinstance(c, (list, tuple)) and len(c) == 2]
    if not coords:
        return None
    first = coords[0]
    return [int(first[0]), int(first[1])]


def _all_valid_coords(detections: dict[str, Any], label: str) -> list[list[int]]:
    raw = detections.get(label) or []
    return [[int(c[0]), int(c[1])] for c in raw
            if isinstance(c, (list, tuple)) and len(c) == 2]


def _make_handler(model_type: str, adapter_ckpt: str, use_detection: bool,
                  use_dino: bool = False,
                  dino_box_threshold: float = 0.35,
                  dino_text_threshold: float = 0.25):
    from module.models.ambres.handler import AmbResHandler

    handler = AmbResHandler()
    handler.setup(
        {
            "model_type": model_type,
            "adapter_ckpt": adapter_ckpt,
            "use_detection": use_detection,
            "use_dino": use_dino,
            "use_sam": False,
            "dino_box_threshold": dino_box_threshold,
            "dino_text_threshold": dino_text_threshold,
        }
    )
    return handler


def extract_g0(
    image_path: str | Path,
    task_description: str,
    *,
    handler: Any = None,
    model_type: str = "fs_prompt",
    adapter_ckpt: str = "",
    session_id: str = "g0_t0",
    allow_ambiguous: bool = False,
) -> dict[str, Any]:
    """Run AmbRes Step 1, forced Step 2, and coordinate detection to build G0.

    Args:
        handler: Pre-built AmbResHandler. If None, one is created from
                 model_type / adapter_ckpt. Pass an existing handler to
                 avoid reloading model weights (e.g. in integration tests
                 or when calling extract_g0 multiple times in a pipeline).
    """
    image_arr, image_shape = _load_image(image_path)
    tensors = [image_arr]
    if handler is None:
        handler = _make_handler(model_type, adapter_ckpt, use_detection=False)

    _unwrap_handler_result(handler.handle("reset", {}, [], session_id), "reset")

    step1 = _unwrap_handler_result(
        handler.handle(
            "query",
            {"task_description": task_description},
            tensors,
            session_id,
        ),
        "query",
    )
    if _ambiguity_from_step(step1) and not allow_ambiguous:
        question = step1.get("clarifying_question", "")
        raise RuntimeError(
            "Step 1 returned ambiguity=true at t0. "
            f"clarifying_question={question!r}"
        )

    step2 = _unwrap_handler_result(
        handler.handle("respond", {"response": ""}, [], session_id),
        "respond",
    )
    object_list = _object_list_from_step(step2) or _object_list_from_step(step1)
    roles = parse_roles(object_list, task_description)

    labels = [roles["target"], roles["destination"]]
    detections = _unwrap_handler_result(
        handler.handle("detect", {"objects": labels}, tensors, session_id),
        "detect",
    )["detections"]

    # t₀ 시각적 모호성 검사: detect 결과가 2개 이상이면 최초부터 ASK가 필요
    # (Step 1의 task 모호성과 별개로, 장면 자체의 visual ambiguity)
    if not allow_ambiguous:
        for role, label in [("target", roles["target"]),
                             ("destination", roles["destination"])]:
            valid_coords = [
                c for c in (detections.get(label) or [])
                if isinstance(c, (list, tuple)) and len(c) == 2
            ]
            if len(valid_coords) > 1:
                raise RuntimeError(
                    f"ambiguity=true at t0. "
                    f"{role} '{label}'이(가) 장면에 {len(valid_coords)}개 감지됨. "
                    f"coords={valid_coords}"
                )

    return {
        "target": {
            "label": roles["target"],
            "coord": _first_coord(detections, roles["target"]),
            "coords": _all_valid_coords(detections, roles["target"]),
        },
        "destination": {
            "label": roles["destination"],
            "coord": _first_coord(detections, roles["destination"]),
            "coords": _all_valid_coords(detections, roles["destination"]),
        },
        "image_shape": image_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract G0 with AmbResVLM at t0.")
    parser.add_argument("image_path")
    parser.add_argument("task_description")
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--session-id", default="g0_t0")
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Continue even if Step 1 reports ambiguity=true.",
    )
    args = parser.parse_args()

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    g0 = extract_g0(
        args.image_path,
        args.task_description,
        model_type=args.model_type,
        adapter_ckpt=args.adapter_ckpt,
        session_id=args.session_id,
        allow_ambiguous=args.allow_ambiguous,
    )
    print(json.dumps(g0, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
