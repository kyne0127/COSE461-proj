"""
AmbRes 핸들러 인터랙티브 테스트 (gRPC 없이)
=============================================

모드:
  --mock   실제 Molmo/SAM2 로드 없이 핸들러 라우팅·세션 관리만 검증 (빠름)
  (기본)   실제 Molmo 모델 로드 후 추론 (GPU 필요, 느림)

예시:
  python scripts/test_ambres_local.py --mock
  python scripts/test_ambres_local.py
  python scripts/test_ambres_local.py --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm
"""

from __future__ import annotations

import argparse
import sys
import os
import logging
import numpy as np
from pathlib import Path
from typing import Any, List

# ── 경로 세팅 ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Optional: make sure AmbRes package is discoverable when running locally.
AMBRES_ROOT = Path("/workspace/AmbRes")
if AMBRES_ROOT.exists():
    sys.path.insert(0, str(AMBRES_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("test_ambres_local")


# ══════════════════════════════════════════════════════════════════════════════
# Mock AmbRes 모델 (실제 Molmo 없이 핸들러 로직 검증용)
# ══════════════════════════════════════════════════════════════════════════════

class _MockAmbresModel:
    """AmbresFSPrompt / AmbresFineTuned 의 경량 대역."""

    def __init__(self, use_detection: bool = False):
        self.use_detection = use_detection
        self.messages: List[dict] = []
        self.images: List[Any] = []

    def reset_chat(self):
        self.messages = []

    def set_image(self, images, reset_chat=False):
        self.images = list(images)
        if reset_chat:
            self.reset_chat()

    def handle_query(self, task_description: str, image) -> dict:
        return {
            "task_objects": ["object_a", "object_b"],
            "obj_detection_messages": [],
            "task_ambiguous": True,
            "clarifying_question": "[MOCK] 어떤 object_b를 원하시나요?",
        }

    def handle_response(self, response: str) -> dict:
        return {
            "task_objects": ["object_a", f"specific {response}"],
            "obj_detection_messages": [],
        }

    def detect_pretty(self, objects: List[str]) -> dict:
        img = self.images[0] if self.images else None
        w, h = (img.size if img else (640, 480))
        return {obj: [[w // 2, h // 2]] for obj in objects}


def _patch_mock():
    import types
    mock_module = types.ModuleType("ambres.ambres_model")
    mock_module.AmbresFSPrompt  = _MockAmbresModel
    mock_module.AmbresFineTuned = _MockAmbresModel
    sys.modules["ambres.ambres_model"] = mock_module
    logger.info("[mock] ambres.ambres_model 패치 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _load_image(path: str, h=120, w=160) -> np.ndarray:
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB").resize((w, h))
    return np.array(img, dtype=np.float32) / 255.0


def _print_json(data: dict):
    import json
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# 인터랙티브 루프
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(handler, session_id: str):
    print("\n" + "═" * 50)
    print("  AmbRes 인터랙티브 테스트")
    print("  종료: Ctrl+C 또는 'q' 입력")
    print("═" * 50)

    while True:
        # ── 이미지 경로 입력 ──────────────────────────────
        print()
        image_path = input("[입력] 이미지 경로 ('q' 종료): ").strip()
        if image_path.lower() == "q":
            break
        if not os.path.exists(image_path):
            print(f"  ✗ 파일을 찾을 수 없습니다: {image_path}")
            continue

        image_arr = _load_image(image_path)
        tensors = [image_arr]
        print(f"  이미지 로드 완료: shape={image_arr.shape}")

        # ── task description 입력 ─────────────────────────
        task_description = input("[입력] Task description: ").strip()
        if task_description.lower() == "q":
            break
        if not task_description:
            print("  ✗ task description을 입력하세요.")
            continue

        # ── reset ─────────────────────────────────────────
        handler.handle("reset", {}, [], session_id)

        # ── query ─────────────────────────────────────────
        print("\n[AmbRes] 추론 중...")
        result = handler.handle("query", {"task_description": task_description}, tensors, session_id)

        if not result.get("success"):
            print(f"  ✗ query 실패: {result.get('error')}")
            continue

        task_objects        = result["task_objects"]
        task_ambiguous      = result["task_ambiguous"]
        clarifying_question = result.get("clarifying_question", "")

        print(f"\n  task_objects       : {task_objects}")
        print(f"  task_ambiguous     : {task_ambiguous}")
        if clarifying_question:
            print(f"  clarifying_question: {clarifying_question}")

        # ── respond (모호한 경우) ──────────────────────────
        if task_ambiguous:
            user_answer = input(f"\n[AmbRes] {clarifying_question}\n> ").strip()
            if user_answer.lower() == "q":
                break

            result2 = handler.handle("respond", {"response": user_answer}, [], session_id)
            if not result2.get("success"):
                print(f"  ✗ respond 실패: {result2.get('error')}")
                continue

            task_objects = result2["task_objects"]
            print(f"\n  최종 task_objects  : {task_objects}")

        # ── detect ────────────────────────────────────────
        run_detect = input("\n[입력] 객체 위치 검출할까요? (y/n): ").strip().lower()
        if run_detect == "y":
            result3 = handler.handle("detect", {"objects": task_objects}, tensors, session_id)
            if result3.get("success"):
                print(f"  detections: {result3['detections']}")
            else:
                print(f"  ✗ detect 실패: {result3.get('error')}")

        print("\n" + "─" * 50)


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AmbRes 핸들러 인터랙티브 테스트 (gRPC 없음)")
    parser.add_argument("--mock",          action="store_true",
                        help="실제 Molmo 없이 mock 모델 사용")
    parser.add_argument("--model-type",    default="fs_prompt",
                        choices=["fs_prompt", "finetune"],
                        help="AmbRes 모델 타입 (기본: fs_prompt)")
    parser.add_argument("--adapter-ckpt",  default="",
                        help="finetune 시 체크포인트 이름")
    parser.add_argument("--use-detection", action="store_true",
                        help="query/respond 단계에서 Molmo detect 자동 수행")
    parser.add_argument("--session-id",    default="interactive_session",
                        help="세션 ID (기본: interactive_session)")
    args = parser.parse_args()

    if args.mock:
        _patch_mock()
        logger.info("[mock 모드] Molmo/SAM2 로드를 건너뜁니다")

    from module.models.ambres.handler import AmbResHandler

    config = {
        "model_type":    args.model_type,
        "adapter_ckpt":  args.adapter_ckpt,
        "use_detection": args.use_detection,
        "use_sam":       False,
    }
    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune 사용 시 --adapter-ckpt 필요")

    logger.info("핸들러 초기화 중... config=%s", config)
    handler = AmbResHandler()
    handler.setup(config)
    logger.info("핸들러 초기화 완료")

    try:
        interactive_loop(handler, session_id=args.session_id)
    except KeyboardInterrupt:
        pass

    print("\n종료.")


if __name__ == "__main__":
    main()
