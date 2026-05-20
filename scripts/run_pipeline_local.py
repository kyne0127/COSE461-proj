"""
Pipeline 인터랙티브 테스트 (gRPC 없이)
======================================

실제 이미지 3장(t₀ / C1 / C2)과 task description을 입력하면
t₀→C1→C2 전체 파이프라인을 실행하고 결과를 출력합니다.
ASK 결정이 나오면 사용자에게 직접 답변을 받아 rolling G₀ update를 수행합니다.

모드:
  --mock   실제 Molmo 로드 없이 mock 핸들러로 동작 확인 (빠름)
  (기본)   실제 Molmo finetune 모델 사용 (GPU 필요)

예시:
  # mock 모드
  python scripts/run_pipeline_local.py --mock

  # 실제 모델, 동일 이미지 3장
  python scripts/run_pipeline_local.py \\
    --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm \\
    --image /workspace/AmbRes/assets/images/5rhU25AdQW4jADxhp8EYuq.jpeg \\
    --task "move the marker next to the sprite bottle"

  # 실제 모델, 이미지 3장 각각 지정
  python scripts/run_pipeline_local.py \\
    --model-type finetune --adapter-ckpt 43qazb3XcrZF5rZWnjRPVm \\
    --image-t0 t0.png --image-c1 c1.png --image-c2 c2.png \\
    --task "place the red block in the tray"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

# ── 경로 세팅 ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
SRC_ROOT    = REPO_ROOT / "src"
AMBRES_ROOT = REPO_ROOT.parent / "AmbRes"
LOGS_DIR    = REPO_ROOT / "logs"
for _p in (SRC_ROOT, REPO_ROOT, AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── tensorflow stub (Molmo processor 조건부 import 대응) ──────────────────────
def _stub_tensorflow() -> None:
    if "tensorflow" in sys.modules:
        return
    tf = ModuleType("tensorflow")
    tf.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)
    tf.Tensor   = type("Tensor",   (), {})
    tf.Variable = type("Variable", (), {})
    sys.modules["tensorflow"] = tf

_stub_tensorflow()

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = LOGS_DIR / f"run_pipeline_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_pipeline_local")


# ══════════════════════════════════════════════════════════════════════════════
# Mock 핸들러 (--mock 모드)
# ══════════════════════════════════════════════════════════════════════════════

class _MockHandler:
    """실제 Molmo 없이 파이프라인 흐름을 검증하는 경량 핸들러.

    시나리오:
      clear        : t₀/C1/C2 모두 CLEAR
      ambiguous-c1 : C1에서 target이 2개 감지 → AMBIGUOUS_TARGET
      missing-c1   : C1에서 target 소실       → INVALID_TARGET (STOP)
      ambiguous-c2 : C2에서 dest가 2개 감지  → AMBIGUOUS_DESTINATION
    """

    _OBJECTS   = ["mock target", "mock destination"]
    _BASE_COORD = {"mock target": [200, 300], "mock destination": [400, 300]}

    def __init__(self, scenario: str = "clear"):
        self.scenario  = scenario
        self._sessions: dict[str, list] = {}   # session_id → call history

    def handle(self, method: str, payload: dict, tensors: list, session_id: str) -> dict:
        history = self._sessions.setdefault(session_id, [])
        history.append(method)
        n = len(history)
        logger.debug("[mock] session=%s call=%d method=%s", session_id, n, method)

        if method == "reset":
            self._sessions[session_id] = []
            return {"success": True}

        if method == "query":
            return {
                "success": True,
                "task_ambiguous": False,
                "task_objects": self._OBJECTS,
                "clarifying_question": "",
            }

        if method == "respond":
            answer = payload.get("response", "")
            obj = [answer if answer else self._OBJECTS[0], self._OBJECTS[1]]
            return {"success": True, "task_objects": obj}

        if method == "detect":
            objects = payload.get("objects", self._OBJECTS)
            return {"success": True, "detections": self._detections(objects, session_id)}

        return {"success": False, "error": f"unknown method: {method}"}

    def _detections(self, objects: list[str], session_id: str) -> dict:
        det: dict[str, list] = {}
        for obj in objects:
            base = self._BASE_COORD.get(obj, [300, 300])
            # C1/C2 세션 여부로 시나리오 분기
            is_checkpoint = any(k in session_id for k in ("_c1", "_c2"))
            if is_checkpoint and self.scenario == "ambiguous-c1" and "target" in obj.lower():
                det[obj] = [base, [base[0] + 150, base[1] - 100]]   # 2개 → AMBIGUOUS
            elif is_checkpoint and self.scenario == "missing-c1" and "target" in obj.lower():
                det[obj] = []                                          # 0개 → INVALID
            elif is_checkpoint and self.scenario == "ambiguous-c2" and "destination" in obj.lower():
                det[obj] = [base, [base[0] + 150, base[1] + 80]]     # 2개 → AMBIGUOUS
            else:
                det[obj] = [base]                                      # 정상 단일 감지
        return det


def _patch_mock(scenario: str) -> _MockHandler:
    """ambres.ambres_model 패치 + MockHandler 반환."""
    import types
    mock_mod = types.ModuleType("ambres.ambres_model")
    mock_mod.AmbresFSPrompt  = lambda **kw: None
    mock_mod.AmbresFineTuned = lambda **kw: None
    sys.modules["ambres.ambres_model"] = mock_mod
    logger.info("[mock] 패치 완료 (scenario=%s)", scenario)
    return _MockHandler(scenario)


# ══════════════════════════════════════════════════════════════════════════════
# 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

_STATE_ICON = {
    "CLEAR":                    "✓",
    "AMBIGUOUS_TARGET":         "⚠",
    "INVALID_TARGET":           "✗",
    "AMBIGUOUS_DESTINATION":    "⚠",
    "INVALID_DESTINATION":      "⚠",
    "UNSAFE_OR_BLOCKED":        "✗",
}
_DECISION_COLOR = {"CONTINUE": "✓", "ASK": "?", "STOP": "✗"}


def _banner(text: str) -> None:
    print(f"\n{'─'*56}")
    print(f"  {text}")
    print(f"{'─'*56}")


def _print_g0(g0: dict) -> None:
    t = g0["target"]
    d = g0["destination"]
    h, w = g0["image_shape"]
    print(f"  target     : {t['label']}  @ {t['coord']}")
    print(f"  destination: {d['label']}  @ {d['coord']}")
    print(f"  image_shape: [{h}, {w}]")


def _print_checkpoint(label: str, outcome: Any) -> None:
    icon = _STATE_ICON.get(outcome.state.value, "?")
    dec  = _DECISION_COLOR.get(outcome.decision.value, "?")
    print(f"\n[{label}] state={outcome.state.value}  decision={dec} {outcome.decision.value}")
    if outcome.user_response:
        print(f"  사용자 답변   : {outcome.user_response!r}")
        print(f"  G₀ 업데이트  : target={outcome.g0_after['target']['label']}"
              f" @ {outcome.g0_after['target']['coord']}")


# ══════════════════════════════════════════════════════════════════════════════
# 인터랙티브 user_response_fn
# ══════════════════════════════════════════════════════════════════════════════

def _safe_input(prompt: str) -> str:
    """input()을 호출하되 EOF(비대화형 환경)는 빈 문자열로 처리."""
    try:
        return input(prompt).strip()
    except EOFError:
        print("  [비대화형 환경: 빈 응답으로 처리]")
        return ""


def _make_interactive_response_fn(mock: bool):
    """ASK 결정 시 터미널에서 사용자 답변을 받는 콜백 함수."""
    def fn(question: str, g0: dict) -> str:
        print(f"\n{'━'*56}")
        print(f"  [AmbRes] {question}")
        print(f"  현재 G₀ → target: {g0['target']['label']} / dest: {g0['destination']['label']}")
        if mock:
            print("  [mock] 빈 Enter 시 rolling update 없이 진행합니다.")
        ans = _safe_input("  [사용자 답변] (Enter = 건너뜀): ")
        print(f"{'━'*56}")
        logger.info("[user_response] question=%r  answer=%r", question, ans)
        return ans
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 경로 입력 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _prompt_image(label: str, default: str | None = None) -> str:
    hint = f" (Enter = {default!r})" if default else ""
    while True:
        val = _safe_input(f"[입력] {label} 이미지 경로{hint}: ")
        if not val and default:
            return default
        if val and Path(val).exists():
            return val
        if val:
            print(f"  ✗ 파일을 찾을 수 없습니다: {val}")
        else:
            print("  ✗ 경로를 입력하거나 Enter로 기본값을 사용하세요.")


def _prompt_task() -> str:
    while True:
        val = _safe_input("[입력] Task description: ")
        if val:
            return val
        print("  ✗ task description을 입력하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_once(
    handler,
    image_t0: str,
    image_c1: str,
    image_c2: str,
    task: str,
    threshold: float,
    mock: bool,
) -> None:
    from pipeline import run_pipeline

    logger.info("파이프라인 시작: task=%r", task)
    logger.info("  t₀=%s  C1=%s  C2=%s", image_t0, image_c1, image_c2)

    response_fn = _make_interactive_response_fn(mock)

    print("\n[t₀] G₀ 추출 중...")
    result = run_pipeline(
        image_t0, image_c1, image_c2,
        task,
        handler=handler,
        threshold=threshold,
        user_response_fn=response_fn,
        allow_ambiguous=True,
        session_prefix="local",
    )
    logger.info("파이프라인 완료: status=%s", result.status)

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    _banner("결과")

    if result.status == "initial_ambiguous":
        print("✗ t₀ ambiguity=true — 초기 장면이 모호합니다.")
        print(f"  {result.stop_reason}")
        return

    print("\n[G₀]")
    _print_g0(result.g0_initial)

    if result.c1:
        _print_checkpoint("C1", result.c1)
    if result.c2:
        _print_checkpoint("C2", result.c2)

    print()
    if result.status == "complete":
        print("✓ Pipeline complete — PLACE 진행 가능")
        if result.c2:
            final = result.c2.g0_after
            print(f"  최종 G₀: target={final['target']['label']} @ {final['target']['coord']}"
                  f"  /  destination={final['destination']['label']} @ {final['destination']['coord']}")
    else:
        print(f"✗ Pipeline stopped at {result.stop_reason}")

    # JSON 전체 출력 (로그 파일에만)
    logger.info("pipeline result:\n%s", json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def interactive_loop(
    handler,
    fixed_t0: str | None,
    fixed_c1: str | None,
    fixed_c2: str | None,
    fixed_task: str | None,
    threshold: float,
    mock: bool,
) -> None:
    _banner("Pipeline 인터랙티브 테스트  (종료: q)")
    print(f"  Log → {_log_file}")
    print(f"  Threshold = {threshold} px")

    while True:
        print()

        # ── 이미지 경로: 고정값은 출력, 미설정이면 프롬프트 ─────────────────
        if fixed_t0:
            t0 = fixed_t0
            print(f"[t₀]          {t0}")
        else:
            t0 = _prompt_image("t₀")
            if t0.lower() == "q":
                break

        if fixed_c1:
            c1 = fixed_c1
            print(f"[C1 pre-pick] {c1}")
        else:
            c1 = _prompt_image("C1 (pre-pick)", default=t0)
            if c1.lower() == "q":
                break

        if fixed_c2:
            c2 = fixed_c2
            print(f"[C2 pre-place]{c2}")
        else:
            c2 = _prompt_image("C2 (pre-place)", default=t0)
            if c2.lower() == "q":
                break

        # ── task description: 고정값은 출력, 미설정이면 프롬프트 ─────────────
        if fixed_task:
            task = fixed_task
            print(f"[task]        {task}")
        else:
            task = _prompt_task()
            if task.lower() == "q":
                break

        # ── 파이프라인 실행 ──────────────────────────────────────────────────
        try:
            run_once(handler, t0, c1, c2, task, threshold, mock)
        except Exception as exc:
            logger.exception("파이프라인 실행 중 오류: %s", exc)
            print(f"\n✗ 오류 발생: {exc}")

        # ── 반복 여부 ────────────────────────────────────────────────────────
        again = _safe_input("\n[입력] 다시 실행할까요? (y/n): ").lower()
        if again != "y":
            break


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline 인터랙티브 테스트 (gRPC 없음)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 모델
    parser.add_argument("--mock", action="store_true",
                        help="실제 Molmo 없이 mock 핸들러 사용")
    parser.add_argument("--mock-scenario",
                        choices=["clear", "ambiguous-c1", "missing-c1", "ambiguous-c2"],
                        default="clear",
                        help="mock 시나리오 (기본: clear)")
    parser.add_argument("--model-type", default="fs_prompt",
                        choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="",
                        help="finetune 체크포인트 ID (예: 43qazb3XcrZF5rZWnjRPVm)")
    # 이미지 (--image는 t0/C1/C2 동일 이미지)
    parser.add_argument("--image", default=None,
                        help="t₀/C1/C2에 동일 이미지 사용")
    parser.add_argument("--image-t0", default=None)
    parser.add_argument("--image-c1", default=None)
    parser.add_argument("--image-c2", default=None)
    # task
    parser.add_argument("--task", default=None,
                        help="task description (미입력 시 인터랙티브 입력)")
    # 파이프라인 옵션
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="consistency monitor 거리 임계값 (px, 기본: 50)")
    args = parser.parse_args()

    # ── 이미지 경로 정규화 ────────────────────────────────────────────────────
    t0 = args.image_t0 or args.image
    c1 = args.image_c1 or args.image
    c2 = args.image_c2 or args.image

    # ── 핸들러 초기화 ─────────────────────────────────────────────────────────
    if args.mock:
        handler = _patch_mock(args.mock_scenario)
        logger.info("[mock 모드] scenario=%s", args.mock_scenario)
    else:
        if args.model_type == "finetune" and not args.adapter_ckpt:
            parser.error("--model-type finetune 시 --adapter-ckpt 필요")
        from module.models.ambres.handler import AmbResHandler
        cfg = {
            "model_type":    args.model_type,
            "adapter_ckpt":  args.adapter_ckpt,
            "use_detection": False,
            "use_sam":       False,
        }
        logger.info("핸들러 초기화 중... %s", cfg)
        handler = AmbResHandler()
        handler.setup(cfg)
        logger.info("핸들러 초기화 완료")

    try:
        interactive_loop(
            handler,
            fixed_t0=t0,
            fixed_c1=c1,
            fixed_c2=c2,
            fixed_task=args.task,
            threshold=args.threshold,
            mock=args.mock,
        )
    except KeyboardInterrupt:
        pass

    print(f"\n종료.  Log → {_log_file}")


if __name__ == "__main__":
    main()
