"""Image-based evaluation runner for B1~B5 and Ours.

Dataset manifest formats:
  JSONL: one sample object per line
  JSON:  either a list of sample objects or {"samples": [...]}.

Required sample fields:
  id, scenario, task, initial_img, c1_img, c2_img, checkpoint,
  gold_state, gold_decision, target_label, destination_label

Image source modes (--source):
  manifest  (default) : load images from local paths defined in manifest
  zed                 : capture images interactively from ZED Mini camera
  usb                 : capture images interactively from USB camera

The evaluator is intentionally image-based. It does not execute robot actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.b1_initial_only import run_b1_initial_only
from baselines.b2_no_memory import run_b2_no_memory
from baselines.b3_count_rule import run_b3_count_rule
from baselines.b4_binary_anomaly import run_b4_binary_anomaly
from baselines.b5_llm_judge import run_b5_llm_judge
from baselines.common import BaselineResult
from extraction.ambres_g0_extractor import _make_handler, extract_g0
from monitoring.consistency_monitor import Decision
from pipeline import PipelineResult, run_pipeline


DEFAULT_METHODS = ["b1", "b2", "b3", "b4", "b5", "ours"]
DECISION_VALUES = {d.value for d in Decision}


@dataclass
class EvalSample:
    sample_id: str
    scenario: str
    task: str
    initial_img: str
    c1_img: str
    c2_img: str
    checkpoint: str
    gold_state: str
    gold_decision: Decision
    target_label: str
    destination_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_img(self) -> str:
        if self.checkpoint == "C1":
            return self.c1_img
        if self.checkpoint == "C2":
            return self.c2_img
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {self.checkpoint!r}")


@dataclass
class EvalPrediction:
    sample_id: str
    scenario: str
    method: str
    checkpoint: str
    gold_state: str
    gold_decision: Decision
    predicted_decision: Decision
    predicted_state: str = ""
    reason: str = ""
    question: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scenario": self.scenario,
            "method": self.method,
            "checkpoint": self.checkpoint,
            "gold_state": self.gold_state,
            "gold_decision": self.gold_decision.value,
            "predicted_decision": self.predicted_decision.value,
            "predicted_state": self.predicted_state,
            "reason": self.reason,
            "question": self.question,
            "correct": self.predicted_decision == self.gold_decision,
            "raw": self.raw,
        }


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _sample_from_dict(raw: dict[str, Any], base_dir: Path) -> EvalSample:
    required = [
        "id",
        "scenario",
        "task",
        "initial_img",
        "c1_img",
        "c2_img",
        "checkpoint",
        "gold_state",
        "gold_decision",
        "target_label",
        "destination_label",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Sample {raw.get('id', '<unknown>')} missing fields: {missing}")

    checkpoint = str(raw["checkpoint"]).upper()
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"Sample {raw['id']} has invalid checkpoint {checkpoint!r}")

    gold_decision_raw = str(raw["gold_decision"]).upper()
    if gold_decision_raw not in DECISION_VALUES:
        raise ValueError(f"Sample {raw['id']} has invalid gold_decision {gold_decision_raw!r}")

    metadata = {
        k: v for k, v in raw.items()
        if k not in required
    }
    return EvalSample(
        sample_id=str(raw["id"]),
        scenario=str(raw["scenario"]),
        task=str(raw["task"]),
        initial_img=_resolve_path(str(raw["initial_img"]), base_dir),
        c1_img=_resolve_path(str(raw["c1_img"]), base_dir),
        c2_img=_resolve_path(str(raw["c2_img"]), base_dir),
        checkpoint=checkpoint,
        gold_state=str(raw["gold_state"]).upper(),
        gold_decision=Decision(gold_decision_raw),
        target_label=str(raw["target_label"]),
        destination_label=str(raw["destination_label"]),
        metadata=metadata,
    )


def load_manifest(path: str | Path) -> list[EvalSample]:
    path = Path(path)
    base_dir = path.parent
    if path.suffix == ".jsonl":
        raws = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        data = json.loads(path.read_text())
        raws = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if not isinstance(raws, list):
        raise ValueError("Manifest must be a JSON list, JSON {'samples': [...]}, or JSONL")
    return [_sample_from_dict(raw, base_dir) for raw in raws]


def manifest_summary(samples: list[EvalSample]) -> dict[str, Any]:
    """Return a lightweight summary that does not require loading models/images."""
    by_scenario: dict[str, int] = {}
    by_checkpoint: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    for sample in samples:
        by_scenario[sample.scenario] = by_scenario.get(sample.scenario, 0) + 1
        by_checkpoint[sample.checkpoint] = by_checkpoint.get(sample.checkpoint, 0) + 1
        key = sample.gold_decision.value
        by_decision[key] = by_decision.get(key, 0) + 1
    return {
        "n_samples": len(samples),
        "scenarios": by_scenario,
        "checkpoints": by_checkpoint,
        "gold_decisions": by_decision,
    }


def image_path_summary(samples: list[EvalSample]) -> dict[str, Any]:
    """Return existence checks for all unique image paths referenced by samples."""
    paths = sorted({
        sample.initial_img
        for sample in samples
    } | {
        sample.c1_img
        for sample in samples
    } | {
        sample.c2_img
        for sample in samples
    })
    missing = [path for path in paths if not Path(path).is_file()]
    return {
        "n_unique_images": len(paths),
        "n_existing_images": len(paths) - len(missing),
        "n_missing_images": len(missing),
        "missing_images": missing,
    }


def _prediction_from_baseline(sample: EvalSample, result: BaselineResult) -> EvalPrediction:
    return EvalPrediction(
        sample_id=sample.sample_id,
        scenario=sample.scenario,
        method=result.method,
        checkpoint=sample.checkpoint,
        gold_state=sample.gold_state,
        gold_decision=sample.gold_decision,
        predicted_decision=result.decision,
        reason=result.reason,
        question=result.question,
        raw=result.to_dict(),
    )


def _prediction_from_ours(sample: EvalSample, result: PipelineResult) -> EvalPrediction:
    checkpoint_outcome = result.c1 if sample.checkpoint == "C1" else result.c2
    if result.status == "initial_ambiguous":
        decision = Decision.ASK
        state = "INITIAL_AMBIGUOUS"
        reason = result.stop_reason
    elif checkpoint_outcome is None:
        decision = Decision.STOP
        state = "NOT_REACHED"
        reason = result.stop_reason or f"{sample.checkpoint} was not reached"
    else:
        decision = checkpoint_outcome.decision
        state = checkpoint_outcome.state.value
        reason = result.stop_reason

    return EvalPrediction(
        sample_id=sample.sample_id,
        scenario=sample.scenario,
        method="OURS",
        checkpoint=sample.checkpoint,
        gold_state=sample.gold_state,
        gold_decision=sample.gold_decision,
        predicted_decision=decision,
        predicted_state=state,
        reason=reason,
        raw=result.to_dict(),
    )


def _make_interactive_response_fn() -> Callable[[str, dict], str]:
    """ASK 결정 시 터미널에서 사용자 답변을 받는 콜백."""
    def fn(question: str, g0: dict) -> str:
        print(f"\n{'━'*60}")
        print(f"  [ASK] {question}")
        print(f"  현재 G₀ → target: {g0['target']['label']} / dest: {g0['destination']['label']}")
        try:
            ans = input("  답변 (Enter = 건너뜀): ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        print(f"{'━'*60}")
        return ans
    return fn


def run_method(
    sample: EvalSample,
    method: str,
    *,
    handler: Any = None,
    threshold: float = 50.0,
    llm_client: Callable[..., Any] | None = None,
    llm_model: str = "gpt-4o",
    user_response_fn: Callable[[str, dict], str] | None = None,
) -> EvalPrediction:
    method_key = method.lower()
    if method_key == "b1":
        result = run_b1_initial_only(
            sample.initial_img,
            sample.task,
            handler=handler,
            session_id=f"{sample.sample_id}_b1",
        )
        return _prediction_from_baseline(sample, result)
    if method_key == "b2":
        result = run_b2_no_memory(
            sample.checkpoint_img,
            sample.task,
            checkpoint=sample.checkpoint,
            handler=handler,
            session_id=f"{sample.sample_id}_b2_{sample.checkpoint.lower()}",
        )
        return _prediction_from_baseline(sample, result)
    if method_key == "b3":
        result = run_b3_count_rule(
            sample.checkpoint_img,
            sample.task,
            sample.target_label,
            sample.destination_label,
            checkpoint=sample.checkpoint,
            handler=handler,
            session_id=f"{sample.sample_id}_b3_{sample.checkpoint.lower()}",
        )
        return _prediction_from_baseline(sample, result)
    if method_key == "b4":
        if handler is None:
            raise ValueError("B4 requires handler because it calls checkpoint detect")
        g0 = extract_g0(
            sample.initial_img,
            sample.task,
            handler=handler,
            session_id=f"{sample.sample_id}_b4_g0",
            allow_ambiguous=True,
        )
        result = run_b4_binary_anomaly(
            g0,
            sample.checkpoint_img,
            checkpoint=sample.checkpoint,
            handler=handler,
            threshold=threshold,
            session_id=f"{sample.sample_id}_b4_{sample.checkpoint.lower()}",
        )
        return _prediction_from_baseline(sample, result)
    if method_key == "b5":
        result = run_b5_llm_judge(
            sample.initial_img,
            sample.checkpoint_img,
            sample.task,
            checkpoint=sample.checkpoint,
            llm_client=llm_client,
            model=llm_model,
        )
        return _prediction_from_baseline(sample, result)
    if method_key == "ours":
        if handler is None:
            raise ValueError("Ours requires handler")
        result = run_pipeline(
            sample.initial_img,
            sample.c1_img,
            sample.c2_img,
            sample.task,
            handler=handler,
            threshold=threshold,
            user_response_fn=user_response_fn,
            session_prefix=f"{sample.sample_id}_ours",
            allow_ambiguous=True,
        )
        return _prediction_from_ours(sample, result)
    raise ValueError(f"Unknown method {method!r}")


def compute_metrics(predictions: Iterable[EvalPrediction]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[EvalPrediction]] = {}
    for pred in predictions:
        grouped.setdefault(pred.method, []).append(pred)

    out: dict[str, dict[str, float | int]] = {}
    for method, rows in grouped.items():
        total = len(rows)
        correct = sum(1 for r in rows if r.predicted_decision == r.gold_decision)
        should_not_continue = [r for r in rows if r.gold_decision != Decision.CONTINUE]
        should_continue = [r for r in rows if r.gold_decision == Decision.CONTINUE]
        false_continue = sum(
            1 for r in should_not_continue
            if r.predicted_decision == Decision.CONTINUE
        )
        false_alarm = sum(
            1 for r in should_continue
            if r.predicted_decision != Decision.CONTINUE
        )
        state_rows = [r for r in rows if r.predicted_state]
        state_correct = sum(
            1 for r in state_rows
            if r.predicted_state == r.gold_state
        )
        out[method] = {
            "n": total,
            "correct": correct,
            "decision_accuracy": correct / total if total else 0.0,
            "miss_rate": (
                false_continue / len(should_not_continue)
                if should_not_continue else 0.0
            ),
            "false_alarm_rate": (
                false_alarm / len(should_continue)
                if should_continue else 0.0
            ),
            "state_accuracy": (
                state_correct / len(state_rows)
                if state_rows else 0.0
            ),
            "state_n": len(state_rows),
        }
    return out


def write_predictions_csv(path: str | Path, predictions: list[EvalPrediction]) -> None:
    fieldnames = [
        "sample_id",
        "scenario",
        "method",
        "checkpoint",
        "gold_state",
        "gold_decision",
        "predicted_decision",
        "predicted_state",
        "reason",
        "question",
        "correct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            row = pred.to_dict()
            writer.writerow({k: row[k] for k in fieldnames})


def write_metrics_json(path: str | Path, metrics: dict[str, dict[str, float | int]]) -> None:
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


# ────────────────────────────────────────────────────────────────────────────
# 카메라 캡처 모드
# ────────────────────────────────────────────────────────────────────────────

def _capture_one_frame(source: str, camera, warmup: int = 3) -> "np.ndarray":
    """카메라에서 프레임 1장을 캡처해 uint8 RGB numpy 배열로 반환합니다."""
    import numpy as np
    # warmup: 노출 안정화를 위해 몇 프레임 버림
    for _ in range(warmup):
        if source == "zed":
            camera.capture_synchronized()
        else:
            camera.capture()
    if source == "zed":
        rgb, _ = camera.capture_synchronized()
    else:
        rgb = camera.capture()
    return rgb  # uint8 [H, W, 3]


def _save_capture(rgb: "np.ndarray", path: Path) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _prompt_capture(label: str, source: str, camera, save_path: Path, warmup: int) -> str:
    """씬 준비 안내 후 Enter를 받아 캡처하고 저장 경로를 반환합니다."""
    print(f"\n  [{label}] 씬을 준비하고 Enter를 누르면 캡처합니다 (q = 종료): ", end="", flush=True)
    try:
        ans = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "q"
    if ans == "q":
        raise SystemExit(0)
    print(f"  캡처 중...", end=" ", flush=True)
    rgb = _capture_one_frame(source, camera, warmup)
    _save_capture(rgb, save_path)
    h, w = rgb.shape[:2]
    print(f"완료 → {save_path}  ({w}×{h})")
    return str(save_path)


def run_camera_eval(
    source: str,
    methods: list[str],
    *,
    task: str,
    save_dir: Path,
    handler: Any,
    threshold: float,
    user_response_fn: Callable | None,
    llm_model: str,
    camera_index: int = 0,
    camera_resolution: str = "VGA",
    warmup: int = 3,
    predictions_csv: str = "",
    metrics_json: str = "",
) -> None:
    """
    ZED 또는 USB 카메라로 이미지를 캡처하며 평가를 진행합니다.

    흐름:
      1. 카메라 초기화
      2. t₀ 씬 준비 → Enter → 캡처 저장
      3. C1 씬 연출 → Enter → 캡처 저장
      4. C2 씬 연출 → Enter → 캡처 저장  (또는 's' 입력으로 건너뜀)
      5. 지정 methods 로 평가 실행
      6. 결과 출력 (gold label 입력 시 정확도 계산)
    """
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))

    # ── 카메라 초기화 ────────────────────────────────────────────────────────
    print(f"\n[camera] source={source}  ", end="")
    if source == "zed":
        from module.desktop.zed_connector import ZEDCapture
        camera = ZEDCapture(resolution=camera_resolution, fps=30)
        print(f"ZED ({camera_resolution}) 초기화 완료")
    else:
        from module.desktop.zed_connector import USBCapture
        camera = USBCapture(device_index=camera_index, width=640, height=480)
        print(f"USB index={camera_index} 초기화 완료")

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_dir = save_dir / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[camera] 저장 경로: {save_dir}")
    print(f"[camera] task: {task!r}")

    try:
        # ── 이미지 캡처 ──────────────────────────────────────────────────────
        t0_path  = _prompt_capture("t₀  (초기 장면)", source, camera, save_dir / "t0.png",  warmup)
        c1_path  = _prompt_capture("C1  (pre-pick  장면 변화)", source, camera, save_dir / "c1.png",  warmup)

        print(f"\n  [C2] pre-place 장면 캡처 (s = C1 이미지 재사용, Enter = 새로 캡처, q = 종료): ",
              end="", flush=True)
        try:
            c2_ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            c2_ans = "s"

        if c2_ans == "q":
            raise SystemExit(0)
        elif c2_ans == "s":
            c2_path = c1_path
            print(f"  C1 이미지 재사용: {c2_path}")
        else:
            c2_path = _prompt_capture("C2  (pre-place 장면 변화)", source, camera, save_dir / "c2.png", warmup)

        # ── C1 / C2 체크포인트 판단 ──────────────────────────────────────────
        print(f"\n  체크포인트 선택 (1=C1, 2=C2, 기본=C1): ", end="", flush=True)
        try:
            cp_ans = input().strip()
        except (EOFError, KeyboardInterrupt):
            cp_ans = "1"
        checkpoint = "C2" if cp_ans == "2" else "C1"

        # ── gold label 입력 (선택) ───────────────────────────────────────────
        print(f"\n  gold_state 입력 (정확도 계산용, Enter 건너뜀): ", end="", flush=True)
        try:
            gold_state_in = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            gold_state_in = ""

        print(f"  gold_decision 입력 (CONTINUE/ASK/STOP, Enter 건너뜀): ", end="", flush=True)
        try:
            gold_decision_in = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            gold_decision_in = ""

        valid_states    = {s.value for s in __import__("monitoring.consistency_monitor",
                           fromlist=["GroundingState"]).GroundingState}
        gold_state_val  = gold_state_in  if gold_state_in   in valid_states           else "CLEAR"
        gold_decision_val = gold_decision_in if gold_decision_in in {"CONTINUE","ASK","STOP"} else "CONTINUE"

        # ── EvalSample 생성 ──────────────────────────────────────────────────
        sample = EvalSample(
            sample_id=f"camera_{ts}",
            scenario="camera",
            task=task,
            initial_img=t0_path,
            c1_img=c1_path,
            c2_img=c2_path,
            checkpoint=checkpoint,
            gold_state=gold_state_val,
            gold_decision=Decision(gold_decision_val),
            target_label="",
            destination_label="",
        )

        # ── 평가 실행 ────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  평가 시작  methods={methods}  checkpoint={checkpoint}")
        print(f"{'='*60}")

        predictions: list[EvalPrediction] = []
        for method in methods:
            print(f"\n[{method.upper()}] 실행 중...")
            pred = run_method(
                sample, method,
                handler=handler,
                threshold=threshold,
                llm_model=llm_model,
                user_response_fn=user_response_fn,
            )
            predictions.append(pred)
            correct_mark = "✓" if pred.predicted_decision == pred.gold_decision else "✗"
            print(f"  결과: {pred.predicted_decision.value}  (gold={pred.gold_decision.value} {correct_mark})")
            if pred.predicted_state:
                print(f"  state: {pred.predicted_state}")
            if pred.reason:
                print(f"  reason: {pred.reason}")

        # ── metrics ──────────────────────────────────────────────────────────
        if gold_decision_in:
            metrics = compute_metrics(predictions)
            print(f"\n{'='*60}")
            print("  Metrics:")
            print(json.dumps(metrics, ensure_ascii=False, indent=4))
            if predictions_csv:
                write_predictions_csv(predictions_csv, predictions)
                print(f"  predictions → {predictions_csv}")
            if metrics_json:
                write_metrics_json(metrics_json, metrics)
                print(f"  metrics     → {metrics_json}")
        else:
            print("\n  (gold label 미입력 — 정확도 계산 건너뜀)")

        print(f"\n  캡처 이미지 저장 위치: {save_dir}")

    finally:
        if source == "zed":
            camera.release()
        else:
            camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B1~B5+Ours on an image manifest.")
    parser.add_argument("manifest", nargs="?", default="",
                        help="manifest 경로 (--source manifest 시 필수)")
    parser.add_argument("--source", choices=["manifest", "zed", "usb"], default="manifest",
                        help="이미지 소스: manifest(로컬 파일) | zed(ZED 카메라) | usb(USB 카메라)")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--openai-api-key", default="",
                        help="OpenAI API key for B5 (기본: OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="ASK 결정 시 터미널에서 실제 사용자 답변을 받아 rolling G₀ update 수행 (Ours 전용).",
    )
    # ── 카메라 모드 전용 옵션 ────────────────────────────────────────────────
    parser.add_argument("--task", default="",
                        help="[카메라 모드] task description")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="[USB 모드] 카메라 인덱스 (기본: 0)")
    parser.add_argument("--camera-resolution", choices=["VGA", "HD720"], default="VGA",
                        help="[ZED 모드] 해상도 (기본: VGA = 640×360)")
    parser.add_argument("--save-dir", default="/tmp/eval_captures",
                        help="[카메라 모드] 캡처 이미지 저장 디렉터리 (기본: /tmp/eval_captures)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="[카메라 모드] 캡처 전 버릴 워밍업 프레임 수 (기본: 3)")
    # ── manifest 전용 옵션 ───────────────────────────────────────────────────
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only load and validate manifest schema; do not load models or run methods.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also verify that all image paths referenced by the manifest exist.",
    )
    args = parser.parse_args()

    # OpenAI API key 설정 (--openai-api-key > 환경변수 순서)
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    methods = [m.lower() for m in args.methods]

    # ASK 시 사용자 입력 콜백 (--interactive 또는 카메라 모드 시 활성화)
    user_response_fn = (
        _make_interactive_response_fn()
        if (args.interactive or args.source in ("zed", "usb"))
        else None
    )

    # ── 카메라 모드 ────────────────────────────────────────────────────────────
    if args.source in ("zed", "usb"):
        if not args.task:
            parser.error("카메라 모드에서는 --task 가 필요합니다.")
        needs_handler = any(m in {"b1", "b2", "b3", "b4", "ours"} for m in methods)
        handler = (
            _make_handler(args.model_type, args.adapter_ckpt, use_detection=False)
            if needs_handler else None
        )
        run_camera_eval(
            source=args.source,
            methods=methods,
            task=args.task,
            save_dir=Path(args.save_dir),
            handler=handler,
            threshold=args.threshold,
            user_response_fn=user_response_fn,
            llm_model=args.llm_model,
            camera_index=args.camera_index,
            camera_resolution=args.camera_resolution,
            warmup=args.warmup,
            predictions_csv=args.predictions_csv,
            metrics_json=args.metrics_json,
        )
        return

    # ── manifest 모드 ──────────────────────────────────────────────────────────
    if not args.manifest:
        parser.error("manifest 경로가 필요합니다.")

    samples = load_manifest(args.manifest)

    if args.check_images:
        paths = image_path_summary(samples)
        if paths["n_missing_images"]:
            print(json.dumps({
                **manifest_summary(samples),
                "image_paths": paths,
            }, ensure_ascii=False, indent=2))
            raise SystemExit(1)

    if args.validate_only:
        summary = manifest_summary(samples)
        if args.check_images:
            summary["image_paths"] = image_path_summary(samples)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # validate-only 통과 후 핸들러 로드 (모델 로딩 지연)
    needs_handler = any(m in {"b1", "b2", "b3", "b4", "ours"} for m in methods)
    handler = (
        _make_handler(args.model_type, args.adapter_ckpt, use_detection=False)
        if needs_handler else None
    )

    predictions: list[EvalPrediction] = []
    for sample in samples:
        for method in methods:
            if args.interactive:
                print(f"\n[{sample.sample_id}] {method.upper()} 실행 중...")
            predictions.append(
                run_method(
                    sample,
                    method,
                    handler=handler,
                    threshold=args.threshold,
                    llm_model=args.llm_model,
                    user_response_fn=user_response_fn,
                )
            )

    metrics = compute_metrics(predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.predictions_csv:
        write_predictions_csv(args.predictions_csv, predictions)
    if args.metrics_json:
        write_metrics_json(args.metrics_json, metrics)


if __name__ == "__main__":
    main()
