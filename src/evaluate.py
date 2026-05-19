"""Image-based evaluation runner for B1~B5 and Ours.

Dataset manifest formats:
  JSONL: one sample object per line
  JSON:  either a list of sample objects or {"samples": [...]}.

Required sample fields:
  id, scenario, task, initial_img, c1_img, c2_img, checkpoint,
  gold_state, gold_decision, target_label, destination_label

The evaluator is intentionally image-based. It does not execute robot actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def run_method(
    sample: EvalSample,
    method: str,
    *,
    handler: Any = None,
    threshold: float = 50.0,
    llm_client: Callable[..., Any] | None = None,
    llm_model: str = "gpt-4o",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B1~B5+Ours on an image manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--model-type", default="fs_prompt", choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--metrics-json", default="")
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

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    methods = [m.lower() for m in args.methods]
    needs_handler = any(m in {"b1", "b2", "b3", "b4", "ours"} for m in methods)
    handler = (
        _make_handler(args.model_type, args.adapter_ckpt, use_detection=False)
        if needs_handler else None
    )

    predictions: list[EvalPrediction] = []
    for sample in samples:
        for method in methods:
            predictions.append(
                run_method(
                    sample,
                    method,
                    handler=handler,
                    threshold=args.threshold,
                    llm_model=args.llm_model,
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
