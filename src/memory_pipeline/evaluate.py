"""
src.memory_pipeline.evaluate
==============================
Self-contained evaluation runner for the memory ablation study.

Methods (all implemented in this folder, no dependency on src/checkpoint_monitors
or src/pipeline.py):

  static   GDino + AmbRes, no memory context (control condition)
  memory   GDino + AmbRes + episode memory context (proposed method)
  fsm      MemoryFSM: trigger-based, object-level delta detection via GDino
           Processes video frames (or checkpoint images) and fires AmbRes
           only when a meaningful object-state change is detected.

All methods use identical GDino + AmbRes stack.
Independent variables:
  static vs memory  — whether episode memory is injected into AmbRes
  fsm vs memory     — whether checkpoints are fixed (C1/C2 images) or
                      trigger-based (video frames, object-level delta)

Output format matches src/evaluate.py so results can be merged into a
single ablation table.

Usage:
  python src/memory_pipeline/evaluate.py dataset/hf_manifest.jsonl \\
    --methods static memory fsm \\
    --predictions-csv results/memory_preds.csv \\
    --metrics-json   results/memory_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
_AMBRES_ROOT = _REPO_ROOT.parent / "AmbRes"
for _p in (_SRC_ROOT, _REPO_ROOT, _AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from memory_pipeline.pipeline import MemoryPipelineResult, run_memory_pipeline
from memory_pipeline.runtime_fsm import (
    DecisionState,
    MemoryFSM,
    MemoryFSMResult,
    PhaseState,
    load_video_frames,
)
from memory_pipeline.static_pipeline import StaticPipelineResult, run_static_pipeline
from monitoring.consistency_monitor import Decision

logger = logging.getLogger(__name__)

DEFAULT_METHODS: list[str] = ["static", "memory", "memory_vlm", "memory_vlm_judge", "memory_qwen", "memory_qwen_gate", "memory_qwen_detect", "memory_finetuned", "memory_ensemble", "fsm"]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _resolve(value: str, base: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    base = path.parent
    raw_lines: list[dict]
    if path.suffix == ".jsonl":
        raw_lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_lines = data["samples"] if isinstance(data, dict) and "samples" in data else data

    for raw in raw_lines:
        for key in ("initial_img", "c1_img", "c2_img"):
            if key in raw:
                raw[key] = _resolve(str(raw[key]), base)
    return raw_lines


# ---------------------------------------------------------------------------
# Prediction dataclass (same schema as src/evaluate.py EvalPrediction)
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    sample_id:          str
    scenario:           str
    method:             str
    checkpoint:         str
    gold_state:         str
    gold_decision:      Decision
    predicted_decision: Decision
    predicted_state:    str = ""
    reason:             str = ""
    question:           str = ""
    explanation:        str = ""
    raw:                dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.predicted_decision == self.gold_decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id":          self.sample_id,
            "scenario":           self.scenario,
            "method":             self.method,
            "checkpoint":         self.checkpoint,
            "gold_state":         self.gold_state,
            "gold_decision":      self.gold_decision.value,
            "predicted_decision": self.predicted_decision.value,
            "predicted_state":    self.predicted_state,
            "reason":             self.reason,
            "question":           self.question,
            "explanation":        self.explanation,
            "correct":            self.correct,
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _prediction_from_static(
    sample: dict,
    result: StaticPipelineResult,
) -> Prediction:
    checkpoint = sample["checkpoint"].upper()
    co = result.c1 if checkpoint == "C1" else result.c2

    if result.status == "stopped" and co is None:
        decision = Decision.STOP
        state = "NOT_REACHED"
    elif co is None:
        decision = Decision.CONTINUE
        state = ""
    else:
        decision = co.decision
        state = co.state.value

    return Prediction(
        sample_id=str(sample["id"]),
        scenario=str(sample["scenario"]),
        method="STATIC",
        checkpoint=checkpoint,
        gold_state=str(sample["gold_state"]).upper(),
        gold_decision=Decision(str(sample["gold_decision"]).upper()),
        predicted_decision=decision,
        predicted_state=state,
        reason=result.stop_reason,
        question=co.question if co else "",
        raw=result.to_dict(),
    )


def _prediction_from_memory(
    sample: dict,
    result: MemoryPipelineResult,
) -> Prediction:
    checkpoint = sample["checkpoint"].upper()
    co = result.c1 if checkpoint == "C1" else result.c2

    if result.status == "stopped" and co is None:
        decision = Decision.STOP
        state = "NOT_REACHED"
        reason = result.stop_reason
    elif co is None:
        decision = Decision.CONTINUE
        state = ""
        reason = ""
    else:
        decision = co.decision
        state = co.state.value
        reason = result.stop_reason

    return Prediction(
        sample_id=str(sample["id"]),
        scenario=str(sample["scenario"]),
        method="MEMORY",
        checkpoint=checkpoint,
        gold_state=str(sample["gold_state"]).upper(),
        gold_decision=Decision(str(sample["gold_decision"]).upper()),
        predicted_decision=decision,
        predicted_state=state,
        reason=reason,
        question=co.question if co else "",
        explanation=co.explanation if co else "",
        raw=result.to_dict(),
    )


def _prediction_from_fsm(
    sample: dict,
    result: MemoryFSMResult,
) -> Prediction:
    checkpoint = sample["checkpoint"].upper()
    step = result.c1 if checkpoint == "C1" else result.c2

    if step is None:
        # FSM never triggered for this checkpoint type → no change → CONTINUE
        decision = Decision.CONTINUE
        state = ""
        reason = "no_trigger"
    else:
        decision = step.decision
        state = step.grounding_state.value
        reason = result.stop_reason

    return Prediction(
        sample_id=str(sample["id"]),
        scenario=str(sample["scenario"]),
        method="FSM",
        checkpoint=checkpoint,
        gold_state=str(sample["gold_state"]).upper(),
        gold_decision=Decision(str(sample["gold_decision"]).upper()),
        predicted_decision=decision,
        predicted_state=state,
        reason=reason,
        question=step.question if step else "",
        raw=result.to_dict(),
    )


# ---------------------------------------------------------------------------
# Per-sample runners
# ---------------------------------------------------------------------------

def run_sample(
    sample: dict[str, Any],
    method: str,
    *,
    handler: Any,
    gdino: Any,
    threshold: float,
    min_frames_gap: int = 1,
) -> Prediction:
    m = method.lower()
    sid = f"{sample['id']}_{m}"

    if m == "static":
        result = run_static_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            session_prefix=sid,
        )
        return _prediction_from_static(sample, result)

    if m == "memory":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_vlm_gate=False,
            session_prefix=sid,
        )
        return _prediction_from_memory(sample, result)

    if m == "memory_vlm":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_vlm_gate=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_VLM"
        return pred

    if m == "memory_vlm_judge":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_vlm_judge=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_VLM_JUDGE"
        return pred

    if m == "memory_qwen":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_qwen_judge=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_QWEN"
        return pred

    if m == "memory_qwen_detect":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_qwen_detect=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_QWEN_DETECT"
        return pred

    if m == "memory_finetuned":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_finetuned_judge=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_FINETUNED"
        return pred

    if m == "memory_ensemble":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_ensemble_judge=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_ENSEMBLE"
        return pred

    if m == "memory_qwen_gate":
        result = run_memory_pipeline(
            sample["initial_img"],
            sample["c1_img"],
            sample["c2_img"],
            sample["task"],
            sample["target_label"],
            sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            use_qwen_gate=True,
            session_prefix=sid,
        )
        pred = _prediction_from_memory(sample, result)
        pred.method = "MEMORY_QWEN_GATE"
        return pred

    if m == "fsm":
        fsm = MemoryFSM(
            task_description=sample["task"],
            target_label=sample["target_label"],
            destination_label=sample["destination_label"],
            handler=handler,
            gdino=gdino,
            threshold=threshold,
            use_memory_context=True,
            session_prefix=sid,
            min_frames_gap=min_frames_gap,
        )

        video_path = sample.get("video_path") or sample.get("video")
        if video_path:
            # Full video evaluation: frame-by-frame trigger detection
            frames = load_video_frames(video_path)
            result = fsm.run(frames, frame_t0=sample.get("initial_img"))
        else:
            # Fallback: use checkpoint images as discrete frames
            # t0 → [c1_img, c2_img]; trigger expected on each image
            result = fsm.run(
                [sample["c1_img"], sample["c2_img"]],
                frame_t0=sample["initial_img"],
            )
        return _prediction_from_fsm(sample, result)

    raise ValueError(f"Unknown method {method!r}. Choose from: {DEFAULT_METHODS}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[Prediction],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Prediction]] = {}
    for p in predictions:
        grouped.setdefault(p.method, []).append(p)

    out: dict[str, dict[str, float | int]] = {}
    for method, rows in grouped.items():
        total = len(rows)
        correct = sum(1 for r in rows if r.correct)
        should_not_cont = [r for r in rows if r.gold_decision != Decision.CONTINUE]
        should_cont = [r for r in rows if r.gold_decision == Decision.CONTINUE]
        false_cont = sum(
            1 for r in should_not_cont
            if r.predicted_decision == Decision.CONTINUE
        )
        false_alarm = sum(
            1 for r in should_cont
            if r.predicted_decision != Decision.CONTINUE
        )
        state_rows = [r for r in rows if r.predicted_state]
        state_correct = sum(
            1 for r in state_rows
            if r.predicted_state == r.gold_state
        )
        out[method] = {
            "n":                 total,
            "correct":           correct,
            "decision_accuracy": round(correct / total, 4) if total else 0.0,
            "miss_rate":         round(false_cont / len(should_not_cont), 4) if should_not_cont else 0.0,
            "false_alarm_rate":  round(false_alarm / len(should_cont), 4) if should_cont else 0.0,
            "state_accuracy":    round(state_correct / len(state_rows), 4) if state_rows else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "sample_id", "scenario", "method", "checkpoint",
    "gold_state", "gold_decision",
    "predicted_decision", "predicted_state",
    "reason", "question", "explanation", "correct",
]


def write_predictions_csv(path: str | Path, preds: list[Prediction]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for p in preds:
            row = p.to_dict()
            writer.writerow({k: row[k] for k in _CSV_FIELDS})


def write_metrics_json(path: str | Path, metrics: dict) -> None:
    Path(path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory ablation evaluation (static vs memory)."
    )
    parser.add_argument("manifest", help="Path to .jsonl or .json manifest")
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to evaluate (default: static memory memory_vlm memory_vlm_judge memory_qwen fsm)",
    )
    parser.add_argument("--model-type", default="fs_prompt",
                        choices=["fs_prompt", "finetune"])
    parser.add_argument("--adapter-ckpt", default="")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Pixel distance threshold for check_grounding")
    parser.add_argument("--dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--min-frames-gap", type=int, default=1,
        help=(
            "FSM: min frames between two trigger evaluations. "
            "Use 1 when evaluating with checkpoint images only (default). "
            "Increase for full video to avoid rapid re-triggering."
        ),
    )
    parser.add_argument("--predictions-csv", default="",
                        help="Output predictions CSV path")
    parser.add_argument("--metrics-json", default="",
                        help="Output metrics JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.model_type == "finetune" and not args.adapter_ckpt:
        parser.error("--model-type finetune requires --adapter-ckpt")

    # ── Load manifest ────────────────────────────────────────────────────────
    samples = load_manifest(args.manifest)
    logger.info("Loaded %d samples", len(samples))

    # ── Load models (once, shared across all samples) ────────────────────────
    logger.info("Loading AmbRes handler (%s) …", args.model_type)
    from extraction.ambres_g0_extractor import _make_handler
    handler = _make_handler(args.model_type, args.adapter_ckpt, use_detection=False)

    logger.info(
        "Loading GroundingDINO (box=%.2f, text=%.2f) …",
        args.dino_box_threshold, args.dino_text_threshold,
    )
    from module.models.ambres.gdino import GroundingDINO
    gdino = GroundingDINO(
        box_threshold=args.dino_box_threshold,
        text_threshold=args.dino_text_threshold,
    )
    logger.info("Models ready.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    predictions: list[Prediction] = []
    n = len(samples)
    for i, sample in enumerate(samples):
        for method in args.methods:
            logger.info("[%d/%d] %s  method=%s", i + 1, n, sample["id"], method)
            try:
                pred = run_sample(
                    sample, method,
                    handler=handler,
                    gdino=gdino,
                    threshold=args.threshold,
                    min_frames_gap=args.min_frames_gap,
                )
                predictions.append(pred)
                status = "OK " if pred.correct else "FAIL"
                logger.info(
                    "  %s  gold=%s  pred=%s",
                    status,
                    pred.gold_decision.value,
                    pred.predicted_decision.value,
                )
            except Exception as exc:
                logger.error("FAILED %s %s: %s", sample["id"], method, exc, exc_info=True)

    # ── Report ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.predictions_csv:
        write_predictions_csv(args.predictions_csv, predictions)
        logger.info("Predictions saved → %s", args.predictions_csv)
    if args.metrics_json:
        write_metrics_json(args.metrics_json, metrics)
        logger.info("Metrics saved → %s", args.metrics_json)


if __name__ == "__main__":
    main()
