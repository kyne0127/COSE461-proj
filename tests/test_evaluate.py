from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate import (
    EvalPrediction,
    compute_metrics,
    image_path_summary,
    load_manifest,
    manifest_summary,
    write_metrics_json,
    write_predictions_csv,
)
from monitoring.consistency_monitor import Decision


def _sample_dict(tmp_path, **overrides):
    for name in ("t0.png", "c1.png", "c2.png"):
        (tmp_path / name).write_bytes(b"not-real-image-needed-for-loader")
    data = {
        "id": "s1_001",
        "scenario": "S1",
        "task": "place red block on gray tray",
        "initial_img": "t0.png",
        "c1_img": "c1.png",
        "c2_img": "c2.png",
        "checkpoint": "C1",
        "gold_state": "CLEAR",
        "gold_decision": "CONTINUE",
        "target_label": "red block",
        "destination_label": "gray tray",
    }
    data.update(overrides)
    return data


def _pred(method, gold, pred, state="", gold_state="CLEAR", sample_id="s"):
    return EvalPrediction(
        sample_id=sample_id,
        scenario="S",
        method=method,
        checkpoint="C1",
        gold_state=gold_state,
        gold_decision=Decision(gold),
        predicted_decision=Decision(pred),
        predicted_state=state,
    )


class TestManifestLoading:
    def test_load_json_list_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([_sample_dict(tmp_path)]))

        samples = load_manifest(manifest)

        assert len(samples) == 1
        assert samples[0].sample_id == "s1_001"
        assert samples[0].gold_decision == Decision.CONTINUE
        assert samples[0].checkpoint_img.endswith("c1.png")

    def test_load_json_object_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"samples": [_sample_dict(tmp_path)]}))

        samples = load_manifest(manifest)

        assert len(samples) == 1
        assert samples[0].target_label == "red block"

    def test_load_jsonl_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps(_sample_dict(tmp_path)) + "\n")

        samples = load_manifest(manifest)

        assert len(samples) == 1
        assert samples[0].destination_label == "gray tray"

    def test_missing_required_field_raises(self, tmp_path):
        raw = _sample_dict(tmp_path)
        raw.pop("target_label")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([raw]))

        with pytest.raises(ValueError, match="missing fields"):
            load_manifest(manifest)

    def test_invalid_decision_raises(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([_sample_dict(tmp_path, gold_decision="MAYBE")]))

        with pytest.raises(ValueError, match="gold_decision"):
            load_manifest(manifest)

    def test_invalid_checkpoint_raises(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([_sample_dict(tmp_path, checkpoint="C3")]))

        with pytest.raises(ValueError, match="checkpoint"):
            load_manifest(manifest)

    def test_manifest_summary_counts_samples(self, tmp_path):
        samples = [
            _sample_dict(tmp_path, id="s1", scenario="S1", checkpoint="C1", gold_decision="CONTINUE"),
            _sample_dict(tmp_path, id="s2", scenario="S2", checkpoint="C2", gold_decision="ASK"),
        ]
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(samples))

        summary = manifest_summary(load_manifest(manifest))

        assert summary["n_samples"] == 2
        assert summary["scenarios"] == {"S1": 1, "S2": 1}
        assert summary["checkpoints"] == {"C1": 1, "C2": 1}
        assert summary["gold_decisions"] == {"CONTINUE": 1, "ASK": 1}

    def test_image_path_summary_reports_existing_and_missing_paths(self, tmp_path):
        sample = _sample_dict(tmp_path, c2_img="missing.png")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([sample]))

        summary = image_path_summary(load_manifest(manifest))

        assert summary["n_unique_images"] == 3
        assert summary["n_existing_images"] == 2
        assert summary["n_missing_images"] == 1
        assert summary["missing_images"][0].endswith("missing.png")


class TestMetrics:
    def test_decision_accuracy_by_method(self):
        metrics = compute_metrics([
            _pred("B1", "CONTINUE", "CONTINUE", sample_id="a"),
            _pred("B1", "ASK", "CONTINUE", sample_id="b"),
            _pred("B2", "ASK", "ASK", sample_id="c"),
        ])

        assert metrics["B1"]["n"] == 2
        assert metrics["B1"]["correct"] == 1
        assert metrics["B1"]["decision_accuracy"] == 0.5
        assert metrics["B2"]["decision_accuracy"] == 1.0

    def test_miss_rate_counts_false_continue_on_ask_stop_gold(self):
        metrics = compute_metrics([
            _pred("M", "ASK", "CONTINUE", sample_id="a"),
            _pred("M", "STOP", "ASK", sample_id="b"),
            _pred("M", "CONTINUE", "CONTINUE", sample_id="c"),
        ])

        assert metrics["M"]["miss_rate"] == 0.5

    def test_false_alarm_rate_counts_ask_stop_on_continue_gold(self):
        metrics = compute_metrics([
            _pred("M", "CONTINUE", "ASK", sample_id="a"),
            _pred("M", "CONTINUE", "CONTINUE", sample_id="b"),
            _pred("M", "ASK", "ASK", sample_id="c"),
        ])

        assert metrics["M"]["false_alarm_rate"] == 0.5

    def test_state_accuracy_only_uses_predictions_with_state(self):
        metrics = compute_metrics([
            _pred("OURS", "CONTINUE", "CONTINUE", state="CLEAR", gold_state="CLEAR", sample_id="a"),
            _pred("OURS", "ASK", "ASK", state="INVALID_TARGET", gold_state="AMBIGUOUS_TARGET", sample_id="b"),
            _pred("B1", "ASK", "ASK", state="", gold_state="AMBIGUOUS_TARGET", sample_id="c"),
        ])

        assert metrics["OURS"]["state_n"] == 2
        assert metrics["OURS"]["state_accuracy"] == 0.5
        assert metrics["B1"]["state_n"] == 0
        assert metrics["B1"]["state_accuracy"] == 0.0


class TestWriters:
    def test_write_predictions_csv(self, tmp_path):
        out = tmp_path / "predictions.csv"
        write_predictions_csv(out, [
            _pred("B1", "CONTINUE", "CONTINUE", sample_id="s1"),
        ])

        text = out.read_text()
        assert "sample_id" in text
        assert "B1" in text

    def test_write_metrics_json(self, tmp_path):
        out = tmp_path / "metrics.json"
        metrics = {"B1": {"n": 1, "decision_accuracy": 1.0}}
        write_metrics_json(out, metrics)

        assert json.loads(out.read_text()) == metrics
