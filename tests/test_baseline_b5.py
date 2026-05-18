from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.b5_llm_judge import (
    METHOD_NAME,
    build_prompt,
    parse_llm_decision,
    run_b5_llm_judge,
)
from monitoring.consistency_monitor import Decision


TASK = "place red block on gray tray"


@pytest.fixture()
def image_paths(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p0 = tmp_path / "initial.png"
    p1 = tmp_path / "checkpoint.png"
    Image.fromarray(arr, "RGB").save(str(p0))
    Image.fromarray(arr, "RGB").save(str(p1))
    return p0, p1


class RecordingClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


class TestB5HappyPath:
    def test_continue_output_returns_continue(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "CONTINUE", "reason": "Grounding is valid"})

        result = run_b5_llm_judge(
            p0,
            p1,
            TASK,
            checkpoint="C1",
            llm_client=client,
            model="test-vlm",
            temperature=0,
        )

        assert result.method == METHOD_NAME
        assert result.decision == Decision.CONTINUE
        assert result.question == ""
        assert result.metadata["stores_g0"] is False
        assert result.metadata["uses_coord"] is False
        assert result.metadata["uses_taxonomy"] is False

    def test_ask_output_sets_question_to_reason(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "ASK", "reason": "Multiple red blocks"})

        result = run_b5_llm_judge(p0, p1, TASK, llm_client=client)

        assert result.decision == Decision.ASK
        assert result.question == "Multiple red blocks"

    def test_stop_output_returns_stop(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "STOP", "reason": "Target is missing"})

        result = run_b5_llm_judge(p0, p1, TASK, llm_client=client)

        assert result.decision == Decision.STOP
        assert result.reason == "Target is missing"


class TestB5ClientContract:
    def test_client_receives_prompt_images_model_and_temperature(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "CONTINUE", "reason": "ok"})

        run_b5_llm_judge(
            p0,
            p1,
            TASK,
            checkpoint="C2",
            llm_client=client,
            model="mock-model",
            temperature=0.2,
        )

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["image_paths"] == [p0, p1]
        assert call["model"] == "mock-model"
        assert call["temperature"] == 0.2
        assert "Task: place red block on gray tray" in call["prompt"]
        assert "Checkpoint: C2" in call["prompt"]

    def test_prompt_contains_required_decisions(self):
        prompt = build_prompt(TASK, "C1")
        assert "CONTINUE" in prompt
        assert "ASK" in prompt
        assert "STOP" in prompt
        assert "target" in prompt
        assert "destination" in prompt


class TestB5Parsing:
    def test_parse_dict_output(self):
        decision, reason, raw = parse_llm_decision(
            {"decision": "continue", "reason": "still valid"}
        )
        assert decision == Decision.CONTINUE
        assert reason == "still valid"
        assert raw["decision"] == "continue"

    def test_parse_json_string_output(self):
        decision, reason, _ = parse_llm_decision(
            '{"decision": "ASK", "reason": "ambiguous target"}'
        )
        assert decision == Decision.ASK
        assert reason == "ambiguous target"

    def test_parse_json_code_fence_output(self):
        decision, reason, _ = parse_llm_decision(
            '```json\n{"decision": "STOP", "reason": "target missing"}\n```'
        )
        assert decision == Decision.STOP
        assert reason == "target missing"

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="valid JSON"):
            parse_llm_decision("CONTINUE because it is fine")

    def test_invalid_decision_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid LLM decision"):
            parse_llm_decision({"decision": "MAYBE", "reason": "unclear"})

    def test_non_string_non_dict_output_raises_type_error(self):
        with pytest.raises(TypeError, match="dict or str"):
            parse_llm_decision(["CONTINUE"])


class TestB5ErrorPaths:
    def test_invalid_checkpoint_raises_value_error(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "CONTINUE", "reason": "ok"})

        with pytest.raises(ValueError, match="checkpoint"):
            run_b5_llm_judge(p0, p1, TASK, checkpoint="C3", llm_client=client)

    def test_negative_temperature_raises_value_error(self, image_paths):
        p0, p1 = image_paths
        client = RecordingClient({"decision": "CONTINUE", "reason": "ok"})

        with pytest.raises(ValueError, match="temperature"):
            run_b5_llm_judge(p0, p1, TASK, temperature=-0.1, llm_client=client)
