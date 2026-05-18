from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.b1_initial_only import METHOD_NAME, run_b1_initial_only
from baselines.common import BaselineDecision, BaselineResult
from monitoring.consistency_monitor import Decision


TASK = "place cup on box"
SESSION = "test_b1"


def _reset():
    return {"success": True}


def _query_task_ambiguous(ambiguous: bool = False, question: str = ""):
    return {
        "success": True,
        "task_ambiguous": ambiguous,
        "task_objects": ["cup", "box"],
        "clarifying_question": question,
    }


def _query_ambiguity(ambiguous: bool = False, question: str = ""):
    return {
        "success": True,
        "ambiguity": ambiguous,
        "object_list": ["cup", "box"],
        "clarifying_question": question,
    }


@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "img.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


@pytest.fixture()
def mock_handler():
    return MagicMock()


class TestB1HappyPath:
    def test_clear_query_returns_continue(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_task_ambiguous(False)]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            result = run_b1_initial_only(image_path, TASK, session_id=SESSION)

        assert result.method == METHOD_NAME
        assert result.decision == Decision.CONTINUE
        assert result.question == ""
        assert result.metadata["uses_checkpoint"] is False
        assert result.metadata["stores_g0"] is False

    def test_ambiguous_query_returns_ask_with_question(self, image_path, mock_handler):
        question = "Which cup did you mean?"
        mock_handler.handle.side_effect = [_reset(), _query_task_ambiguous(True, question)]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            result = run_b1_initial_only(image_path, TASK, session_id=SESSION)

        assert result.decision == Decision.ASK
        assert result.question == question
        assert "ambiguous" in result.reason.lower()


class TestB1AmbiguityKeys:
    def test_task_ambiguous_key_is_supported(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_task_ambiguous(True, "Which one?")]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            result = run_b1_initial_only(image_path, TASK, session_id=SESSION)
        assert result.decision == Decision.ASK

    def test_ambiguity_key_is_supported(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_ambiguity(True, "Which one?")]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            result = run_b1_initial_only(image_path, TASK, session_id=SESSION)
        assert result.decision == Decision.ASK


class TestB1CallSequence:
    def test_calls_reset_then_query_only(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_task_ambiguous(False)]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            run_b1_initial_only(image_path, TASK, session_id=SESSION)

        assert mock_handler.handle.call_count == 2
        reset_call, query_call = mock_handler.handle.call_args_list

        assert reset_call.args == ("reset", {}, [], SESSION)
        assert query_call.args[0] == "query"
        assert query_call.args[1] == {"task_description": TASK}
        assert len(query_call.args[2]) == 1
        assert query_call.args[3] == SESSION

        called_methods = [c.args[0] for c in mock_handler.handle.call_args_list]
        assert "respond" not in called_methods
        assert "detect" not in called_methods

    def test_injected_handler_skips_make_handler(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_task_ambiguous(False)]
        with patch("baselines.b1_initial_only._make_handler") as make_handler:
            result = run_b1_initial_only(
                image_path,
                TASK,
                handler=mock_handler,
                session_id=SESSION,
            )

        make_handler.assert_not_called()
        assert result.decision == Decision.CONTINUE


class TestB1ErrorPaths:
    def test_query_failure_raises_runtime_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            {"success": False, "error": "Inference failed"},
        ]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match="AmbRes query failed"):
                run_b1_initial_only(image_path, TASK, session_id=SESSION)

    def test_missing_ambiguity_flag_raises_value_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            {"success": True, "task_objects": ["cup", "box"]},
        ]
        with patch("baselines.b1_initial_only._make_handler", return_value=mock_handler):
            with pytest.raises(ValueError, match="ambiguity"):
                run_b1_initial_only(image_path, TASK, session_id=SESSION)


class TestBaselineResult:
    def test_to_dict_is_json_serializable(self):
        result = BaselineResult(
            method="TEST",
            decision=Decision.CONTINUE,
            reason="ok",
            question="",
            raw_output={"task_ambiguous": False},
            metadata={"n": 1},
        )

        data = result.to_dict()
        assert data["decision"] == "CONTINUE"
        json.dumps(data)

    def test_baseline_decision_aliases_decision_enum(self):
        assert BaselineDecision is Decision
