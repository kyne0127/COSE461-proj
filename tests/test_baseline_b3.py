from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.b3_count_rule import METHOD_NAME, run_b3_count_rule
from monitoring.consistency_monitor import Decision


TASK = "place red block on gray tray"
TARGET = "red block"
DEST = "gray tray"
SESSION = "test_b3"


def _reset():
    return {"success": True}


def _query(objects, ambiguous=False):
    return {
        "success": True,
        "task_ambiguous": ambiguous,
        "task_objects": list(objects),
        "clarifying_question": "Which one?" if ambiguous else "",
    }


def _query_object_list(objects, ambiguous=False):
    return {
        "success": True,
        "ambiguity": ambiguous,
        "object_list": list(objects),
        "clarifying_question": "Which one?" if ambiguous else "",
    }


@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "checkpoint.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


@pytest.fixture()
def mock_handler():
    return MagicMock()


class TestB3DecisionPolicy:
    def test_one_target_one_destination_returns_continue(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(
                image_path,
                TASK,
                TARGET,
                DEST,
                checkpoint="C1",
                session_id=SESSION,
            )

        assert result.method == METHOD_NAME
        assert result.decision == Decision.CONTINUE
        assert result.metadata["n_target"] == 1
        assert result.metadata["n_destination"] == 1
        assert result.metadata["stores_g0"] is False

    def test_missing_target_returns_stop(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.STOP
        assert TARGET in result.reason

    def test_multiple_targets_returns_ask(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, TARGET, DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.ASK
        assert result.metadata["n_target"] == 2
        assert TARGET in result.question

    def test_missing_destination_returns_ask(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.ASK
        assert result.metadata["n_destination"] == 0
        assert DEST in result.question

    def test_multiple_destinations_returns_ask(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, DEST, DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.ASK
        assert result.metadata["n_destination"] == 2
        assert DEST in result.question


class TestB3CountSemantics:
    def test_exact_full_label_match_only(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            _query(["block", "red block", "gray tray", "tray"]),
        ]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.CONTINUE
        assert result.metadata["n_target"] == 1
        assert result.metadata["n_destination"] == 1

    def test_ignores_vlm_ambiguity_flag_when_counts_are_clear(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, DEST], ambiguous=True)]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.CONTINUE
        assert result.metadata["ignores_vlm_ambiguity"] is True

    def test_object_list_key_variant_is_supported(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query_object_list([TARGET, DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            result = run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

        assert result.decision == Decision.CONTINUE


class TestB3CallSequence:
    def test_calls_reset_then_query_only(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, DEST])]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

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
        mock_handler.handle.side_effect = [_reset(), _query([TARGET, DEST])]
        with patch("baselines.b3_count_rule._make_handler") as make_handler:
            result = run_b3_count_rule(
                image_path,
                TASK,
                TARGET,
                DEST,
                handler=mock_handler,
                session_id=SESSION,
            )

        make_handler.assert_not_called()
        assert result.decision == Decision.CONTINUE


class TestB3ErrorPaths:
    def test_invalid_checkpoint_raises_value_error(self, image_path, mock_handler):
        with pytest.raises(ValueError, match="checkpoint"):
            run_b3_count_rule(
                image_path,
                TASK,
                TARGET,
                DEST,
                checkpoint="C3",
                handler=mock_handler,
                session_id=SESSION,
            )

    def test_empty_target_label_raises_value_error(self, image_path, mock_handler):
        with pytest.raises(ValueError, match="target_label"):
            run_b3_count_rule(image_path, TASK, "", DEST, handler=mock_handler)

    def test_empty_destination_label_raises_value_error(self, image_path, mock_handler):
        with pytest.raises(ValueError, match="destination_label"):
            run_b3_count_rule(image_path, TASK, TARGET, "", handler=mock_handler)

    def test_query_failure_raises_runtime_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            {"success": False, "error": "Inference failed"},
        ]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match="AmbRes query failed"):
                run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)

    def test_missing_object_list_raises_value_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            {"success": True, "task_ambiguous": False},
        ]
        with patch("baselines.b3_count_rule._make_handler", return_value=mock_handler):
            with pytest.raises(ValueError, match="object list"):
                run_b3_count_rule(image_path, TASK, TARGET, DEST, session_id=SESSION)
