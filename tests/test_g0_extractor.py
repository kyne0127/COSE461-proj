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

from extraction.ambres_g0_extractor import extract_g0  # noqa: E402

REAL_IMAGE = Path("/workspace/AmbRes/assets/images/real_0.png")
TASK = "place cup on box"
SESSION = "test_session"
DETECTIONS = {"cup": [[320, 240]], "box": [[100, 150]]}


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------

def _reset():
    return {"success": True}


def _query(ambiguous=False, objects=("cup", "box"), question=""):
    return {
        "success": True,
        "task_ambiguous": ambiguous,
        "task_objects": list(objects),
        "clarifying_question": question,
    }


def _respond(objects=("cup", "box")):
    return {"success": True, "task_objects": list(objects)}


def _detect(detections=None):
    return {"success": True, "detections": detections or DETECTIONS}


def _four_calls(mock_handler, objects=("cup", "box"), ambiguous=False):
    mock_handler.handle.side_effect = [
        _reset(),
        _query(ambiguous=ambiguous, objects=objects),
        _respond(objects=objects),
        _detect(),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "test.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


@pytest.fixture()
def mock_handler():
    return MagicMock()


# ---------------------------------------------------------------------------
# TestHappyPath
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_returns_correct_g0_structure(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, TASK, session_id=SESSION)
        assert g0["target"] == {"label": "cup", "coord": [320, 240]}
        assert g0["destination"] == {"label": "box", "coord": [100, 150]}
        assert g0["image_shape"] == [4, 4]

    def test_image_shape_is_list_of_two_ints(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, TASK, session_id=SESSION)
        h, w = g0["image_shape"]
        assert isinstance(h, int) and isinstance(w, int)

    def test_coord_values_are_python_ints(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, TASK, session_id=SESSION)
        for role in ("target", "destination"):
            x, y = g0[role]["coord"]
            assert isinstance(x, int) and isinstance(y, int)

    def test_exactly_four_handle_calls(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            extract_g0(image_path, TASK, session_id=SESSION)
        assert mock_handler.handle.call_count == 4


# ---------------------------------------------------------------------------
# TestCallSequence
# ---------------------------------------------------------------------------

class TestCallSequence:
    def _run(self, image_path, mock_handler, **kwargs):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            extract_g0(image_path, TASK, session_id=SESSION, **kwargs)
        return mock_handler.handle.call_args_list

    def test_reset_is_first_call(self, image_path, mock_handler):
        calls = self._run(image_path, mock_handler)
        a = calls[0].args
        assert a[0] == "reset"
        assert a[1] == {}
        assert a[2] == []
        assert a[3] == SESSION

    def test_query_is_second_call_with_task_description(self, image_path, mock_handler):
        calls = self._run(image_path, mock_handler)
        a = calls[1].args
        assert a[0] == "query"
        assert a[1] == {"task_description": TASK}
        assert len(a[2]) == 1  # image tensor
        assert a[3] == SESSION

    def test_respond_always_called_with_empty_string_even_when_non_ambiguous(
        self, image_path, mock_handler
    ):
        """핵심: ambiguity=False여도 respond는 반드시 빈 문자열로 호출돼야 함."""
        calls = self._run(image_path, mock_handler)
        a = calls[2].args
        assert a[0] == "respond"
        assert a[1] == {"response": ""}
        assert a[2] == []
        assert a[3] == SESSION

    def test_detect_is_fourth_call_with_role_labels(self, image_path, mock_handler):
        calls = self._run(image_path, mock_handler)
        a = calls[3].args
        assert a[0] == "detect"
        assert a[1] == {"objects": ["cup", "box"]}
        assert len(a[2]) == 1  # image tensor
        assert a[3] == SESSION


# ---------------------------------------------------------------------------
# TestAmbiguityPaths
# ---------------------------------------------------------------------------

class TestAmbiguityPaths:
    def test_ambiguous_raises_runtime_error_by_default(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            _query(ambiguous=True, question="Which cup?"),
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match="ambiguity=true"):
                extract_g0(image_path, TASK, session_id=SESSION)

    def test_ambiguous_error_contains_clarifying_question(self, image_path, mock_handler):
        question = "Which cup did you mean?"
        mock_handler.handle.side_effect = [
            _reset(),
            _query(ambiguous=True, question=question),
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match=question):
                extract_g0(image_path, TASK, session_id=SESSION, allow_ambiguous=False)

    def test_ambiguous_stops_before_respond_and_detect(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            _query(ambiguous=True, question="Which box?"),
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError):
                extract_g0(image_path, TASK, session_id=SESSION)
        assert mock_handler.handle.call_count == 2
        called = [c.args[0] for c in mock_handler.handle.call_args_list]
        assert "respond" not in called
        assert "detect" not in called

    def test_allow_ambiguous_runs_all_four_steps(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            _query(ambiguous=True, question="Which box?"),
            _respond(),
            _detect(),
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, TASK, session_id=SESSION, allow_ambiguous=True)
        assert mock_handler.handle.call_count == 4
        assert mock_handler.handle.call_args_list[2].args[1] == {"response": ""}
        assert g0["target"]["label"] == "cup"


# ---------------------------------------------------------------------------
# TestErrorPaths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_reset_failure_raises_runtime_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            {"success": False, "error": "Reset error"},
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match="AmbRes reset failed"):
                extract_g0(image_path, TASK, session_id=SESSION)

    def test_query_failure_raises_runtime_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            {"success": False, "error": "Inference failed"},
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(RuntimeError, match="AmbRes query failed"):
                extract_g0(image_path, TASK, session_id=SESSION)

    def test_missing_detection_label_raises_value_error(self, image_path, mock_handler):
        mock_handler.handle.side_effect = [
            _reset(),
            _query(),
            _respond(),
            _detect({"box": [[100, 150]]}),  # "cup" 누락
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            with pytest.raises(ValueError, match="cup"):
                extract_g0(image_path, TASK, session_id=SESSION)


# ---------------------------------------------------------------------------
# TestKeyNameVariants
# ---------------------------------------------------------------------------

class TestKeyNameVariants:
    def test_ambiguity_and_object_list_key_names(self, image_path, mock_handler):
        """handler가 'ambiguity'/'object_list' 키를 반환하는 경우도 처리."""
        mock_handler.handle.side_effect = [
            {"success": True},
            {
                "success": True,
                "ambiguity": False,
                "object_list": ["mug", "tray"],
                "clarifying_question": "",
            },
            {"success": True, "object_list": ["mug", "tray"]},
            {"success": True, "detections": {"mug": [[200, 300]], "tray": [[400, 100]]}},
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, "pick mug and put it on tray", session_id=SESSION)
        assert g0["target"]["label"] == "mug"
        assert g0["destination"]["label"] == "tray"

    def test_step2_empty_object_list_falls_back_to_step1(self, image_path, mock_handler):
        """respond가 빈 목록 반환 시 step1의 object_list로 fallback."""
        mock_handler.handle.side_effect = [
            _reset(),
            _query(objects=["cup", "box"]),
            {"success": True, "task_objects": []},  # 빈 리스트 → step1 fallback
            _detect(),
        ]
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(image_path, TASK, session_id=SESSION)
        assert g0["target"]["label"] == "cup"
        assert g0["destination"]["label"] == "box"


# ---------------------------------------------------------------------------
# TestMakeHandlerArgs
# ---------------------------------------------------------------------------

class TestMakeHandlerArgs:
    def test_make_handler_called_with_defaults(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler) as mk:
            extract_g0(image_path, TASK, session_id=SESSION)
        mk.assert_called_once_with("fs_prompt", "", use_detection=False)

    def test_make_handler_called_with_custom_args(self, image_path, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler) as mk:
            extract_g0(
                image_path, TASK,
                model_type="finetune",
                adapter_ckpt="my_ckpt",
                session_id=SESSION,
            )
        mk.assert_called_once_with("finetune", "my_ckpt", use_detection=False)


# ---------------------------------------------------------------------------
# TestRealImage (선택적 — real_0.png 없으면 skip)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_IMAGE.exists(),
    reason="real_0.png not available",
)
class TestRealImage:
    def test_real_image_shape(self, mock_handler):
        _four_calls(mock_handler)
        with patch("extraction.ambres_g0_extractor._make_handler", return_value=mock_handler):
            g0 = extract_g0(REAL_IMAGE, TASK, session_id=SESSION)
        assert g0["image_shape"] == [480, 640]
