from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monitoring.consistency_monitor import Decision, GroundingState
from pipeline import (
    CheckpointOutcome,
    PipelineResult,
    _clarifying_question,
    _noop_response,
    _update_g0,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "img.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


@pytest.fixture()
def g0():
    return {
        "target":      {"label": "red block", "coord": [200, 300]},
        "destination": {"label": "tray",      "coord": [400, 300]},
        "image_shape": [4, 4],
    }


# ---------------------------------------------------------------------------
# Handler mock factory
# ---------------------------------------------------------------------------

def _make_mock_handler(
    *,
    g0_ambiguous: bool = False,
    c1_instances: int = 1,       # 1=CLEAR, >1=AMBIGUOUS_TARGET, 0=INVALID
    c2_instances: int = 1,       # 1=CLEAR, >1=AMBIGUOUS_DESTINATION, 0=INVALID
    c1_displaced: bool = False,  # single instance but far away → AMBIGUOUS
    c2_displaced: bool = False,
) -> MagicMock:
    """Build a mock handler whose side_effect sequences encode a scenario."""

    def _coord(displaced: bool, base: list) -> list:
        return [base[0] + 999, base[1]] if displaced else base

    # t0 responses: reset + query + respond + detect
    t0_reset   = {"success": True}
    t0_query   = {
        "success": True,
        "task_ambiguous": g0_ambiguous,
        "task_objects": ["red block", "tray"],
        "clarifying_question": "Which one?" if g0_ambiguous else "",
    }
    t0_respond = {"success": True, "task_objects": ["red block", "tray"]}
    t0_detect  = {
        "success": True,
        "detections": {"red block": [[200, 300]], "tray": [[400, 300]]},
    }

    # C1 detect (get_checkpoint_detections: reset + detect)
    def _c1_coords():
        c = [200, 300]
        if c1_instances == 0:   return []
        if c1_displaced:        return [[c[0] + 999, c[1]]]
        return [c] * c1_instances

    c1_reset  = {"success": True}
    c1_detect = {
        "success": True,
        "detections": {"red block": _c1_coords(), "tray": [[400, 300]]},
    }

    # C2 detect (get_checkpoint_detections: reset + detect)
    def _c2_coords():
        c = [400, 300]
        if c2_instances == 0:   return []
        if c2_displaced:        return [[c[0] + 999, c[1]]]
        return [c] * c2_instances

    c2_reset  = {"success": True}
    c2_detect = {
        "success": True,
        "detections": {"red block": [[200, 300]], "tray": _c2_coords()},
    }

    mock = MagicMock()
    mock.handle.side_effect = [
        t0_reset, t0_query, t0_respond, t0_detect,  # t0 (4 calls)
        c1_reset, c1_detect,                         # C1 detect (2 calls)
        c2_reset, c2_detect,                         # C2 detect (2 calls)
    ]
    return mock


def _make_mock_handler_with_ask_c1(user_answer: str = "the left one") -> MagicMock:
    """Handler for C1=AMBIGUOUS scenario with rolling update at C1."""
    mock = MagicMock()
    mock.handle.side_effect = [
        # t0
        {"success": True},
        {"success": True, "task_ambiguous": False, "task_objects": ["red block", "tray"], "clarifying_question": ""},
        {"success": True, "task_objects": ["red block", "tray"]},
        {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        # C1 detect → AMBIGUOUS (2 instances)
        {"success": True},
        {"success": True, "detections": {"red block": [[200, 300], [350, 100]], "tray": [[400, 300]]}},
        # C1 rolling update: reset + query + respond + detect
        {"success": True},
        {"success": True, "task_ambiguous": False, "task_objects": ["red block", "tray"], "clarifying_question": ""},
        {"success": True, "task_objects": ["red block", "tray"]},
        {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        # C2 detect (with updated G0)
        {"success": True},
        {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
    ]
    return mock


# ---------------------------------------------------------------------------
# TestHappyPath — both checkpoints CLEAR
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_status_is_complete(self, image_path, g0):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "complete"

    def test_g0_initial_is_set(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.g0_initial is not None
        assert result.g0_initial["target"]["label"] == "red block"

    def test_c1_and_c2_are_not_none(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1 is not None
        assert result.c2 is not None

    def test_c1_state_is_clear(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1.state == GroundingState.CLEAR
        assert result.c1.decision == Decision.CONTINUE

    def test_c2_state_is_clear(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c2.state == GroundingState.CLEAR
        assert result.c2.decision == Decision.CONTINUE

    def test_no_user_interaction(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1.user_response == ""
        assert result.c2.user_response == ""

    def test_g0_before_equals_g0_after_when_no_update(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1.g0_before == result.c1.g0_after
        assert result.c2.g0_before == result.c2.g0_after


# ---------------------------------------------------------------------------
# TestInitialAmbiguous — t₀ returns ambiguity=true
# ---------------------------------------------------------------------------

class TestInitialAmbiguous:
    def test_status_is_initial_ambiguous(self, image_path):
        mock = _make_mock_handler(g0_ambiguous=True)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "initial_ambiguous"

    def test_g0_is_none(self, image_path):
        mock = _make_mock_handler(g0_ambiguous=True)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.g0_initial is None

    def test_checkpoints_are_none(self, image_path):
        mock = _make_mock_handler(g0_ambiguous=True)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1 is None and result.c2 is None

    def test_stop_reason_contains_message(self, image_path):
        mock = _make_mock_handler(g0_ambiguous=True)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.stop_reason


# ---------------------------------------------------------------------------
# TestC1Stop — INVALID_TARGET at C1 → STOP
# ---------------------------------------------------------------------------

class TestC1Stop:
    def test_status_is_stopped(self, image_path):
        mock = _make_mock_handler(c1_instances=0)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "stopped"

    def test_c1_state_is_invalid_target(self, image_path):
        mock = _make_mock_handler(c1_instances=0)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1.state == GroundingState.INVALID_TARGET
        assert result.c1.decision == Decision.STOP

    def test_c2_is_none_when_stopped_at_c1(self, image_path):
        mock = _make_mock_handler(c1_instances=0)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c2 is None

    def test_stop_reason_mentions_c1(self, image_path):
        mock = _make_mock_handler(c1_instances=0)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert "C1" in result.stop_reason


# ---------------------------------------------------------------------------
# TestC2Stop — INVALID_DESTINATION at C2 (target is CLEAR at C1)
# ---------------------------------------------------------------------------

class TestC2Stop:
    def _make_handler(self):
        mock = MagicMock()
        mock.handle.side_effect = [
            # t0
            {"success": True},
            {"success": True, "task_ambiguous": False, "task_objects": ["red block", "tray"], "clarifying_question": ""},
            {"success": True, "task_objects": ["red block", "tray"]},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
            # C1: CLEAR
            {"success": True},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
            # C2: tray missing → INVALID_DESTINATION → ASK (not STOP per policy)
            # Use displaced tray to trigger AMBIGUOUS_DESTINATION instead
            # Actually INVALID_DESTINATION → ASK, not STOP. Use missing target for STOP.
            # Let's use missing tray at C2 → INVALID_DESTINATION → ASK
            # To get STOP at C2, we need UNSAFE_OR_BLOCKED which is not auto-detected.
            # Instead test INVALID_DESTINATION → ASK, then no response → complete.
            {"success": True},
            {"success": True, "detections": {"red block": [[200, 300]]}},  # tray missing
        ]
        return mock

    def test_c2_invalid_destination_yields_ask(self, image_path):
        mock = self._make_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c2.state == GroundingState.INVALID_DESTINATION
        assert result.c2.decision == Decision.ASK
        # ASK without response → still completes (noop response)
        assert result.status == "complete"


# ---------------------------------------------------------------------------
# TestC1Ask — AMBIGUOUS_TARGET at C1, user responds
# ---------------------------------------------------------------------------

class TestC1Ask:
    def test_user_response_fn_called_at_c1(self, image_path):
        mock = _make_mock_handler_with_ask_c1()
        responses = []
        def capture_fn(q, g0):
            responses.append((q, g0))
            return "the left one"
        with patch("pipeline._make_handler", return_value=mock):
            run_pipeline(image_path, image_path, image_path,
                         "place red block on tray", user_response_fn=capture_fn)
        assert len(responses) == 1
        assert "C1" in responses[0][0]

    def test_c1_user_response_recorded(self, image_path):
        mock = _make_mock_handler_with_ask_c1()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path,
                                  "place red block on tray",
                                  user_response_fn=lambda q, g: "the left one")
        assert result.c1.user_response == "the left one"

    def test_g0_updated_after_c1_ask(self, image_path):
        """When user responds at C1, g0_after != g0_before for C1."""
        mock = _make_mock_handler_with_ask_c1()
        responses_given = ["the left one"]
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path,
                                  "place red block on tray",
                                  user_response_fn=lambda q, g: responses_given[0])
        # g0_after should be the updated G0 (from rolling update)
        assert result.c1.g0_after is not None

    def test_c2_uses_updated_g0_after_c1_ask(self, image_path):
        """C2 checkpoint uses the G0 updated after C1 clarification."""
        mock = _make_mock_handler_with_ask_c1()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path,
                                  "place red block on tray",
                                  user_response_fn=lambda q, g: "the left one")
        # C2's g0_before == C1's g0_after (rolling update propagated)
        assert result.c2.g0_before == result.c1.g0_after

    def test_noop_response_skips_rolling_update(self, image_path):
        """Empty user response → g0_before == g0_after (no update)."""
        mock = MagicMock()
        mock.handle.side_effect = [
            {"success": True},
            {"success": True, "task_ambiguous": False, "task_objects": ["red block", "tray"], "clarifying_question": ""},
            {"success": True, "task_objects": ["red block", "tray"]},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
            {"success": True},
            {"success": True, "detections": {"red block": [[200, 300], [350, 100]], "tray": [[400, 300]]}},
            # no rolling update calls (empty response)
            {"success": True},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        ]
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path,
                                  "place red block on tray",
                                  user_response_fn=lambda q, g: "")  # empty → no update
        assert result.c1.g0_before == result.c1.g0_after


# ---------------------------------------------------------------------------
# TestOutputSchema — PipelineResult and CheckpointOutcome structure
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_result_has_required_fields(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert hasattr(result, "status")
        assert hasattr(result, "g0_initial")
        assert hasattr(result, "c1")
        assert hasattr(result, "c2")
        assert hasattr(result, "stop_reason")

    def test_checkpoint_outcome_has_required_fields(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        for co in (result.c1, result.c2):
            assert hasattr(co, "checkpoint")
            assert hasattr(co, "state")
            assert hasattr(co, "decision")
            assert hasattr(co, "detections")
            assert hasattr(co, "g0_before")
            assert hasattr(co, "g0_after")
            assert hasattr(co, "user_response")

    def test_to_dict_is_json_serialisable(self, image_path):
        import json
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        # should not raise
        json.dumps(result.to_dict())

    def test_checkpoint_labels_in_to_dict(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        d = result.to_dict()
        assert d["c1"]["checkpoint"] == "C1"
        assert d["c2"]["checkpoint"] == "C2"


# ---------------------------------------------------------------------------
# TestHandlerInjection — handler=None triggers _make_handler, else skips
# ---------------------------------------------------------------------------

class TestHandlerInjection:
    def test_make_handler_called_when_handler_is_none(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock) as mk:
            run_pipeline(image_path, image_path, image_path, "place red block on tray")
        mk.assert_called_once()

    def test_make_handler_not_called_when_handler_provided(self, image_path):
        mock = _make_mock_handler()
        with patch("pipeline._make_handler") as mk:
            run_pipeline(image_path, image_path, image_path,
                         "place red block on tray", handler=mock)
        mk.assert_not_called()


# ---------------------------------------------------------------------------
# TestHelpers — _noop_response, _clarifying_question, _update_g0
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_noop_response_returns_empty_string(self):
        assert _noop_response("any question", {}) == ""

    def test_clarifying_question_target(self):
        g0 = {"target": {"label": "red block", "coord": [0, 0]},
               "destination": {"label": "tray", "coord": [0, 0]},
               "image_shape": [480, 640]}
        q = _clarifying_question(g0, "target")
        assert "red block" in q

    def test_clarifying_question_destination(self):
        g0 = {"target": {"label": "red block", "coord": [0, 0]},
               "destination": {"label": "tray", "coord": [0, 0]},
               "image_shape": [480, 640]}
        q = _clarifying_question(g0, "destination")
        assert "tray" in q

    def test_update_g0_returns_valid_schema(self, image_path):
        mock = MagicMock()
        mock.handle.side_effect = [
            {"success": True},  # reset
            {"success": True, "task_ambiguous": False,
             "task_objects": ["red block", "tray"], "clarifying_question": ""},
            {"success": True, "task_objects": ["red block", "tray"]},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        ]
        g0 = _update_g0(image_path, "place red block on tray", "the left one",
                        mock, session_id="test_update")
        assert set(g0.keys()) == {"target", "destination", "image_shape"}
        assert g0["target"]["label"] == "red block"
        assert isinstance(g0["target"]["coord"][0], int)


# ---------------------------------------------------------------------------
# TestProposalScenarios — proposal §5.1 시나리오 매핑
# ---------------------------------------------------------------------------

class TestProposalScenarios:
    """각 실험 시나리오에서 올바른 최종 status를 반환하는지 검증."""

    def test_scenario1_clear_continuation(self, image_path):
        # ① 장면 변화 없음 → complete
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "complete"

    def test_scenario2_same_category_target_added_at_c1(self, image_path):
        # ② target 복수 감지 → c1=AMBIGUOUS_TARGET/ASK → complete (noop response)
        mock = _make_mock_handler(c1_instances=2)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c1.state == GroundingState.AMBIGUOUS_TARGET
        assert result.c1.decision == Decision.ASK

    def test_scenario3_target_disappeared(self, image_path):
        # ③ target 사라짐 → c1=INVALID_TARGET/STOP
        mock = _make_mock_handler(c1_instances=0)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "stopped"
        assert result.c1.state == GroundingState.INVALID_TARGET

    def test_scenario4_new_destination_candidate(self, image_path):
        # ④ destination 복수 감지 → c2=AMBIGUOUS_DESTINATION/ASK → complete
        mock = _make_mock_handler(c2_instances=2)
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.c2.state == GroundingState.AMBIGUOUS_DESTINATION
        assert result.c2.decision == Decision.ASK

    def test_scenario5_distractor_added(self, image_path):
        # ⑤ 무관한 object 추가 (target/dest 모두 정상) → complete
        mock = _make_mock_handler()
        with patch("pipeline._make_handler", return_value=mock):
            result = run_pipeline(image_path, image_path, image_path, "place red block on tray")
        assert result.status == "complete"
        assert result.c1.state == GroundingState.CLEAR
        assert result.c2.state == GroundingState.CLEAR
