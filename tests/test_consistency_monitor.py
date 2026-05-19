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

from monitoring.consistency_monitor import (
    Decision,
    GroundingState,
    check_grounding,
    get_checkpoint_detections,
)

THRESHOLD = 50.0

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def g0():
    return {
        "target":      {"label": "red block", "coord": [200, 300]},
        "destination": {"label": "tray",      "coord": [400, 300]},
        "image_shape": [480, 640],
    }


@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "checkpoint.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


# ---------------------------------------------------------------------------
# TestGroundingStateTaxonomy — 각 state의 트리거 조건 검증
# ---------------------------------------------------------------------------

class TestC1TargetChecks:
    """C1 (pre-pick): target 관련 state 분류"""

    def test_clear_when_target_at_same_position(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_clear_within_threshold(self, g0):
        # 30px 이동 — threshold(50) 이내
        detections = {"red block": [[220, 315]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_invalid_target_when_label_missing(self, g0):
        detections = {"tray": [[400, 300]]}  # "red block" 없음
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.INVALID_TARGET

    def test_invalid_target_when_empty_detections(self, g0):
        state, _ = check_grounding(g0, {}, "C1", THRESHOLD)
        assert state == GroundingState.INVALID_TARGET

    def test_ambiguous_target_when_multiple_instances(self, g0):
        # 같은 label의 좌표가 여러 개 → 어느 것이 원래 target인지 모름
        detections = {"red block": [[200, 300], [350, 200]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_ambiguous_target_when_displaced_beyond_threshold(self, g0):
        # threshold 초과 이동 → 다른 instance로 판단
        detections = {"red block": [[400, 300]], "tray": [[400, 300]]}  # 200px 이동
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_boundary_exactly_at_threshold_is_clear(self, g0):
        # distance == threshold → CLEAR (> 이므로 같으면 CLEAR)
        detections = {"red block": [[200 + 50, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_boundary_just_over_threshold_is_ambiguous(self, g0):
        detections = {"red block": [[200 + 51, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_TARGET


class TestC2DestinationChecks:
    """C2 (pre-place): destination 관련 state 분류"""

    def test_clear_when_destination_at_same_position(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_clear_within_threshold(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[420, 310]]}
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_invalid_destination_when_label_missing(self, g0):
        detections = {"red block": [[200, 300]]}  # "tray" 없음
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.INVALID_DESTINATION

    def test_invalid_destination_when_empty_detections(self, g0):
        state, _ = check_grounding(g0, {}, "C2", THRESHOLD)
        assert state == GroundingState.INVALID_DESTINATION

    def test_ambiguous_destination_when_multiple_instances(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300], [100, 100]]}
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_DESTINATION

    def test_ambiguous_destination_when_displaced(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[100, 100]]}  # 340px 이동
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_DESTINATION

    def test_boundary_exactly_at_threshold_is_clear(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400 + 50, 300]]}
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_boundary_just_over_threshold_is_ambiguous(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400 + 51, 300]]}
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_DESTINATION


# ---------------------------------------------------------------------------
# TestDecisionPolicy — state → decision 매핑 검증
# ---------------------------------------------------------------------------

class TestDecisionPolicy:
    def test_clear_yields_continue(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300]]}
        _, decision = check_grounding(g0, detections, "C1", THRESHOLD)
        assert decision == Decision.CONTINUE

    def test_ambiguous_target_yields_ask(self, g0):
        detections = {"red block": [[200, 300], [350, 200]], "tray": [[400, 300]]}
        _, decision = check_grounding(g0, detections, "C1", THRESHOLD)
        assert decision == Decision.ASK

    def test_invalid_target_yields_stop(self, g0):
        _, decision = check_grounding(g0, {}, "C1", THRESHOLD)
        assert decision == Decision.STOP

    def test_ambiguous_destination_yields_ask(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300], [100, 100]]}
        _, decision = check_grounding(g0, detections, "C2", THRESHOLD)
        assert decision == Decision.ASK

    def test_invalid_destination_yields_ask(self, g0):
        # dest 소실은 사용자 입력으로 회복 가능 → ASK
        detections = {"red block": [[200, 300]]}
        _, decision = check_grounding(g0, detections, "C2", THRESHOLD)
        assert decision == Decision.ASK


# ---------------------------------------------------------------------------
# TestReturnTypes — 반환 타입 검증
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_returns_tuple_of_two(self, g0):
        result = check_grounding(g0, {"red block": [[200, 300]], "tray": [[400, 300]]}, "C1")
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_grounding_state(self, g0):
        state, _ = check_grounding(g0, {}, "C1")
        assert isinstance(state, GroundingState)

    def test_second_element_is_decision(self, g0):
        _, decision = check_grounding(g0, {}, "C1")
        assert isinstance(decision, Decision)

    def test_state_value_is_string(self, g0):
        state, _ = check_grounding(g0, {}, "C1")
        assert isinstance(state.value, str)


# ---------------------------------------------------------------------------
# TestInvalidCheckpoint — 잘못된 checkpoint 인자
# ---------------------------------------------------------------------------

class TestInvalidCheckpoint:
    def test_invalid_checkpoint_raises_value_error(self, g0):
        with pytest.raises(ValueError, match="checkpoint"):
            check_grounding(g0, {}, "C3")

    def test_lowercase_checkpoint_raises_value_error(self, g0):
        with pytest.raises(ValueError):
            check_grounding(g0, {}, "c1")

    def test_empty_checkpoint_raises_value_error(self, g0):
        with pytest.raises(ValueError):
            check_grounding(g0, {}, "")


# ---------------------------------------------------------------------------
# TestEdgeCases — 경계 케이스
# ---------------------------------------------------------------------------

class TestCameraMotionCompensation:
    """Camera-normalized relative-distance check (S5 distractor false-alarm fix).

    When the robot moves between t₀ and a checkpoint, the camera shifts too,
    displacing ALL pixel coordinates uniformly.  The monitor compensates by
    checking whether the (target − destination) relative vector is preserved
    when the destination also moved beyond the absolute threshold.
    """

    @pytest.fixture()
    def g0_scene(self):
        # Matches the real red-mug/sprite-bottle scene geometry
        return {
            "target":      {"label": "red mug",       "coord": [1364, 1060]},
            "destination": {"label": "sprite bottle",  "coord": [708,  1164]},
            "image_shape": [2252, 4000],
        }

    def test_s5_camera_moved_target_stationary_is_clear(self, g0_scene):
        # S5: camera moved ~450px; target is at the expected C1 position.
        # rel_dist ≈ 186px < threshold*5=250 → CLEAR
        detections = {
            "red mug":      [[941,  723]],   # moved 541px absolute (camera)
            "sprite bottle": [[466,  784]],   # moved 450px absolute (camera)
        }
        state, dec = check_grounding(g0_scene, detections, "C1", threshold=50.0)
        assert state == GroundingState.CLEAR
        assert dec   == Decision.CONTINUE

    def test_s6_camera_moved_target_also_moved_is_ambiguous(self, g0_scene):
        # S6: camera moved AND target moved to a very different world position.
        # rel_dist ≈ 1337px >> threshold*5=250 → AMBIGUOUS_TARGET
        detections = {
            "red mug":      [[2459, 662]],   # 1165px absolute (camera + genuine move)
            "sprite bottle": [[466,  784]],   # 450px absolute (camera only)
        }
        state, dec = check_grounding(g0_scene, detections, "C1", threshold=50.0)
        assert state == GroundingState.AMBIGUOUS_TARGET
        assert dec   == Decision.ASK

    def test_camera_stationary_target_moved_is_ambiguous(self, g0_scene):
        # Destination stays at G₀ → camera did not move → any target shift is genuine.
        detections = {
            "red mug":      [[1564, 1060]],  # 200px absolute, dest stays
            "sprite bottle": [[708,  1164]],  # 0px – camera stationary
        }
        state, _ = check_grounding(g0_scene, detections, "C1", threshold=50.0)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_no_destination_detected_treats_target_displacement_as_ambiguous(self, g0_scene):
        # No anchor → cannot compensate for camera motion → conservative AMBIGUOUS
        detections = {"red mug": [[941, 723]]}   # no sprite bottle
        state, _ = check_grounding(g0_scene, detections, "C1", threshold=50.0)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_custom_rel_threshold_overrides_default(self, g0_scene):
        # With rel_threshold=100 (< 186) S5 should still be AMBIGUOUS_TARGET
        detections = {
            "red mug":      [[941, 723]],
            "sprite bottle": [[466, 784]],
        }
        state, _ = check_grounding(g0_scene, detections, "C1",
                                   threshold=50.0, rel_threshold=100.0)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_c2_camera_moved_destination_stationary_is_clear(self, g0_scene):
        # C2: camera moved, destination is at expected position → CLEAR
        detections = {
            "red mug":      [[941,  723]],   # target also moved (camera)
            "sprite bottle": [[466,  784]],   # dest moved 450px (camera)
        }
        state, dec = check_grounding(g0_scene, detections, "C2", threshold=50.0)
        assert state == GroundingState.CLEAR
        assert dec   == Decision.CONTINUE


class TestEdgeCases:
    def test_same_label_for_target_and_destination(self):
        # target과 destination이 같은 label인 경우 (실제 모델 실패 케이스에서 발생)
        g0 = {
            "target":      {"label": "blue cup", "coord": [200, 200]},
            "destination": {"label": "blue cup", "coord": [200, 200]},
            "image_shape": [480, 640],
        }
        detections = {"blue cup": [[200, 200]]}
        state_c1, dec_c1 = check_grounding(g0, detections, "C1", THRESHOLD)
        state_c2, dec_c2 = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state_c1 == GroundingState.CLEAR
        assert state_c2 == GroundingState.CLEAR

    def test_c1_does_not_check_destination(self, g0):
        # C1에서 destination이 없어도 target이 있으면 CLEAR
        detections = {"red block": [[200, 300]]}  # tray 없음
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_c2_does_not_check_target(self, g0):
        # C2에서 target이 없어도 destination이 있으면 CLEAR
        detections = {"tray": [[400, 300]]}  # red block 없음
        state, _ = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_extra_labels_in_detections_are_ignored(self, g0):
        # 관련 없는 오브젝트가 추가돼도 (distractor) CLEAR
        detections = {
            "red block": [[200, 300]],
            "tray": [[400, 300]],
            "banana": [[100, 100]],  # distractor
        }
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR

    def test_threshold_zero_always_fails_unless_exact(self, g0):
        # threshold=0 → 1px 이동만 해도 AMBIGUOUS
        detections = {"red block": [[201, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", threshold=0.0)
        assert state == GroundingState.AMBIGUOUS_TARGET

    def test_threshold_zero_exact_match_is_clear(self, g0):
        detections = {"red block": [[200, 300]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", threshold=0.0)
        assert state == GroundingState.CLEAR

    def test_large_threshold_always_clear_when_single_instance(self, g0):
        # threshold 매우 크면 어떤 위치에 있어도 CLEAR (단일 instance)
        detections = {"red block": [[0, 0]], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", threshold=9999.0)
        assert state == GroundingState.CLEAR

    def test_coord_list_with_none_values_treated_as_missing(self, g0):
        # 빈 리스트 → INVALID
        detections = {"red block": [], "tray": [[400, 300]]}
        state, _ = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.INVALID_TARGET


# ---------------------------------------------------------------------------
# TestScenarioAlignment — proposal §5.1 시나리오 매핑
# ---------------------------------------------------------------------------

class TestScenarioAlignment:
    """proposal §5.1의 5개 시나리오가 올바른 state/decision으로 분류되는지 검증"""

    def test_scenario1_clear_continuation(self, g0):
        # ① 장면 변화 없음 → CLEAR → CONTINUE
        detections = {"red block": [[200, 300]], "tray": [[400, 300]]}
        state, decision = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.CLEAR
        assert decision == Decision.CONTINUE

    def test_scenario2_same_category_target_added(self, g0):
        # ② 같은 카테고리 target 추가 → AMBIGUOUS_TARGET → ASK
        detections = {"red block": [[200, 300], [500, 100]], "tray": [[400, 300]]}
        state, decision = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_TARGET
        assert decision == Decision.ASK

    def test_scenario3_target_disappeared(self, g0):
        # ③ target 사라짐 → INVALID_TARGET → STOP
        detections = {"tray": [[400, 300]]}
        state, decision = check_grounding(g0, detections, "C1", THRESHOLD)
        assert state == GroundingState.INVALID_TARGET
        assert decision == Decision.STOP

    def test_scenario4_new_destination_candidate(self, g0):
        # ④ 새 destination 후보 추가 → AMBIGUOUS_DESTINATION → ASK
        detections = {"red block": [[200, 300]], "tray": [[400, 300], [50, 50]]}
        state, decision = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state == GroundingState.AMBIGUOUS_DESTINATION
        assert decision == Decision.ASK

    def test_scenario5_distractor_added(self, g0):
        # ⑤ 무관한 물체 추가 → CLEAR → CONTINUE
        detections = {
            "red block": [[200, 300]],
            "tray": [[400, 300]],
            "irrelevant_object": [[50, 50]],
        }
        state_c1, dec_c1 = check_grounding(g0, detections, "C1", THRESHOLD)
        state_c2, dec_c2 = check_grounding(g0, detections, "C2", THRESHOLD)
        assert state_c1 == GroundingState.CLEAR and dec_c1 == Decision.CONTINUE
        assert state_c2 == GroundingState.CLEAR and dec_c2 == Decision.CONTINUE


# ---------------------------------------------------------------------------
# TestGetCheckpointDetections — get_checkpoint_detections 헬퍼 검증 (mock)
# ---------------------------------------------------------------------------

class TestGetCheckpointDetections:
    def test_calls_reset_then_detect(self, g0, image_path):
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = [
            {"success": True},  # reset
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        ]
        result = get_checkpoint_detections(image_path, g0, mock_handler, session_id="c1")
        calls = mock_handler.handle.call_args_list
        assert calls[0].args[0] == "reset"
        assert calls[1].args[0] == "detect"
        assert result == {"red block": [[200, 300]], "tray": [[400, 300]]}

    def test_detect_called_with_g0_labels(self, g0, image_path):
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = [
            {"success": True},
            {"success": True, "detections": {"red block": [[200, 300]], "tray": [[400, 300]]}},
        ]
        get_checkpoint_detections(image_path, g0, mock_handler)
        detect_call = mock_handler.handle.call_args_list[1]
        labels = detect_call.args[1]["objects"]
        assert set(labels) == {"red block", "tray"}

    def test_deduplicates_labels_when_target_eq_destination(self, image_path):
        g0_same = {
            "target":      {"label": "blue cup", "coord": [200, 200]},
            "destination": {"label": "blue cup", "coord": [200, 200]},
            "image_shape": [480, 640],
        }
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = [
            {"success": True},
            {"success": True, "detections": {"blue cup": [[200, 200]]}},
        ]
        get_checkpoint_detections(image_path, g0_same, mock_handler)
        detect_call = mock_handler.handle.call_args_list[1]
        labels = detect_call.args[1]["objects"]
        assert labels.count("blue cup") == 1  # 중복 제거

    def test_reset_failure_raises_runtime_error(self, g0, image_path):
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = [
            {"success": False, "error": "session error"},
        ]
        with pytest.raises(RuntimeError, match="reset"):
            get_checkpoint_detections(image_path, g0, mock_handler)

    def test_detect_failure_raises_runtime_error(self, g0, image_path):
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = [
            {"success": True},
            {"success": False, "error": "detect error"},
        ]
        with pytest.raises(RuntimeError, match="detect"):
            get_checkpoint_detections(image_path, g0, mock_handler)
