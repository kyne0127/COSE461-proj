"""Integration tests — require real AmbRes model and GPU.

Run with:  pytest tests/test_integration.py -m integration -v
Skip with: pytest tests/ -m "not integration"   (default unit-test run)

Design principles:
  - Pass `real_handler` fixture into every function that needs it — never
    call _make_handler() or pass model_type/adapter_ckpt inside a test,
    so the 7B-parameter model is loaded ONCE per session (scope="module").
  - Do NOT assert specific coord values (non-deterministic per run).
  - DO assert schema, types, and internal consistency.
  - Use allow_ambiguous=True so tests are not sensitive to the model's
    ambiguity judgement on a given image.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# conftest.py defines: real_handler, REAL_IMAGE_MARKER, REAL_IMAGE_BLOCK,
#                      TASK_MARKER, TASK_BLOCK
from conftest import (
    REAL_IMAGE_BLOCK,
    REAL_IMAGE_MARKER,
    TASK_BLOCK,
    TASK_MARKER,
)

from extraction.ambres_g0_extractor import extract_g0
from monitoring.consistency_monitor import Decision, GroundingState, check_grounding, get_checkpoint_detections
from utils.pilot_threshold import _collect_detections_real, build_report
from pipeline import PipelineResult, CheckpointOutcome, run_pipeline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assert_g0_schema(g0: dict, image_path: Path) -> None:
    assert set(g0.keys()) == {"target", "destination", "image_shape"}
    for role in ("target", "destination"):
        assert set(g0[role].keys()) == {"label", "coord"}
        assert isinstance(g0[role]["label"], str) and g0[role]["label"]
        coord = g0[role]["coord"]
        assert isinstance(coord, list) and len(coord) == 2
        assert all(isinstance(v, int) for v in coord), \
            f"{role} coord must be ints, got {[type(v).__name__ for v in coord]}"
    h, w = g0["image_shape"]
    assert isinstance(h, int) and isinstance(w, int)
    from PIL import Image
    img = Image.open(image_path)
    assert [img.height, img.width] == g0["image_shape"], \
        f"image_shape {g0['image_shape']} != actual [{img.height}, {img.width}]"


def _extract(image, task, handler, session_id):
    """Thin wrapper: always inject real_handler, never create a new one."""
    return extract_g0(
        image, task,
        handler=handler,
        session_id=session_id,
        allow_ambiguous=True,
    )


# ---------------------------------------------------------------------------
# TestExtractG0MarkerIntegration
# 5rhU25AdQW4jADxhp8EYuq.jpeg — "move the marker next to the sprite bottle"
# Known-good image: target=marker, destination=sprite bottle (verified manually)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_MARKER.exists(), reason="marker image not found")
class TestExtractG0MarkerIntegration:

    def test_output_schema(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_schema")
        _assert_g0_schema(g0, REAL_IMAGE_MARKER)

    def test_image_shape_is_2252_4000(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_shape")
        assert g0["image_shape"] == [2252, 4000]

    def test_labels_are_nonempty_strings(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_labels")
        assert g0["target"]["label"] and g0["destination"]["label"]

    def test_coords_within_image_bounds(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_bounds")
        h, w = g0["image_shape"]
        for role in ("target", "destination"):
            x, y = g0[role]["coord"]
            assert 0 <= x < w, f"{role} x={x} out of width {w}"
            assert 0 <= y < h, f"{role} y={y} out of height {h}"

    def test_coords_are_python_ints(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_ints")
        for role in ("target", "destination"):
            x, y = g0[role]["coord"]
            assert isinstance(x, int) and isinstance(y, int)

    def test_known_roles_marker_and_sprite(self, real_handler):
        """Sanity: model assigns 'marker'=target, 'sprite bottle'=destination."""
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_m_roles")
        assert g0["target"]["label"] == "marker"
        assert g0["destination"]["label"] == "sprite bottle"


# ---------------------------------------------------------------------------
# TestExtractG0BlockIntegration
# real_0.png — "place the blue cup on the table"
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_BLOCK.exists(), reason="real_0.png not found")
class TestExtractG0BlockIntegration:

    def test_output_schema(self, real_handler):
        g0 = _extract(REAL_IMAGE_BLOCK, TASK_BLOCK, real_handler, "integ_b_schema")
        _assert_g0_schema(g0, REAL_IMAGE_BLOCK)

    def test_image_shape_is_480_640(self, real_handler):
        g0 = _extract(REAL_IMAGE_BLOCK, TASK_BLOCK, real_handler, "integ_b_shape")
        assert g0["image_shape"] == [480, 640]

    def test_coords_are_python_ints(self, real_handler):
        g0 = _extract(REAL_IMAGE_BLOCK, TASK_BLOCK, real_handler, "integ_b_ints")
        for role in ("target", "destination"):
            x, y = g0[role]["coord"]
            assert isinstance(x, int) and isinstance(y, int)

    def test_coords_within_image_bounds(self, real_handler):
        g0 = _extract(REAL_IMAGE_BLOCK, TASK_BLOCK, real_handler, "integ_b_bounds")
        h, w = g0["image_shape"]
        for role in ("target", "destination"):
            x, y = g0[role]["coord"]
            assert 0 <= x < w and 0 <= y < h


# ---------------------------------------------------------------------------
# TestGetCheckpointDetectionsIntegration
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_MARKER.exists(), reason="marker image not found")
class TestGetCheckpointDetectionsIntegration:

    @pytest.fixture(scope="class")
    def g0_marker(self, real_handler):
        return _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_gcp_g0")

    def test_returns_dict_with_label_keys(self, real_handler, g0_marker):
        det = get_checkpoint_detections(
            REAL_IMAGE_MARKER, g0_marker, real_handler, session_id="integ_gcp_keys"
        )
        assert isinstance(det, dict)
        assert g0_marker["target"]["label"] in det or \
               g0_marker["destination"]["label"] in det, \
               f"Neither G0 label found in detections: {list(det.keys())}"

    def test_coord_format_is_list_of_pairs(self, real_handler, g0_marker):
        det = get_checkpoint_detections(
            REAL_IMAGE_MARKER, g0_marker, real_handler, session_id="integ_gcp_fmt"
        )
        for label, coords in det.items():
            assert isinstance(coords, list)
            for c in coords:
                assert isinstance(c, list) and len(c) == 2, \
                    f"{label}: coord must be [x,y], got {c!r}"

    def test_same_image_gives_consistent_coords(self, real_handler, g0_marker):
        """Molmo is near-deterministic: same image → coords within 5px."""
        d1 = get_checkpoint_detections(
            REAL_IMAGE_MARKER, g0_marker, real_handler, session_id="integ_cons_1"
        )
        d2 = get_checkpoint_detections(
            REAL_IMAGE_MARKER, g0_marker, real_handler, session_id="integ_cons_2"
        )
        for label in set(d1) & set(d2):
            if d1[label] and d2[label]:
                dx = d1[label][0][0] - d2[label][0][0]
                dy = d1[label][0][1] - d2[label][0][1]
                assert math.sqrt(dx**2 + dy**2) < 5, \
                    f"{label}: coord shifted {math.sqrt(dx**2+dy**2):.1f}px between two identical runs"


# ---------------------------------------------------------------------------
# TestEndToEndIntegration
# G0 (from t0) → detect same image (as checkpoint) → check_grounding → CLEAR
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_MARKER.exists(), reason="marker image not found")
class TestEndToEndIntegration:

    @pytest.fixture(scope="class")
    def g0_and_detections(self, real_handler):
        g0 = _extract(REAL_IMAGE_MARKER, TASK_MARKER, real_handler, "integ_e2e_g0")
        det = get_checkpoint_detections(
            REAL_IMAGE_MARKER, g0, real_handler, session_id="integ_e2e_det"
        )
        return g0, det

    def test_same_image_at_c1_is_clear_or_ambiguous(self, real_handler, g0_and_detections):
        """Same image at C1: CLEAR if single instance, AMBIGUOUS if model detects multiple.

        Molmo sometimes finds multiple "marker" regions in the same image.
        Both outcomes are correct pipeline behavior — this test verifies the
        state is at least logically consistent with the detection count.
        """
        g0, det = g0_and_detections
        target_label = g0["target"]["label"]
        state, decision = check_grounding(g0, det, "C1", threshold=30.0)
        n_instances = len(det.get(target_label) or [])
        if n_instances == 1:
            assert state == GroundingState.CLEAR, \
                f"Single instance → expected CLEAR, got {state}"
            assert decision == Decision.CONTINUE
        else:
            # Multiple detections → AMBIGUOUS is the correct response
            assert state == GroundingState.AMBIGUOUS_TARGET, \
                f"{n_instances} instances → expected AMBIGUOUS_TARGET, got {state}"
            assert decision == Decision.ASK

    def test_same_image_at_c2_is_clear(self, real_handler, g0_and_detections):
        g0, det = g0_and_detections
        state, decision = check_grounding(g0, det, "C2", threshold=30.0)
        assert state == GroundingState.CLEAR, \
            f"Same-image C2 should be CLEAR, got {state}\ndetections={det}"
        assert decision == Decision.CONTINUE

    def test_g0_coord_close_to_checkpoint_detection(self, real_handler, g0_and_detections):
        """G0 coord and checkpoint detection for same image must be < 30px apart."""
        g0, det = g0_and_detections
        target_label = g0["target"]["label"]
        if target_label not in det or not det[target_label]:
            pytest.skip(f"Target '{target_label}' not detected at checkpoint")
        g0_coord = g0["target"]["coord"]
        cp_coord  = det[target_label][0]
        dist = math.sqrt(
            (g0_coord[0] - cp_coord[0])**2 + (g0_coord[1] - cp_coord[1])**2
        )
        assert dist < 30, \
            f"G0 {g0_coord} vs checkpoint {cp_coord}: dist={dist:.1f}px (expected < 30)"


# ---------------------------------------------------------------------------
# TestPilotThresholdIntegration
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_MARKER.exists(), reason="marker image not found")
class TestPilotThresholdIntegration:

    def test_returns_coords_for_known_objects(self, real_handler):
        coords = _collect_detections_real(
            [REAL_IMAGE_MARKER], ["marker", "sprite bottle"], real_handler, n_trials=1
        )
        assert set(coords.keys()) == {"marker", "sprite bottle"}

    def test_coord_format_per_trial(self, real_handler):
        """_collect_detections_real stores raw detect output (float), not int-cast.

        Only extract_g0 (via _first_coord) converts to int. The pilot threshold
        intentionally keeps floats to preserve precision in variance statistics.
        """
        coords = _collect_detections_real(
            [REAL_IMAGE_MARKER], ["marker"], real_handler, n_trials=2
        )
        for c in coords["marker"]:
            assert isinstance(c, list) and len(c) == 2
            assert all(isinstance(v, (int, float)) for v in c)  # raw: float ok

    def test_build_report_with_real_data_and_print_threshold(self, real_handler):
        coords = _collect_detections_real(
            [REAL_IMAGE_MARKER, REAL_IMAGE_MARKER],
            ["marker", "sprite bottle"],
            real_handler, n_trials=1,
        )
        report = build_report(coords, safety_factor=3.0)
        assert report["recommended_threshold_px"] >= 0
        assert isinstance(report["recommended_threshold_px"], int)
        # Print for manual inspection — helps set the real default threshold
        print(f"\n[pilot] recommended_threshold_px = {report['recommended_threshold_px']}")
        print(f"[pilot] intra_scene = {report['intra_scene']}")


# ---------------------------------------------------------------------------
# TestPipelineIntegration — t₀ → C1 → C2 전체 흐름을 실제 모델로 검증
#
# 같은 이미지를 t0/C1/C2 모두에 사용 → 장면 변화 없음 → 기대 동작:
#   - status: "complete" (STOP 없음)
#   - C1: CLEAR or AMBIGUOUS_TARGET (Molmo가 marker를 여러 개 감지할 수 있음)
#   - C2: CLEAR or AMBIGUOUS_DESTINATION
#   - G₀ 스키마·타입 정상
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not REAL_IMAGE_MARKER.exists(), reason="marker image not found")
class TestPipelineIntegration:
    """run_pipeline()을 실제 모델로 실행 — t₀→C1→C2 전체 흐름 검증."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, real_handler):
        """단일 이미지(marker)를 t0/C1/C2 모두에 사용, 장면 변화 없음."""
        return run_pipeline(
            REAL_IMAGE_MARKER,
            REAL_IMAGE_MARKER,
            REAL_IMAGE_MARKER,
            TASK_MARKER,
            handler=real_handler,
            threshold=30.0,
            allow_ambiguous=True,
            session_prefix="integ_pipe",
        )

    # ── 상태 검증 ──────────────────────────────────────────────────────────

    def test_status_is_complete(self, real_handler, pipeline_result):
        """같은 이미지 → 장면 변화 없음 → STOP 없이 complete."""
        assert pipeline_result.status == "complete", (
            f"Expected 'complete', got '{pipeline_result.status}': "
            f"{pipeline_result.stop_reason}"
        )

    def test_g0_initial_is_valid(self, real_handler, pipeline_result):
        assert pipeline_result.g0_initial is not None
        assert set(pipeline_result.g0_initial.keys()) == {"target", "destination", "image_shape"}

    def test_g0_initial_image_shape(self, real_handler, pipeline_result):
        assert pipeline_result.g0_initial["image_shape"] == [2252, 4000]

    def test_g0_initial_labels_are_correct(self, real_handler, pipeline_result):
        """모델이 marker=target, sprite bottle=destination으로 분류."""
        assert pipeline_result.g0_initial["target"]["label"] == "marker"
        assert pipeline_result.g0_initial["destination"]["label"] == "sprite bottle"

    def test_g0_initial_coords_are_ints(self, real_handler, pipeline_result):
        for role in ("target", "destination"):
            x, y = pipeline_result.g0_initial[role]["coord"]
            assert isinstance(x, int) and isinstance(y, int)

    # ── C1 검증 ────────────────────────────────────────────────────────────

    def test_c1_is_not_none(self, real_handler, pipeline_result):
        assert pipeline_result.c1 is not None

    def test_c1_checkpoint_label(self, real_handler, pipeline_result):
        assert pipeline_result.c1.checkpoint == "C1"

    def test_c1_state_is_clear_or_ambiguous(self, real_handler, pipeline_result):
        """같은 이미지 → CLEAR. Molmo가 여러 marker 감지 시 AMBIGUOUS_TARGET도 허용."""
        assert pipeline_result.c1.state in (
            GroundingState.CLEAR,
            GroundingState.AMBIGUOUS_TARGET,
        ), f"Unexpected C1 state: {pipeline_result.c1.state}"

    def test_c1_decision_matches_state(self, real_handler, pipeline_result):
        state = pipeline_result.c1.state
        decision = pipeline_result.c1.decision
        if state == GroundingState.CLEAR:
            assert decision == Decision.CONTINUE
        else:
            assert decision == Decision.ASK

    def test_c1_detections_contain_target_label(self, real_handler, pipeline_result):
        target_label = pipeline_result.g0_initial["target"]["label"]
        assert target_label in pipeline_result.c1.detections, (
            f"Target '{target_label}' not found in C1 detections: "
            f"{list(pipeline_result.c1.detections.keys())}"
        )

    def test_c1_g0_before_matches_initial(self, real_handler, pipeline_result):
        assert pipeline_result.c1.g0_before == pipeline_result.g0_initial

    def test_c1_no_user_response(self, real_handler, pipeline_result):
        """noop user_response_fn → user_response == ""."""
        assert pipeline_result.c1.user_response == ""

    def test_c1_g0_before_equals_after_when_noop(self, real_handler, pipeline_result):
        """빈 응답이므로 rolling update 없음 → g0_before == g0_after."""
        assert pipeline_result.c1.g0_before == pipeline_result.c1.g0_after

    # ── C2 검증 ────────────────────────────────────────────────────────────

    def test_c2_is_not_none(self, real_handler, pipeline_result):
        assert pipeline_result.c2 is not None

    def test_c2_checkpoint_label(self, real_handler, pipeline_result):
        assert pipeline_result.c2.checkpoint == "C2"

    def test_c2_state_is_clear_or_ambiguous(self, real_handler, pipeline_result):
        """destination(sprite bottle) 체크 → CLEAR or AMBIGUOUS_DESTINATION."""
        assert pipeline_result.c2.state in (
            GroundingState.CLEAR,
            GroundingState.AMBIGUOUS_DESTINATION,
        ), f"Unexpected C2 state: {pipeline_result.c2.state}"

    def test_c2_decision_matches_state(self, real_handler, pipeline_result):
        state = pipeline_result.c2.state
        decision = pipeline_result.c2.decision
        if state == GroundingState.CLEAR:
            assert decision == Decision.CONTINUE
        else:
            assert decision == Decision.ASK

    def test_c2_g0_before_is_c1_g0_after(self, real_handler, pipeline_result):
        """C2는 C1 이후 G₀를 이어받아야 함 (rolling update propagation)."""
        assert pipeline_result.c2.g0_before == pipeline_result.c1.g0_after

    # ── 출력 직렬화 검증 ────────────────────────────────────────────────────

    def test_to_dict_is_json_serialisable(self, real_handler, pipeline_result):
        import json
        d = pipeline_result.to_dict()
        json.dumps(d)  # should not raise

    def test_to_dict_status_is_complete(self, real_handler, pipeline_result):
        assert pipeline_result.to_dict()["status"] == "complete"

    def test_print_summary(self, real_handler, pipeline_result):
        """실행 결과 출력 — 수동 검토용."""
        import json
        print("\n[pipeline] full result:")
        print(json.dumps(pipeline_result.to_dict(), ensure_ascii=False, indent=2))
