from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.pilot_threshold import (
    build_report,
    centroid,
    compute_inter_instance_distances,
    compute_intra_stats,
    euclidean,
    percentile,
    recommend_threshold,
    _collect_detections_mock,
    _collect_detections_real,
)


# ---------------------------------------------------------------------------
# TestCentroid
# ---------------------------------------------------------------------------

class TestCentroid:
    def test_single_coord(self):
        assert centroid([[100, 200]]) == [100.0, 200.0]

    def test_two_coords(self):
        assert centroid([[0, 0], [10, 10]]) == [5.0, 5.0]

    def test_symmetric_coords(self):
        c = centroid([[100, 300], [200, 300], [150, 300]])
        assert c == [150.0, 300.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            centroid([])


# ---------------------------------------------------------------------------
# TestEuclidean
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_same_point_is_zero(self):
        assert euclidean([0, 0], [0, 0]) == 0.0

    def test_horizontal_distance(self):
        assert euclidean([0, 0], [3, 0]) == pytest.approx(3.0)

    def test_vertical_distance(self):
        assert euclidean([0, 0], [0, 4]) == pytest.approx(4.0)

    def test_diagonal_345(self):
        assert euclidean([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_symmetric(self):
        assert euclidean([10, 20], [30, 40]) == euclidean([30, 40], [10, 20])


# ---------------------------------------------------------------------------
# TestPercentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_50th_of_sorted_list(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        # nearest-rank: ceil(50/100 * 5) - 1 = 2 → vals[2] = 3.0
        assert percentile(vals, 50) == 3.0

    def test_100th_is_max(self):
        vals = [1.0, 5.0, 3.0]
        assert percentile(vals, 100) == 5.0

    def test_single_element(self):
        assert percentile([42.0], 95) == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 95)

    def test_95th_with_known_data(self):
        # 10 values 0..9: ceil(95/100*10)-1 = 9 → vals[9] = 9.0
        vals = list(range(10))
        assert percentile(vals, 95) == 9.0

    def test_order_independent(self):
        vals = [5.0, 1.0, 3.0, 2.0, 4.0]
        shuffled = [3.0, 5.0, 1.0, 4.0, 2.0]
        assert percentile(vals, 95) == percentile(shuffled, 95)


# ---------------------------------------------------------------------------
# TestComputeIntraStats
# ---------------------------------------------------------------------------

class TestComputeIntraStats:
    def test_single_object_single_coord(self):
        stats = compute_intra_stats({"cup": [[100, 200]]})
        assert stats["cup"]["n"] == 1
        assert stats["cup"]["centroid"] == [100.0, 200.0]
        assert stats["cup"]["distances"] == [0.0]
        assert stats["cup"]["p95"] == 0.0

    def test_no_variation_gives_zero_spread(self):
        # same coord repeated → zero distances
        coords = [[150, 300]] * 5
        stats = compute_intra_stats({"box": coords})
        assert stats["box"]["max"] == 0.0
        assert stats["box"]["p95"] == 0.0

    def test_known_variation(self):
        # centroid at [0,0], coords at ±3 on x → distances all 3.0
        coords = [[3, 0], [-3, 0], [0, 3], [0, -3]]
        stats = compute_intra_stats({"obj": coords})
        assert stats["obj"]["centroid"] == [0.0, 0.0]
        assert all(abs(d - 3.0) < 1e-9 for d in stats["obj"]["distances"])
        assert stats["obj"]["p95"] == pytest.approx(3.0)

    def test_multiple_labels(self):
        data = {
            "cup": [[100, 200], [102, 198]],
            "box": [[300, 400], [305, 395]],
        }
        stats = compute_intra_stats(data)
        assert "cup" in stats and "box" in stats

    def test_empty_coords_skipped(self):
        stats = compute_intra_stats({"ghost": []})
        assert "ghost" not in stats

    def test_custom_percentile(self):
        coords = [[i, 0] for i in range(10)]  # distances 0..4.5 from centroid
        stats = compute_intra_stats({"line": coords}, p=50)
        assert "p50" in stats["line"]


# ---------------------------------------------------------------------------
# TestComputeInterInstanceDistances
# ---------------------------------------------------------------------------

class TestComputeInterInstanceDistances:
    def test_single_instance_returns_none(self):
        result = compute_inter_instance_distances({"cup": [[100, 200]]})
        assert result["cup"] is None

    def test_two_instances(self):
        result = compute_inter_instance_distances({"cup": [[0, 0], [30, 40]]})
        assert result["cup"] == pytest.approx(50.0)

    def test_three_instances_min_distance(self):
        # distances: (0,0)↔(10,0)=10, (0,0)↔(100,0)=100, (10,0)↔(100,0)=90
        result = compute_inter_instance_distances({"obj": [[0, 0], [10, 0], [100, 0]]})
        assert result["obj"] == pytest.approx(10.0)

    def test_empty_coords(self):
        result = compute_inter_instance_distances({"ghost": []})
        assert result["ghost"] is None

    def test_multiple_labels(self):
        data = {
            "cup": [[0, 0], [50, 0]],
            "box": [[100, 100]],
        }
        result = compute_inter_instance_distances(data)
        assert result["cup"] == pytest.approx(50.0)
        assert result["box"] is None


# ---------------------------------------------------------------------------
# TestRecommendThreshold
# ---------------------------------------------------------------------------

class TestRecommendThreshold:
    def _make_stats(self, p95: float) -> dict:
        return {"obj": {"n": 5, "centroid": [0, 0], "distances": [], "p95": p95, "max": p95}}

    def test_single_label(self):
        stats = self._make_stats(5.0)
        assert recommend_threshold(stats, safety_factor=3.0) == 15

    def test_takes_max_across_labels(self):
        stats = {
            "cup": {"p95": 3.0, "n": 5, "centroid": [0, 0], "distances": [], "max": 3.0},
            "box": {"p95": 7.0, "n": 5, "centroid": [0, 0], "distances": [], "max": 7.0},
        }
        assert recommend_threshold(stats, safety_factor=2.0) == 14  # 7.0 × 2 = 14

    def test_rounds_to_integer(self):
        stats = self._make_stats(3.3)
        result = recommend_threshold(stats, safety_factor=3.0)
        assert isinstance(result, int)

    def test_empty_stats_raises(self):
        with pytest.raises(ValueError):
            recommend_threshold({})

    def test_missing_p_key_raises(self):
        with pytest.raises(ValueError):
            recommend_threshold({"obj": {"n": 1}}, p_key="p95")

    def test_zero_noise_gives_zero_threshold(self):
        stats = self._make_stats(0.0)
        assert recommend_threshold(stats, safety_factor=3.0) == 0

    def test_custom_safety_factor(self):
        stats = self._make_stats(10.0)
        assert recommend_threshold(stats, safety_factor=5.0) == 50


# ---------------------------------------------------------------------------
# TestBuildReport
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_output_keys(self):
        coords = {"cup": [[100, 200], [101, 199], [102, 201]]}
        report = build_report(coords)
        assert set(report.keys()) >= {
            "intra_scene", "inter_instance_min_dist",
            "safety_factor", "recommended_threshold_px", "note",
        }

    def test_threshold_is_int(self):
        coords = {"cup": [[100, 200], [103, 203]]}
        report = build_report(coords)
        assert isinstance(report["recommended_threshold_px"], int)

    def test_note_contains_threshold(self):
        coords = {"cup": [[100, 200], [100, 200]]}
        report = build_report(coords)
        assert str(report["recommended_threshold_px"]) in report["note"]

    def test_custom_safety_factor(self):
        # zero noise → threshold = 0 × any factor = 0
        coords = {"cup": [[100, 200]] * 5}
        report = build_report(coords, safety_factor=10.0)
        assert report["recommended_threshold_px"] == 0
        assert report["safety_factor"] == 10.0


# ---------------------------------------------------------------------------
# TestCollectDetectionsMock
# ---------------------------------------------------------------------------

class TestCollectDetectionsMock:
    def test_returns_coords_for_each_object(self):
        coords = _collect_detections_mock(["cup", "box"], n_trials=5, seed=0)
        assert set(coords.keys()) == {"cup", "box"}

    def test_n_trials_determines_count(self):
        coords = _collect_detections_mock(["cup"], n_trials=7, seed=0)
        assert len(coords["cup"]) == 7

    def test_coords_are_two_element_lists(self):
        coords = _collect_detections_mock(["obj"], n_trials=3, seed=0)
        for c in coords["obj"]:
            assert isinstance(c, list) and len(c) == 2

    def test_seed_reproducibility(self):
        a = _collect_detections_mock(["cup"], n_trials=5, seed=42)
        b = _collect_detections_mock(["cup"], n_trials=5, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = _collect_detections_mock(["cup"], n_trials=5, seed=1)
        b = _collect_detections_mock(["cup"], n_trials=5, seed=2)
        assert a != b

    def test_noise_zero_gives_same_coord_each_trial(self):
        coords = _collect_detections_mock(["cup"], n_trials=5, noise_std=0.0, seed=0)
        first = coords["cup"][0]
        assert all(c == first for c in coords["cup"])

    def test_larger_noise_gives_larger_spread(self):
        low  = _collect_detections_mock(["cup"], n_trials=50, noise_std=1.0,  seed=0)
        high = _collect_detections_mock(["cup"], n_trials=50, noise_std=20.0, seed=0)
        c_low  = centroid(low["cup"])
        c_high = centroid(high["cup"])
        spread_low  = max(euclidean(c, c_low)  for c in low["cup"])
        spread_high = max(euclidean(c, c_high) for c in high["cup"])
        assert spread_high > spread_low


# ---------------------------------------------------------------------------
# TestCollectDetectionsReal — mock handler
# ---------------------------------------------------------------------------

class TestCollectDetectionsReal:
    @pytest.fixture()
    def image_path(self, tmp_path):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        p = tmp_path / "img.png"
        Image.fromarray(arr, "RGB").save(str(p))
        return p

    def _make_handler(self, coord):
        mock = MagicMock()
        mock.handle.side_effect = lambda method, payload, tensors, session_id: (
            {"success": True} if method == "reset"
            else {"success": True, "detections": {"cup": [coord]}}
        )
        return mock

    def test_single_image_repeated_n_trials(self, image_path):
        handler = self._make_handler([150, 250])
        coords = _collect_detections_real([image_path], ["cup"], handler, n_trials=3)
        assert len(coords["cup"]) == 3

    def test_multiple_images_detected_once_each(self, image_path):
        handler = self._make_handler([150, 250])
        coords = _collect_detections_real(
            [image_path, image_path, image_path], ["cup"], handler, n_trials=99
        )
        # multiple images → once per image (n_trials ignored)
        assert len(coords["cup"]) == 3

    def test_reset_called_before_each_detect(self, image_path):
        handler = self._make_handler([100, 200])
        _collect_detections_real([image_path], ["cup"], handler, n_trials=2)
        calls = [c.args[0] for c in handler.handle.call_args_list]
        # pattern: reset, detect, reset, detect
        assert calls == ["reset", "detect", "reset", "detect"]

    def test_missing_detection_skipped(self, image_path):
        mock = MagicMock()
        mock.handle.side_effect = lambda method, payload, tensors, session_id: (
            {"success": True} if method == "reset"
            else {"success": True, "detections": {}}  # object not detected
        )
        coords = _collect_detections_real([image_path], ["cup"], mock, n_trials=3)
        assert coords["cup"] == []


# ---------------------------------------------------------------------------
# TestEndToEndMock — full pipeline with mock data
# ---------------------------------------------------------------------------

class TestEndToEndMock:
    def test_mock_pipeline_produces_valid_report(self):
        coords = _collect_detections_mock(["red block", "tray"], n_trials=10, seed=7)
        report = build_report(coords, safety_factor=3.0)
        assert report["recommended_threshold_px"] >= 0
        assert isinstance(report["recommended_threshold_px"], int)

    def test_report_json_serialisable(self):
        coords = _collect_detections_mock(["cup"], n_trials=5, seed=0)
        report = build_report(coords)
        # should not raise
        json.dumps(report)

    def test_low_noise_gives_low_threshold(self):
        coords = _collect_detections_mock(["cup"], n_trials=20, noise_std=1.0, seed=0)
        report = build_report(coords, safety_factor=3.0)
        # 1px noise × 3 safety → threshold should be in single digits
        assert report["recommended_threshold_px"] < 20

    def test_high_noise_gives_higher_threshold(self):
        low_coords  = _collect_detections_mock(["cup"], n_trials=30, noise_std=2.0,  seed=0)
        high_coords = _collect_detections_mock(["cup"], n_trials=30, noise_std=20.0, seed=0)
        low_report  = build_report(low_coords)
        high_report = build_report(high_coords)
        assert high_report["recommended_threshold_px"] > low_report["recommended_threshold_px"]
