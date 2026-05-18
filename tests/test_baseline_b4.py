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

from baselines.b4_binary_anomaly import METHOD_NAME, run_b4_binary_anomaly
from monitoring.consistency_monitor import Decision


SESSION = "test_b4"


@pytest.fixture()
def image_path(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "checkpoint.png"
    Image.fromarray(arr, "RGB").save(str(p))
    return p


@pytest.fixture()
def g0():
    return {
        "target": {"label": "red block", "coord": [200, 300]},
        "destination": {"label": "gray tray", "coord": [400, 300]},
        "image_shape": [480, 640],
    }


@pytest.fixture()
def mock_handler():
    return MagicMock()


def _patch_detections(detections):
    return patch(
        "baselines.b4_binary_anomaly.get_checkpoint_detections",
        return_value=detections,
    )


class TestB4DecisionPolicy:
    def test_all_roles_valid_returns_continue(self, image_path, g0, mock_handler):
        detections = {"red block": [[200, 300]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.method == METHOD_NAME
        assert result.decision == Decision.CONTINUE
        assert result.question == ""
        assert result.metadata["uses_taxonomy"] is False
        assert result.metadata["target_valid"] is True
        assert result.metadata["destination_valid"] is True

    def test_missing_target_returns_ask_not_stop(self, image_path, g0, mock_handler):
        detections = {"red block": [], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert "not found" in result.reason
        assert result.metadata["target_valid"] is False

    def test_missing_destination_returns_ask(self, image_path, g0, mock_handler):
        detections = {"red block": [[200, 300]], "gray tray": []}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert result.metadata["destination_valid"] is False

    def test_multiple_target_candidates_returns_ask(self, image_path, g0, mock_handler):
        detections = {"red block": [[200, 300], [350, 100]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert result.metadata["target_detail"]["n_coords"] == 2

    def test_multiple_destination_candidates_returns_ask(self, image_path, g0, mock_handler):
        detections = {"red block": [[200, 300]], "gray tray": [[400, 300], [500, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert result.metadata["destination_detail"]["n_coords"] == 2

    def test_displaced_target_returns_ask(self, image_path, g0, mock_handler):
        detections = {"red block": [[999, 300]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                threshold=50,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert result.metadata["target_detail"]["min_distance"] > 50


class TestB4Threshold:
    def test_boundary_exactly_at_threshold_is_continue(self, image_path, g0, mock_handler):
        detections = {"red block": [[250, 300]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                threshold=50,
                session_id=SESSION,
            )

        assert result.decision == Decision.CONTINUE

    def test_boundary_just_over_threshold_is_ask(self, image_path, g0, mock_handler):
        detections = {"red block": [[251, 300]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                threshold=50,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK

    def test_invalid_empty_coord_lists_are_treated_as_missing(self, image_path, g0, mock_handler):
        detections = {"red block": [[]], "gray tray": [[400, 300]]}
        with _patch_detections(detections):
            result = run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                session_id=SESSION,
            )

        assert result.decision == Decision.ASK
        assert result.metadata["target_detail"]["n_coords"] == 0


class TestB4DetectionCall:
    def test_uses_get_checkpoint_detections_with_handler(self, image_path, g0, mock_handler):
        detections = {"red block": [[200, 300]], "gray tray": [[400, 300]]}
        with _patch_detections(detections) as get_det:
            run_b4_binary_anomaly(
                g0,
                image_path,
                checkpoint="C2",
                handler=mock_handler,
                session_id=SESSION,
            )

        get_det.assert_called_once_with(
            image_path,
            g0,
            mock_handler,
            session_id=SESSION,
        )


class TestB4ErrorPaths:
    def test_invalid_checkpoint_raises_value_error(self, image_path, g0, mock_handler):
        with pytest.raises(ValueError, match="checkpoint"):
            run_b4_binary_anomaly(
                g0,
                image_path,
                checkpoint="C3",
                handler=mock_handler,
                session_id=SESSION,
            )

    def test_handler_is_required(self, image_path, g0):
        with pytest.raises(ValueError, match="handler"):
            run_b4_binary_anomaly(g0, image_path, handler=None, session_id=SESSION)

    def test_negative_threshold_raises_value_error(self, image_path, g0, mock_handler):
        with pytest.raises(ValueError, match="threshold"):
            run_b4_binary_anomaly(
                g0,
                image_path,
                handler=mock_handler,
                threshold=-1,
                session_id=SESSION,
            )
