"""
module.models.camera_probe.model
================================
Camera-aware probe model for E2E validation.

This model is intentionally simple and deterministic:
- Uses image statistics so camera changes affect output.
- Produces full-state action vectors for direct follower control tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry


@ModelRegistry.register("camera_probe")
class CameraProbeModel(BaseLeRobotModel):
    MODEL_TYPE = "camera_probe"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._action_dim = int(config.get("action_dim", 12))
        self._gain = float(config.get("gain", 0.4))

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        # Probe model has no learnable checkpoint; mark as loaded.
        _ = checkpoint_path
        self._mark_loaded()

    def predict_action(self, observation: Observation) -> np.ndarray:
        state = observation.state
        if state is None:
            state = np.zeros(self._action_dim, dtype=np.float32)

        state = state.astype(np.float32).reshape(-1)
        if state.shape[0] != self._action_dim:
            if state.shape[0] < self._action_dim:
                pad = np.zeros(self._action_dim - state.shape[0], dtype=np.float32)
                state = np.concatenate([state, pad], axis=0)
            else:
                state = state[: self._action_dim]

        img_means: list[float] = []
        for _, img in sorted(observation.images.items()):
            arr = img.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            img_means.append(float(arr.mean()))

        mean_intensity = float(np.mean(img_means)) if img_means else 0.0
        idx = np.arange(self._action_dim, dtype=np.float32)
        visual_wave = np.sin((idx + 1.0) * (0.5 + mean_intensity * np.pi)).astype(np.float32)

        # Keep command bounded and visibly camera-dependent.
        action = 0.15 * np.tanh(state) + self._gain * visual_wave
        return action.astype(np.float32)

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        _ = batch
        return {"loss": 0.0}
