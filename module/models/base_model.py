"""
module.models.base_model
========================
Abstract base class for all LeRobot plug-and-play model modules.

Every model must implement:
    - load_checkpoint(path)
    - predict_action(observation) → np.ndarray
    - train_step(batch) → dict[str, float]   (loss dict)

Optional overrides:
    - reset()          called at episode start
    - save_checkpoint(path)
    - get_config() → dict
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Observation:
    """
    Unified observation container passed to model.predict_action().
    Decoupled from gRPC proto so models don't depend on protobuf.
    """

    def __init__(
        self,
        images:     Optional[Dict[str, np.ndarray]] = None,
        state:      Optional[np.ndarray] = None,
        task_text:  str = "",
        episode_id: int = 0,
        step:       int = 0,
        extra:      Optional[Dict[str, Any]] = None,
    ) -> None:
        self.images     = images or {}          # cam_name → (H, W, C) float32 [0,1]
        self.state      = state                  # (state_dim,) float32
        self.task_text  = task_text              # for VLA models
        self.episode_id = episode_id
        self.step       = step
        self.extra      = extra or {}

    @property
    def image_list(self) -> List[np.ndarray]:
        return list(self.images.values())

    def __repr__(self) -> str:
        return (
            f"Observation(cameras={list(self.images.keys())}, "
            f"state_shape={self.state.shape if self.state is not None else None}, "
            f"step={self.step})"
        )


class TrainingBatch:
    """
    Unified training batch container.
    The training service converts HuggingFace datasets into this format.
    """

    def __init__(
        self,
        images:  Optional[Dict[str, Any]] = None,  # cam → (B, T, H, W, C)
        states:  Optional[Any] = None,              # (B, T, state_dim)
        actions: Optional[Any] = None,              # (B, T, action_dim)
        rewards: Optional[Any] = None,              # (B, T)
        dones:   Optional[Any] = None,              # (B, T)
        extra:   Optional[Dict[str, Any]] = None,
    ) -> None:
        self.images  = images or {}
        self.states  = states
        self.actions = actions
        self.rewards = rewards
        self.dones   = dones
        self.extra   = extra or {}


# ────────────────────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────────────────────

class BaseLeRobotModel(ABC):
    """
    Base class for all pluggable LeRobot model backends.

    Subclasses should:
      1. Decorate with @ModelRegistry.register("model_type")
      2. Call super().__init__(config) in __init__
      3. Implement the three abstract methods
    """

    #: Model type string — must match the registry key
    MODEL_TYPE: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config     = config
        self._loaded    = False
        self._device    = config.get("device", "cuda")
        self.logger     = logging.getLogger(f"lerobot.model.{self.MODEL_TYPE}")
        self.logger.info("Initializing %s (device=%s)", self.MODEL_TYPE, self._device)

    # ------------------------------------------------------------------ #
    # Required interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from checkpoint_path."""
        ...

    @abstractmethod
    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        Given an Observation, return the predicted action as float32 numpy array.
        Shape: (action_dim,) for single-step models,
               (chunk_size, action_dim) for chunk-based models (e.g. ACT).
        """
        ...

    @abstractmethod
    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute one gradient update step.
        Returns a dict of loss names → scalar values (for logging).
        """
        ...

    # ------------------------------------------------------------------ #
    # Optional overrides
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Called at the start of each new episode. Override for stateful models."""
        pass

    def save_checkpoint(self, save_path: str | Path) -> None:
        """Save model weights. Override if the model has custom save logic."""
        raise NotImplementedError(f"{self.MODEL_TYPE} does not implement save_checkpoint()")

    def get_config(self) -> Dict[str, Any]:
        """Return current effective config dict."""
        return dict(self.config)

    def get_num_params(self) -> int:
        """Return total parameter count (if torch model)."""
        try:
            return sum(p.numel() for p in self._model.parameters())  # type: ignore[attr-defined]
        except AttributeError:
            return -1

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _mark_loaded(self) -> None:
        self._loaded = True
        self.logger.info("%s checkpoint loaded successfully", self.MODEL_TYPE)

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n = self.get_num_params()
        param_str = f"{n:,}" if n >= 0 else "N/A"
        return (
            f"{self.__class__.__name__}("
            f"type={self.MODEL_TYPE}, "
            f"loaded={self._loaded}, "
            f"device={self._device}, "
            f"params={param_str})"
        )
