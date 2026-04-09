"""
module.models.custom.model
==========================
Custom Model Template — plug-in boilerplate.

Copy this file, rename the class, implement the three abstract methods,
and register with @ModelRegistry.register("your_model_name").

The new model will automatically be discoverable by the server
without any other configuration changes.

Example usage:
    # 1. Copy this file to module/models/mymodel/model.py
    # 2. Rename class to MyModel
    # 3. Change MODEL_TYPE = "mymodel"
    # 4. Implement load_checkpoint, predict_action, train_step
    # 5. The server auto-discovers it via ModelRegistry.auto_discover()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("custom")
class CustomModel(BaseLeRobotModel):
    """
    ⚠️  This is a template. Replace the implementation with your own model.

    Config keys (example):
        device:         "cuda" | "cpu"
        action_dim:     int    output action dimension
        state_dim:      int    input state dimension
        hidden_dim:     int    hidden layer size (default: 256)
        lr:             float  learning rate (default: 1e-4)
    """

    MODEL_TYPE = "custom"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._model     = None
        self._optimizer = None
        self._action_dim = config.get("action_dim", 7)
        self._state_dim  = config.get("state_dim", 14)
        self._hidden_dim = config.get("hidden_dim", 256)

    # ------------------------------------------------------------------ #
    # ① Load checkpoint
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load your model from checkpoint_path.
        Replace the example MLP with your actual architecture.
        """
        checkpoint_path = Path(checkpoint_path)
        self.logger.info("Loading CustomModel from %s", checkpoint_path)

        try:
            import torch
            import torch.nn as nn

            # ── Example: simple MLP policy ──────────────────────────────
            class SimpleMLP(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim),
                        nn.Tanh(),
                    )

                def forward(self, x):
                    return self.net(x)
            # ─────────────────────────────────────────────────────────────

            self._model = SimpleMLP(
                self._state_dim, self._action_dim, self._hidden_dim
            )

            weights_path = checkpoint_path / "model.pt"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                self.logger.info("Loaded weights from %s", weights_path)
            else:
                self.logger.warning(
                    "No checkpoint found at %s — using random init", weights_path
                )

            self._model.to(self._device)
            self._model.eval()
            self._mark_loaded()

        except ImportError as e:
            raise ImportError(f"CustomModel requires torch.\n{e}") from e

    # ------------------------------------------------------------------ #
    # ② Predict action
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        Given an Observation, return the predicted action.
        Modify this to match your model's input/output format.

        Returns: (action_dim,) float32 numpy array
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        import torch

        # Example: use only proprioceptive state
        if observation.state is None:
            state = np.zeros(self._state_dim, dtype=np.float32)
        else:
            state = observation.state.astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(self._device)
            action = self._model(x).squeeze(0).cpu().numpy()

        return action.astype(np.float32)

    # ------------------------------------------------------------------ #
    # ③ Training step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        One gradient update step.
        Replace with your actual loss computation.

        Returns: dict of loss_name → float value
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Model not loaded.")
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self.config.get("lr", 1e-4),
            )

        self._model.train()
        self._optimizer.zero_grad()

        # ── Example: BC (behavioural cloning) MSE loss ───────────────────
        states  = torch.tensor(batch.states,  dtype=torch.float32).to(self._device)
        actions = torch.tensor(batch.actions, dtype=torch.float32).to(self._device)

        # Flatten time dim if needed: (B, T, D) → (B*T, D)
        if states.dim() == 3:
            B, T, D = states.shape
            states  = states.reshape(B * T, D)
            actions = actions.reshape(B * T, -1)

        pred = self._model(states)
        loss = F.mse_loss(pred, actions)
        # ─────────────────────────────────────────────────────────────────

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        self._model.eval()
        return {"loss": float(loss.item())}

    # ------------------------------------------------------------------ #
    # Optional overrides
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Called at episode start. Clear any recurrent state here."""
        pass

    def save_checkpoint(self, save_path: str | Path) -> None:
        import torch
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), save_path / "model.pt")
        self.logger.info("CustomModel saved to %s", save_path)
