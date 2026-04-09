"""
module.models.tdmpc2.model
==========================
TD-MPC2 (Model-based RL) plug-in wrapper.

TD-MPC2 differs from IL methods:
  - Requires an environment / reward signal for RL fine-tuning
  - Inference uses MPC planning (MPPI by default)
  - train_step() runs a model-based Bellman update

Dependencies:
    pip install lerobot torch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("tdmpc2")
class TDMPC2Model(BaseLeRobotModel):
    """
    Plug-and-play wrapper around LeRobot's TDMPC2Policy.

    Config keys:
        device:           "cuda" | "cpu"
        mpc_horizon:      int    planning horizon (default: 5)
        n_samples:        int    MPPI samples (default: 512)
        use_amp:          bool   mixed precision (default: True)
        task_dim:         int    task embedding dim (default: 96) — multi-task
    """

    MODEL_TYPE = "tdmpc2"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy    = None
        self._optimizer = None
        self._use_amp   = config.get("use_amp", True)

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.logger.info("Loading TD-MPC2 checkpoint from %s", checkpoint_path)

        try:
            import torch
            from lerobot.common.policies.tdmpc2.modeling_tdmpc2 import TDMPC2Policy
            from lerobot.common.policies.tdmpc2.configuration_tdmpc2 import TDMPC2Config

            cfg_path = checkpoint_path / "config.json"
            if cfg_path.exists():
                cfg = TDMPC2Config.from_pretrained(str(checkpoint_path))
            else:
                cfg = TDMPC2Config(**{
                    k: v for k, v in self.config.items()
                    if k in TDMPC2Config.__dataclass_fields__
                })

            self._policy = TDMPC2Policy(cfg, dataset_stats=None)
            weights_path = checkpoint_path / "model.safetensors"
            if not weights_path.exists():
                weights_path = checkpoint_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self._device)
                self._policy.load_state_dict(state_dict, strict=False)
            else:
                self.logger.warning("No weights found — using random init")

            self._policy.to(self._device)
            self._policy.eval()
            self._mark_loaded()

        except ImportError as e:
            raise ImportError(
                f"TDMPC2Model requires lerobot and torch.\n{e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        Run MPPI planning to select the best action.
        Returns: (action_dim,) float32 numpy array
        """
        if self._policy is None:
            raise RuntimeError("Model not loaded.")

        import torch
        from torch.cuda.amp import autocast

        batch = self._obs_to_batch(observation)
        with torch.no_grad(), autocast(enabled=self._use_amp):
            action = self._policy.select_action(batch)

        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _obs_to_batch(self, obs: Observation) -> Dict[str, Any]:
        import torch
        batch: Dict[str, Any] = {}
        for cam_name, img in obs.images.items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
            batch[f"observation.images.{cam_name}"] = t
        if obs.state is not None:
            batch["observation.state"] = torch.from_numpy(
                obs.state.astype(np.float32)
            ).unsqueeze(0).to(self._device)
        return batch

    # ------------------------------------------------------------------ #
    # Training (model-based RL step)
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        TD-MPC2 training step:
         1. World model update (reconstruction + reward prediction)
         2. Policy update (actor loss)
         3. Value function update (TD target)
        """
        import torch
        from torch.cuda.amp import autocast, GradScaler

        if self._policy is None:
            raise RuntimeError("Model not loaded.")
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._policy.parameters(),
                lr=self.config.get("lr", 3e-4),
            )
            self._scaler = GradScaler()

        self._policy.train()
        self._optimizer.zero_grad()
        pb = self._batch_to_policy(batch)

        with autocast(enabled=self._use_amp):
            loss, loss_dict = self._policy.compute_loss(pb)

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        self._policy.eval()
        return {k: float(v) for k, v in loss_dict.items()}

    def _batch_to_policy(self, batch: TrainingBatch) -> Dict[str, Any]:
        import torch
        pb: Dict[str, Any] = {}
        for cam, imgs in batch.images.items():
            pb[f"observation.images.{cam}"] = torch.tensor(
                imgs, dtype=torch.float32
            ).to(self._device)
        if batch.states is not None:
            pb["observation.state"] = torch.tensor(
                batch.states, dtype=torch.float32
            ).to(self._device)
        if batch.actions is not None:
            pb["action"] = torch.tensor(
                batch.actions, dtype=torch.float32
            ).to(self._device)
        if batch.rewards is not None:
            pb["next.reward"] = torch.tensor(
                batch.rewards, dtype=torch.float32
            ).to(self._device)
        if batch.dones is not None:
            pb["next.done"] = torch.tensor(
                batch.dones, dtype=torch.bool
            ).to(self._device)
        return pb

    # ------------------------------------------------------------------ #
    # Reset / Save
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def save_checkpoint(self, save_path: str | Path) -> None:
        import torch
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), save_path / "model.safetensors")
        self.logger.info("TD-MPC2 checkpoint saved to %s", save_path)
