"""
module.models.diffusion.model
==============================
Diffusion Policy plug-in wrapper.

Wraps lerobot's DiffusionPolicy (DDPM / DDIM denoising).

Dependencies:
    pip install lerobot torch torchvision diffusers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("diffusion")
class DiffusionPolicyModel(BaseLeRobotModel):
    """
    Plug-and-play wrapper around LeRobot's DiffusionPolicy.

    Config keys:
        device:             "cuda" | "cpu"
        num_inference_steps: int   DDIM steps at inference (default: 10)
        obs_horizon:        int    observation horizon (default: 2)
        action_horizon:     int    action execution horizon (default: 8)
        pred_horizon:       int    prediction horizon (default: 16)
        use_amp:            bool   mixed precision (default: True)
    """

    MODEL_TYPE = "diffusion"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy             = None
        self._optimizer          = None
        self._num_inf_steps      = config.get("num_inference_steps", 10)
        self._action_horizon     = config.get("action_horizon", 8)
        self._pred_horizon       = config.get("pred_horizon", 16)
        self._use_amp            = config.get("use_amp", True)

        # Sliding window action buffer
        self._action_buf: np.ndarray | None = None
        self._buf_idx    = 0

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.logger.info("Loading DiffusionPolicy checkpoint from %s", checkpoint_path)

        try:
            import torch
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig

            cfg_path = checkpoint_path / "config.json"
            if cfg_path.exists():
                diff_cfg = DiffusionConfig.from_pretrained(str(checkpoint_path))
            else:
                diff_cfg = DiffusionConfig(**{
                    k: v for k, v in self.config.items()
                    if k in DiffusionConfig.__dataclass_fields__
                })

            self._policy = DiffusionPolicy(diff_cfg)
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
                "DiffusionPolicyModel requires lerobot, torch, and diffusers.\n"
                f"Install: pip install lerobot diffusers\n{e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("Model not loaded.")

        # Refill buffer
        if self._action_buf is None or self._buf_idx >= self._action_horizon:
            self._action_buf = self._run_diffusion(observation)
            self._buf_idx    = 0

        action = self._action_buf[self._buf_idx]
        self._buf_idx += 1
        return action

    def _run_diffusion(self, observation: Observation) -> np.ndarray:
        """Run full DDIM denoising → (pred_horizon, action_dim)."""
        import torch
        from torch.cuda.amp import autocast

        batch = self._obs_to_batch(observation)
        with torch.no_grad(), autocast(enabled=self._use_amp):
            actions = self._policy.select_action(batch)

        return actions.squeeze(0).cpu().numpy().astype(np.float32)

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
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        import torch
        from torch.cuda.amp import autocast, GradScaler

        if self._policy is None:
            raise RuntimeError("Model not loaded.")
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._policy.parameters(),
                lr=self.config.get("lr", 1e-4),
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
        return pb

    # ------------------------------------------------------------------ #
    # Reset / Save
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._action_buf = None
        self._buf_idx    = 0

    def save_checkpoint(self, save_path: str | Path) -> None:
        import torch
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), save_path / "model.safetensors")
        self.logger.info("DiffusionPolicy checkpoint saved to %s", save_path)
