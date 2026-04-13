"""
module.models.rgbd_diffusion.model
===================================
RGB-D Diffusion Policy Implementation

This module implements a Diffusion Policy variant that accepts RGB-D (RGBD) input:
- Global view: RGBD tensor from ZED Mini (RGB concat with Depth)
- Ego view: RGB from USB wrist camera
- State: 6-DOF joint angles (radians)
- Action: 6-DOF delta joint commands

Paper Observation Type: RGB-D (Case B)
Preprocessing: Concatenate ZED RGB + Depth (mm→m conversion) along channel axis
                Result shape: [H, W, 4] → Reshaped to (B, 4, H, W) for model forward pass

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


def depth_mm_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """
    Convert depth from millimeters to meters.

    Args:
        depth_mm: Depth array in mm (typically [H, W, 1])

    Returns:
        Depth array in meters with same shape
    """
    return depth_mm / 1000.0


def create_rgbd_tensor(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Concatenate RGB and Depth along channel axis to create RGBD tensor.

    Args:
        rgb: RGB image [H, W, 3] float32 [0,1]
        depth: Depth image [H, W, 1] float32 (metres)

    Returns:
        RGBD tensor [H, W, 4] float32
    """
    if rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape [H, W, 3], got {rgb.shape}")
    if depth.shape[-1] != 1:
        raise ValueError(f"Expected Depth shape [H, W, 1], got {depth.shape}")

    rgbd = np.concatenate([rgb, depth], axis=-1)  # [H, W, 4]
    return rgbd


@ModelRegistry.register("rgbd_diffusion")
class RGBDDiffusionPolicyModel(BaseLeRobotModel):
    """
    Plug-and-play RGB-D Diffusion Policy wrapper.

    Accepts:
        - Global input: RGBD [H, W, 4] from ZED Mini (RGB concat Depth in m)
        - Ego input: USB RGB [H, W, 3]
        - State: Joint angles [6] (radians)

    Outputs:
        - Action: Delta joint commands [6]

    Config keys:
        device:              "cuda" | "cpu"
        num_inference_steps: int   DDIM steps at inference (default: 10)
        obs_horizon:         int   observation horizon (default: 2)
        action_horizon:      int   action execution horizon (default: 8)
        pred_horizon:        int   prediction horizon (default: 16)
        use_amp:             bool  mixed precision (default: True)
    """

    MODEL_TYPE = "rgbd_diffusion"

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
        self.logger.info("Loading RGB-D DiffusionPolicy checkpoint from %s", checkpoint_path)

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
                "RGBDDiffusionPolicyModel requires lerobot, torch, and diffusers.\n"
                f"Install: pip install lerobot diffusers\n{e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        Predict action from RGB-D observation.

        Expected observation.images keys:
            - "zed_rgb": [H, W, 3] float32 [0,1]
            - "zed_depth": [H, W, 1] float32 (metres)
            - "usb_rgb": [H, W, 3] float32 [0,1]

        Returns:
            Action array [6] or [action_horizon, 6] depending on model
        """
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
        """
        Run full DDIM denoising → (pred_horizon, action_dim).

        Preprocesses RGB-D by concatenating ZED RGB and depth.
        """
        import torch
        from torch.cuda.amp import autocast

        batch = self._obs_to_batch(observation)
        with torch.no_grad(), autocast(enabled=self._use_amp):
            actions = self._policy.select_action(batch)

        return actions.squeeze(0).cpu().numpy().astype(np.float32)

    def _obs_to_batch(self, obs: Observation) -> Dict[str, Any]:
        """
        Convert Observation to policy batch format.

        Handles RGB-D preprocessing:
            - Creates RGBD tensor from zed_rgb + zed_depth (concat on channel axis)
            - Reshapes to (B, 4, H, W) for model forward
            - Adds usb_rgb as separate camera input
        """
        import torch
        batch: Dict[str, Any] = {}

        # Process ZED RGB-D
        if "zed_rgb" in obs.images and "zed_depth" in obs.images:
            zed_rgb = obs.images["zed_rgb"]      # [H, W, 3]
            zed_depth = obs.images["zed_depth"]  # [H, W, 1] (already in metres)

            # Ensure proper normalization
            if zed_rgb.dtype == np.uint8:
                zed_rgb = zed_rgb.astype(np.float32) / 255.0

            # Create RGBD tensor: [H, W, 4]
            rgbd = create_rgbd_tensor(zed_rgb, zed_depth)

            # Convert to torch tensor: (B, 4, H, W)
            t_rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0).to(self._device)
            batch["observation.images.zed_rgbd"] = t_rgbd

        # Process USB RGB
        if "usb_rgb" in obs.images:
            usb_rgb = obs.images["usb_rgb"]  # [H, W, 3]

            if usb_rgb.dtype == np.uint8:
                usb_rgb = usb_rgb.astype(np.float32) / 255.0

            t_usb = torch.from_numpy(usb_rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
            batch["observation.images.usb_rgb"] = t_usb

        # Process state
        if obs.state is not None:
            batch["observation.state"] = torch.from_numpy(
                obs.state.astype(np.float32)
            ).unsqueeze(0).to(self._device)

        return batch

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute one gradient update step.

        Handles RGB-D preprocessing in training context.
        """
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
        """
        Convert TrainingBatch to policy batch format for training.

        Preprocesses RGB-D data if present.
        """
        import torch
        pb: Dict[str, Any] = {}

        for cam, imgs in batch.images.items():
            # Handle RGB-D preprocessing in training
            if cam == "zed_rgbd":
                # imgs shape: (B, T, H, W, 4)
                pb[f"observation.images.{cam}"] = torch.tensor(
                    imgs, dtype=torch.float32
                ).to(self._device)
            else:
                # Standard RGB camera
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
        """Reset action buffer at episode start."""
        self._action_buf = None
        self._buf_idx    = 0

    def save_checkpoint(self, save_path: str | Path) -> None:
        """Save model weights."""
        import torch
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), save_path / "model.safetensors")
        self.logger.info("RGB-D DiffusionPolicy checkpoint saved to %s", save_path)
