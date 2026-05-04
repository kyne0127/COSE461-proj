"""
module.models.act.model
=======================
ACT (Action Chunking with Transformers) plug-in wrapper.

Wraps lerobot's ACT policy, exposing the standard BaseLeRobotModel interface.
Training is delegated to lerobot's built-in train script via subprocess,
or can be run step-by-step via train_step() when using a custom loop.

Dependencies:
    pip install lerobot torch torchvision
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("act")
class ACTModel(BaseLeRobotModel):
    """
    Plug-and-play wrapper around LeRobot's ACTPolicy.

    Config keys:
        device:           "cuda" | "cpu"  (default: "cuda")
        chunk_size:       int             action chunk length (default: 100)
        n_obs_steps:      int             observation horizon (default: 1)
        input_shapes:     dict            camera + state dims
        output_shapes:    dict            action dims
        use_amp:          bool            mixed precision (default: True)
    """

    MODEL_TYPE = "act"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy     = None
        self._optimizer  = None
        self._scaler     = None        # GradScaler for AMP
        self._chunk_size = config.get("chunk_size", 100)
        self._use_amp    = config.get("use_amp", True)
        self._action_buf: Optional[np.ndarray] = None   # (chunk_size, action_dim)
        self._buf_idx    = 0

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.logger.info("Loading ACT checkpoint from %s", checkpoint_path)

        try:
            import torch
            try:
                # lerobot >= 0.5.x
                from lerobot.policies.act.modeling_act import ACTPolicy
                from lerobot.policies.act.configuration_act import ACTConfig
            except ImportError:
                # Backward compatibility for older layouts
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                from lerobot.common.policies.act.configuration_act import ACTConfig
            from lerobot.configs.types import FeatureType, PolicyFeature

            def _to_policy_feature_map(obj):
                if not isinstance(obj, dict):
                    return obj
                out = {}
                for k, v in obj.items():
                    if isinstance(v, PolicyFeature):
                        out[k] = v
                        continue
                    if isinstance(v, dict) and "type" in v and "shape" in v:
                        ftype = v["type"]
                        if isinstance(ftype, str):
                            ftype = FeatureType[ftype]
                        out[k] = PolicyFeature(type=ftype, shape=tuple(v["shape"]))
                    else:
                        out[k] = v
                return out

            cfg_path = checkpoint_path / "config.json"
            if cfg_path.exists():
                import json

                with open(cfg_path) as f:
                    raw_cfg = json.load(f)

                # Remove metadata keys not accepted by ACTConfig dataclass.
                valid_keys = set(ACTConfig.__dataclass_fields__.keys())
                cfg_dict = {k: v for k, v in raw_cfg.items() if k in valid_keys}
                cfg_dict["input_features"] = _to_policy_feature_map(cfg_dict.get("input_features"))
                cfg_dict["output_features"] = _to_policy_feature_map(cfg_dict.get("output_features"))
                act_cfg = ACTConfig(**cfg_dict)
            else:
                cfg_dict = {
                    k: v for k, v in self.config.items()
                    if k in ACTConfig.__dataclass_fields__
                }
                cfg_dict["input_features"] = _to_policy_feature_map(cfg_dict.get("input_features"))
                cfg_dict["output_features"] = _to_policy_feature_map(cfg_dict.get("output_features"))
                act_cfg = ACTConfig(**cfg_dict)

            self._policy = ACTPolicy(act_cfg)
            weights_path = checkpoint_path / "model.safetensors"
            if not weights_path.exists():
                weights_path = checkpoint_path / "pytorch_model.bin"

            if weights_path.exists():
                if weights_path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    state_dict = load_file(str(weights_path), device=str(self._device))
                else:
                    state_dict = torch.load(weights_path, map_location=self._device)
                self._policy.load_state_dict(state_dict, strict=False)
            else:
                self.logger.warning(
                    "No weight file found in %s — using random init", checkpoint_path
                )

            self._policy.to(self._device)
            self._policy.eval()

            if self._use_amp:
                from torch.cuda.amp import GradScaler
                self._scaler = GradScaler()

            self._mark_loaded()

        except ImportError as e:
            raise ImportError(
                "ACTModel requires lerobot and torch. "
                f"Install with: pip install lerobot torch\n{e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        ACT predicts a chunk of actions. We buffer the chunk and return
        one action per call (temporal ensemble can be enabled via config).

        Returns: (action_dim,) float32 numpy array
        """
        if self._policy is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        # Re-fill buffer when exhausted. Some ACT checkpoints can return
        # shorter chunks than configured chunk_size.
        if self._action_buf is None or self._buf_idx >= len(self._action_buf):
            self._action_buf = self._run_policy(observation)
            self._buf_idx    = 0

        action = self._action_buf[self._buf_idx]
        self._buf_idx += 1
        return action

    def _run_policy(self, observation: Observation) -> np.ndarray:
        """Run ACTPolicy forward pass → (chunk_size, action_dim)."""
        import torch
        from torch.cuda.amp import autocast

        with torch.no_grad():
            batch = self._obs_to_batch(observation)
            with autocast(enabled=self._use_amp):
                actions = self._policy.select_action(batch)  # (1, chunk, action_dim)

        arr = actions.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _obs_to_batch(self, obs: Observation) -> Dict[str, Any]:
        """Convert Observation → policy-ready batch dict."""
        import torch

        batch: Dict[str, Any] = {}

        # Images: (1, C, H, W) float32 [0, 1]
        for cam_name, img in obs.images.items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
            batch[f"observation.images.{cam_name}"] = t

        # State
        if obs.state is not None:
            batch["observation.state"] = torch.from_numpy(
                obs.state.astype(np.float32)
            ).unsqueeze(0).to(self._device)

        return batch

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """Single gradient step. Returns loss dict."""
        import torch
        from torch.cuda.amp import autocast

        if self._policy is None:
            raise RuntimeError("Model not loaded.")

        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._policy.parameters(),
                lr=self.config.get("lr", 1e-4),
                weight_decay=self.config.get("weight_decay", 1e-4),
            )

        self._policy.train()
        self._optimizer.zero_grad()

        policy_batch = self._batch_to_policy(batch)

        with autocast(enabled=self._use_amp):
            loss, loss_dict = self._policy.compute_loss(policy_batch)

        if self._use_amp and self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self._optimizer.step()

        self._policy.eval()
        return {k: float(v) for k, v in loss_dict.items()}

    def _batch_to_policy(self, batch: TrainingBatch) -> Dict[str, Any]:
        import torch

        pb: Dict[str, Any] = {}
        for cam, imgs in batch.images.items():
            t = torch.tensor(imgs, dtype=torch.float32).to(self._device)
            pb[f"observation.images.{cam}"] = t
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
        """Clear action buffer at episode start."""
        self._action_buf = None
        self._buf_idx    = 0
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def save_checkpoint(self, save_path: str | Path) -> None:
        import torch
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), save_path / "model.safetensors")
        self.logger.info("ACT checkpoint saved to %s", save_path)
