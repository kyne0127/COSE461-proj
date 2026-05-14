"""
module.models.smolvla.model
===========================
SmolVLA compact VLA local inference wrapper.

SmolVLA from HuggingFace/lerobot:
  - SigLIP vision encoder + SmolLM2 language backbone
  - ~500M params, designed for 8 GB VRAM
  - Accepts camera images + proprioception + language instruction
  - Returns action sequence via flow matching

Dependencies:
    pip install lerobot torch transformers
    # SmolVLA checkpoint: lerobot/smolvla
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("smolvla")
class SmolVLAModel(BaseLeRobotModel):
    """
    Local wrapper for SmolVLA compact VLA model.

    Intended for desktop inference (RTX 3060 / 8 GB VRAM).
    Molmo/AmbRes remains on the GPU server; SmolVLA runs locally
    for low-latency action generation after task disambiguation.

    Config keys:
        device:            "cuda" | "cpu"
        model_id:          str   HuggingFace model ID (default: "lerobot/smolvla")
        action_horizon:    int   steps to execute from each predicted chunk (default: 1)
        use_amp:           bool  mixed-precision inference (default: True)
        precision:         str   "bfloat16" | "float16" | "float32" (default: "bfloat16")
        lr:                float fine-tuning learning rate (default: 1e-4)
        weight_decay:      float fine-tuning weight decay (default: 1e-5)
    """

    MODEL_TYPE = "smolvla"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy         = None
        self._optimizer      = None
        self._scaler         = None
        self._model_id       = config.get("model_id", "lerobot/smolvla")
        self._action_horizon = config.get("action_horizon", 1)
        self._use_amp        = config.get("use_amp", True)
        self._precision      = config.get("precision", "bfloat16")

        # chunk-execution buffer
        self._action_buf: np.ndarray | None = None
        self._buf_idx = 0

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.logger.info(
            "Loading SmolVLA from %s (HF fallback: %s)",
            checkpoint_path, self._model_id,
        )
        try:
            import torch
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            load_path = str(checkpoint_path) if checkpoint_path.exists() else self._model_id
            self._policy = SmolVLAPolicy.from_pretrained(load_path)
            self._policy.to(self._device)
            self._policy.eval()

            if self._precision == "bfloat16" and torch.cuda.is_available():
                self._policy = self._policy.to(torch.bfloat16)
            elif self._precision == "float16" and torch.cuda.is_available():
                self._policy = self._policy.half()

            self._mark_loaded()

        except ImportError as e:
            raise ImportError(
                "SmolVLAModel requires lerobot and torch.\n"
                f"Install: pip install lerobot transformers\n{e}"
            ) from e
        except Exception as e:
            self.logger.error("Failed to load SmolVLA: %s", e)
            raise

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        VLA inference: task_text + cameras + state → action.
        Returns: (action_dim,) float32 numpy array.
        """
        if self._policy is None:
            raise RuntimeError("SmolVLAModel not loaded. Call load_checkpoint() first.")

        if self._action_buf is None or self._buf_idx >= self._action_horizon:
            self._action_buf = self._run_vla(observation)
            self._buf_idx = 0

        action = self._action_buf[self._buf_idx]
        self._buf_idx += 1
        return action

    def _run_vla(self, observation: Observation) -> np.ndarray:
        import torch

        batch = self._obs_to_batch(observation)
        amp_ctx = torch.amp.autocast(
            "cuda",
            enabled=self._use_amp and torch.cuda.is_available(),
        )
        with torch.no_grad(), amp_ctx:
            actions = self._policy.select_action(batch)

        # actions: (1, action_horizon, action_dim) or (1, action_dim)
        arr = actions.squeeze(0).cpu().float().numpy()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]   # (1, action_dim)
        return arr

    def _obs_to_batch(self, obs: Observation) -> Dict[str, Any]:
        import torch

        batch: Dict[str, Any] = {"task": [obs.task_text]}

        for cam_name, img in obs.images.items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
            if self._precision == "bfloat16":
                t = t.to(torch.bfloat16)
            elif self._precision == "float16":
                t = t.half()
            batch[f"observation.images.{cam_name}"] = t

        if obs.state is not None:
            s = torch.from_numpy(obs.state.astype(np.float32)).unsqueeze(0).to(self._device)
            if self._precision == "bfloat16":
                s = s.to(torch.bfloat16)
            elif self._precision == "float16":
                s = s.half()
            batch["observation.state"] = s

        return batch

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        import torch
        from torch.amp import GradScaler, autocast

        if self._policy is None:
            raise RuntimeError("SmolVLAModel not loaded.")

        if self._optimizer is None:
            params = [p for p in self._policy.parameters() if p.requires_grad]
            self._optimizer = torch.optim.AdamW(
                params,
                lr=self.config.get("lr", 1e-4),
                weight_decay=self.config.get("weight_decay", 1e-5),
            )
            self._scaler = GradScaler()

        self._policy.train()
        self._optimizer.zero_grad()
        pb = self._batch_to_policy(batch)

        with autocast("cuda", enabled=self._use_amp):
            loss, loss_dict = self._policy.compute_loss(pb)

        self._scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._scaler.step(self._optimizer)
        self._scaler.update()

        self._policy.eval()
        return {k: float(v) for k, v in loss_dict.items()}

    def _batch_to_policy(self, batch: TrainingBatch) -> Dict[str, Any]:
        import torch

        pb: Dict[str, Any] = {}
        pb["task"] = batch.extra.get("task_texts", [""])
        for cam, imgs in batch.images.items():
            pb[f"observation.images.{cam}"] = torch.tensor(
                imgs, dtype=torch.float32,
            ).to(self._device)
        if batch.states is not None:
            pb["observation.state"] = torch.tensor(
                batch.states, dtype=torch.float32,
            ).to(self._device)
        if batch.actions is not None:
            pb["action"] = torch.tensor(
                batch.actions, dtype=torch.float32,
            ).to(self._device)
        return pb

    # ------------------------------------------------------------------ #
    # Reset / Save
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._action_buf = None
        self._buf_idx = 0
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def save_checkpoint(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(str(save_path))
        self.logger.info("SmolVLA checkpoint saved to %s", save_path)
