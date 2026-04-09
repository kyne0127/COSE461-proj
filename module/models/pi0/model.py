"""
module.models.pi0.model
=======================
Pi0 / Vision-Language-Action (VLA) plug-in wrapper.

Pi0 from Physical Intelligence (or any VLA that follows the same interface):
  - Accepts camera images + language instruction
  - Returns action sequences via flow matching / diffusion

Dependencies:
    pip install lerobot torch transformers
    # Pi0 checkpoint available on HuggingFace: lerobot/pi0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("pi0")
class Pi0Model(BaseLeRobotModel):
    """
    Plug-and-play wrapper for Pi0 / VLA models.

    Config keys:
        device:            "cuda" | "cpu"
        model_id:          str   HuggingFace model ID (default: "lerobot/pi0")
        action_horizon:    int   steps to execute from chunk (default: 1)
        use_amp:           bool  mixed precision (default: True)
        language_instruction: str  default task description if not per-step
    """

    MODEL_TYPE = "pi0"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy         = None
        self._optimizer      = None
        self._use_amp        = config.get("use_amp", True)
        self._model_id       = config.get("model_id", "lerobot/pi0")
        self._default_task   = config.get("language_instruction", "")
        self._action_horizon = config.get("action_horizon", 1)

        # Action buffer for chunk execution
        self._action_buf: np.ndarray | None = None
        self._buf_idx    = 0

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.logger.info("Loading Pi0 model from %s (or HF: %s)",
                         checkpoint_path, self._model_id)

        try:
            import torch
            # Try loading from local path first, fall back to HuggingFace
            load_path = str(checkpoint_path) if checkpoint_path.exists() else self._model_id

            # LeRobot Pi0 policy
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            from lerobot.common.policies.pi0.configuration_pi0 import PI0Config

            cfg_path = checkpoint_path / "config.json"
            if cfg_path.exists():
                pi0_cfg = PI0Config.from_pretrained(load_path)
            else:
                pi0_cfg = PI0Config(**{
                    k: v for k, v in self.config.items()
                    if k in PI0Config.__dataclass_fields__
                })

            self._policy = PI0Policy.from_pretrained(load_path, config=pi0_cfg)
            self._policy.to(self._device)
            self._policy.eval()
            self._mark_loaded()

        except ImportError as e:
            raise ImportError(
                "Pi0Model requires lerobot, torch, and transformers.\n"
                f"Install: pip install lerobot transformers\n{e}"
            ) from e
        except Exception as e:
            self.logger.error("Failed to load Pi0: %s", e)
            raise

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        VLA inference: language instruction + cameras → action chunk.
        Returns: (action_dim,) float32 numpy array.
        """
        if self._policy is None:
            raise RuntimeError("Model not loaded.")

        # Refill buffer
        if self._action_buf is None or self._buf_idx >= self._action_horizon:
            self._action_buf = self._run_vla(observation)
            self._buf_idx    = 0

        action = self._action_buf[self._buf_idx]
        self._buf_idx += 1
        return action

    def _run_vla(self, observation: Observation) -> np.ndarray:
        """Run Pi0 flow-matching forward pass."""
        import torch
        from torch.cuda.amp import autocast

        task_text = observation.task_text or self._default_task
        batch = self._obs_to_batch(observation, task_text)

        with torch.no_grad(), autocast(enabled=self._use_amp):
            actions = self._policy.select_action(batch)

        return actions.squeeze(0).cpu().numpy().astype(np.float32)

    def _obs_to_batch(self, obs: Observation, task_text: str) -> Dict[str, Any]:
        import torch
        batch: Dict[str, Any] = {"task": [task_text]}
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
    # Training (supervised fine-tuning / flow matching loss)
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        import torch
        from torch.cuda.amp import autocast, GradScaler

        if self._policy is None:
            raise RuntimeError("Model not loaded.")
        if self._optimizer is None:
            # Fine-tune only non-frozen params
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

        with autocast(enabled=self._use_amp):
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
        pb["task"] = batch.extra.get("task_texts", [""] * 1)
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
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def save_checkpoint(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(str(save_path))
        self.logger.info("Pi0 checkpoint saved to %s", save_path)
