"""
module.models.smolvla.model
===========================
SmolVLA compact VLA local inference wrapper.

Uses lerobot 0.5.x API:
  - Import path: lerobot.policies.smolvla
  - Preprocessing:  make_smolvla_pre_post_processors → tokenise + normalise
  - Inference:      prepare_observation_for_inference → preprocessor → select_action → postprocessor
  - Action chunking (n_action_steps=50) is handled by SmolVLAPolicy's internal deque

Fine-tuning modes (controlled by config):
  visual_finetune=False (default):
      train_expert_only=True  — lm_expert only, all else frozen
  visual_finetune=True:
      SigLIP vision_model + lm_expert learn; SmolLM2 text_model stays frozen
      → adapts visual representations to new camera / environment
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
    Local wrapper for SmolVLA compact VLA model (lerobot 0.5.x).

    Runs on desktop GPU (RTX 3060 / 8 GB VRAM) for low-latency action generation.
    Action chunking (n_action_steps=50) is handled internally by SmolVLAPolicy.

    Config keys:
        device:          "cuda" | "cpu"
        model_id:        HuggingFace ID or local path  (default: "lerobot/smolvla_base")
        use_amp:         mixed-precision inference       (default: True)
        precision:       "bfloat16" | "float16" | "float32"  (default: "bfloat16")
        visual_finetune: enable visual domain adaptation mode (default: False)
                         True  → SigLIP + lm_expert learn, SmolLM2 text frozen
                         False → lm_expert only (SmolVLA default)
        grad_checkpoint: gradient checkpointing to reduce VRAM (default: True when visual_finetune)
        lr:              fine-tuning learning rate       (default: 1e-4)
        weight_decay:    fine-tuning weight decay        (default: 1e-5)
    """

    MODEL_TYPE = "smolvla"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._policy          = None
        self._preprocessor    = None
        self._postprocessor   = None
        self._optimizer       = None
        self._scaler          = None
        self._model_id        = config.get("model_id", "lerobot/smolvla_base")
        self._use_amp         = config.get("use_amp", True)
        self._precision       = config.get("precision", "bfloat16")
        self._visual_finetune = config.get("visual_finetune", False)
        self._grad_checkpoint = config.get("grad_checkpoint", self._visual_finetune)

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        load_path = str(checkpoint_path) if checkpoint_path.exists() else self._model_id
        self.logger.info("Loading SmolVLA from '%s'", load_path)

        try:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.smolvla.processor_smolvla import (
                make_smolvla_pre_post_processors,
            )

            self._policy = SmolVLAPolicy.from_pretrained(load_path)
            self._policy = self._policy.to(self._device)
            self._policy.eval()

            if self._precision == "bfloat16" and torch.cuda.is_available():
                self._policy = self._policy.to(torch.bfloat16)
            elif self._precision == "float16" and torch.cuda.is_available():
                self._policy = self._policy.to(torch.float16)

            dataset_stats = getattr(self._policy, "dataset_stats", None)
            self._preprocessor, self._postprocessor = make_smolvla_pre_post_processors(
                self._policy.config, dataset_stats=dataset_stats
            )

            if self._visual_finetune:
                self._apply_visual_finetune_freeze()

            if self._grad_checkpoint:
                self._enable_gradient_checkpointing()

            self._mark_loaded()
            self.logger.info(
                "SmolVLA loaded on %s (%s) | visual_finetune=%s | grad_checkpoint=%s",
                self._device, self._precision,
                self._visual_finetune, self._grad_checkpoint,
            )

        except ImportError as e:
            raise ImportError(
                "SmolVLAModel requires lerobot>=0.5 and torch.\n"
                f"Install: pip install lerobot transformers\n{e}"
            ) from e
        except Exception:
            self.logger.exception("Failed to load SmolVLA from '%s'", load_path)
            raise

    def _apply_visual_finetune_freeze(self) -> None:
        """
        Visual domain adaptation mode:
          - SigLIP  (vlm.model.vision_model)   → trainable  (visual adaptation)
          - lm_expert                           → trainable  (action adaptation)
          - SmolLM2 (vlm.model.text_model)      → frozen     (language unchanged)

        Overrides the default train_expert_only=True behaviour which would also
        freeze the vision encoder.
        """
        vlm_with_expert = self._policy.model.vlm_with_expert

        # Start from all-frozen
        for p in self._policy.parameters():
            p.requires_grad_(False)

        # Unfreeze SigLIP vision encoder
        for p in vlm_with_expert.vlm.model.vision_model.parameters():
            p.requires_grad_(True)

        # Unfreeze action expert
        for p in vlm_with_expert.lm_expert.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self._policy.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self._policy.parameters())
        self.logger.info(
            "Visual finetune: trainable %dM / %dM params (%.1f%%) "
            "[vision_model + lm_expert, text_model frozen]",
            trainable // 1_000_000,
            total     // 1_000_000,
            100.0 * trainable / total,
        )

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on SigLIP and SmolLM2 to reduce VRAM."""
        vlm = self._policy.model.vlm_with_expert.vlm
        try:
            vlm.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled on VLM")
        except Exception as e:
            self.logger.warning("Could not enable gradient checkpointing: %s", e)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        VLA inference: task_text + cameras + state → action (action_dim,) float32.

        SmolVLA maintains an internal action deque (n_action_steps=50) and
        re-runs the model only when the deque is exhausted.
        """
        if self._policy is None:
            raise RuntimeError("SmolVLAModel not loaded. Call load_checkpoint() first.")

        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        raw_obs: Dict[str, np.ndarray] = {}
        for cam_name, img in observation.images.items():
            raw_obs[f"observation.images.{cam_name}"] = img
        if observation.state is not None:
            raw_obs["observation.state"] = observation.state.astype(np.float32)

        device = torch.device(self._device)
        prepared = prepare_observation_for_inference(
            raw_obs, device, task=observation.task_text
        )
        batch = self._preprocessor(prepared)

        amp_dtype = (
            torch.bfloat16 if self._precision == "bfloat16"
            else torch.float16 if self._precision == "float16"
            else torch.float32
        )
        amp_ctx = torch.amp.autocast(
            "cuda",
            enabled=self._use_amp and torch.cuda.is_available(),
            dtype=amp_dtype,
        )
        with torch.no_grad(), amp_ctx:
            action_tensor = self._policy.select_action(batch)

        action_tensor = self._postprocessor(action_tensor)
        return action_tensor.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        One gradient step on trainable parameters.

        In visual_finetune mode: updates SigLIP + lm_expert only.
        In default mode:         updates lm_expert only.

        Expects TrainingBatch with raw (unnormalised) data.
        """
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
            if self._visual_finetune:
                # Lower LR for vision encoder to avoid disrupting pretrained features
                vision_params = list(
                    self._policy.model.vlm_with_expert.vlm.model.vision_model.parameters()
                )
                vision_ids = {id(p) for p in vision_params}
                expert_params = [p for p in params if id(p) not in vision_ids]
                self._optimizer = torch.optim.AdamW(
                    [
                        {"params": vision_params, "lr": self.config.get("lr_vision", 1e-5)},
                        {"params": expert_params, "lr": self.config.get("lr", 1e-4)},
                    ],
                    weight_decay=self.config.get("weight_decay", 1e-5),
                )
            self._scaler = GradScaler()

        self._policy.train()
        self._optimizer.zero_grad()

        pb = self._batch_to_policy(batch)
        pb = self._preprocessor(pb)

        with autocast("cuda", enabled=self._use_amp):
            loss, loss_dict = self._policy.forward(pb)

        self._scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._policy.parameters() if p.requires_grad], 1.0
        )
        self._scaler.step(self._optimizer)
        self._scaler.update()

        self._policy.eval()
        return {k: float(v) for k, v in loss_dict.items()}

    def _batch_to_policy(self, batch: TrainingBatch) -> Dict[str, Any]:
        """Convert TrainingBatch to the dict format expected by the preprocessor.

        Images: (B, H, W, C) uint8 numpy → (B, C, H, W) float32 [0,1] tensor
        The preprocessor then handles tokenisation, state/action normalisation,
        and device placement.
        """
        import torch

        pb: Dict[str, Any] = {}
        pb["task"] = batch.extra.get("task_texts", [""])
        for cam, imgs in batch.images.items():
            t = torch.tensor(imgs, dtype=torch.float32)  # (B, H, W, C)
            if t.ndim == 4:
                t = t.permute(0, 3, 1, 2)               # (B, C, H, W)
            if t.max() > 1.0:
                t = t / 255.0
            pb[f"observation.images.{cam}"] = t.to(self._device)
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
        """Reset SmolVLA's internal action deque (call at episode start)."""
        if self._policy is not None:
            self._policy.reset()

    def save_checkpoint(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(str(save_path))
        self.logger.info("SmolVLA checkpoint saved to %s", save_path)
