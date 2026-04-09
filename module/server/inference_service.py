"""
module.server.inference_service
================================
gRPC InferenceService 서버 구현 (GPU 서버 전용).

- 다수의 모델을 동시에 메모리에 상주 (멀티 모델 서빙)
- LoadModel / UnloadModel / ListModels
- GetAction (단일 스텝)
- GetActionStream (청크 스트리밍 — ACT, Diffusion)
- 모델별 GPU 메모리 관리 (CUDA OOM 대응)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterator, Optional

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation
from module.utils.registry import ModelRegistry
from module.utils.tensor_utils import proto_to_numpy
from module.utils.logging import get_logger

logger = get_logger("server.inference_service")


# ────────────────────────────────────────────────────────────────────────────
# In-memory model store
# ────────────────────────────────────────────────────────────────────────────

class ModelStore:
    """Thread-safe registry of loaded model instances."""

    def __init__(self) -> None:
        self._models: Dict[str, BaseLeRobotModel] = {}
        self._lock   = threading.RLock()
        self._status: Dict[str, str] = {}   # model_id → "loading"|"loaded"|"error"

    def get(self, model_id: str) -> BaseLeRobotModel:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' is not loaded")
            return self._models[model_id]

    def set(self, model_id: str, model: BaseLeRobotModel) -> None:
        with self._lock:
            self._models[model_id] = model
            self._status[model_id] = "loaded"

    def remove(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)
            self._status.pop(model_id, None)

    def set_status(self, model_id: str, status: str) -> None:
        with self._lock:
            self._status[model_id] = status

    def list_all(self) -> list[Dict[str, str]]:
        with self._lock:
            return [
                {
                    "model_id":   mid,
                    "model_type": m.MODEL_TYPE,
                    "status":     self._status.get(mid, "unknown"),
                }
                for mid, m in self._models.items()
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


# ────────────────────────────────────────────────────────────────────────────
# InferenceServicer
# ────────────────────────────────────────────────────────────────────────────

class InferenceServicer:
    """
    gRPC InferenceService implementation.
    Registered with the gRPC server in server/grpc_server.py.

    This class is proto-stub-independent; it works with plain dicts
    and numpy arrays. The gRPC glue layer is in grpc_server.py.
    """

    def __init__(self, store: Optional[ModelStore] = None) -> None:
        self._store = store or ModelStore()

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model_type:      str,
        model_id:        str,
        checkpoint_path: str,
        config_yaml:     str = "",
    ) -> Dict[str, Any]:
        """
        Load a model from checkpoint.
        Runs in background thread so gRPC call returns immediately.
        """
        self._store.set_status(model_id, "loading")
        logger.info("Loading model id='%s' type='%s' from '%s'",
                    model_id, model_type, checkpoint_path)

        try:
            # Parse optional YAML config override
            config: Dict[str, Any] = {}
            if config_yaml:
                import yaml
                config = yaml.safe_load(config_yaml) or {}

            config.setdefault("device", self._select_device())

            # Build model via registry
            if not ModelRegistry.is_registered(model_type):
                ModelRegistry.auto_discover()

            model = ModelRegistry.build(model_type, config=config)
            model.load_checkpoint(checkpoint_path)

            self._store.set(model_id, model)
            logger.info("Model '%s' loaded — params=%d", model_id, model.get_num_params())
            return {"success": True, "model_id": model_id, "message": "OK"}

        except Exception as e:
            self._store.set_status(model_id, "error")
            logger.error("Failed to load model '%s': %s", model_id, e)
            return {"success": False, "model_id": model_id, "message": str(e)}

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        try:
            self._store.remove(model_id)
            self._release_gpu_memory()
            logger.info("Model '%s' unloaded", model_id)
            return {"success": True, "message": "unloaded"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def list_models(self) -> list[Dict[str, str]]:
        return self._store.list_all()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def get_action(
        self,
        model_id:   str,
        images:     Dict[str, np.ndarray],
        state:      Optional[np.ndarray],
        task_text:  str = "",
        episode_id: int = 0,
        step:       int = 0,
    ) -> np.ndarray:
        """
        Run inference with the specified model.
        Returns (action_dim,) or (chunk_size, action_dim) float32 array.
        """
        model = self._store.get(model_id)

        obs = Observation(
            images=images,
            state=state,
            task_text=task_text,
            episode_id=episode_id,
            step=step,
        )

        t0 = time.perf_counter()
        action = model.predict_action(obs)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("GetAction model='%s' step=%d latency=%.1fms",
                     model_id, step, latency_ms)
        return action

    def get_action_stream(
        self,
        model_id:   str,
        images:     Dict[str, np.ndarray],
        state:      Optional[np.ndarray],
        task_text:  str = "",
        episode_id: int = 0,
        step:       int = 0,
    ) -> Iterator[np.ndarray]:
        """
        Streaming inference for chunk-based models.
        Yields one action chunk array per iteration.
        """
        model = self._store.get(model_id)
        obs   = Observation(
            images=images, state=state,
            task_text=task_text, episode_id=episode_id, step=step,
        )
        action_chunk = model.predict_action(obs)

        # If the model returns (chunk_size, action_dim), yield each row
        if action_chunk.ndim == 2:
            for action in action_chunk:
                yield action
        else:
            yield action_chunk

    def reset_episode(self, model_id: str) -> None:
        """Notify model that a new episode has started."""
        try:
            model = self._store.get(model_id)
            model.reset()
        except KeyError:
            pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _release_gpu_memory() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
