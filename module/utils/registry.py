"""
module.utils.registry
=====================
Plug-and-play model registry.

Usage (server side):
    from module.utils.registry import ModelRegistry

    # Auto-discover all built-in models
    ModelRegistry.auto_discover()

    # Get a model class by name
    cls = ModelRegistry.get("act")
    model = cls(config)

    # Register a custom model
    @ModelRegistry.register("my_model")
    class MyModel(BaseLeRobotModel):
        ...
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from module.models.base_model import BaseLeRobotModel

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Built-in model locations (model_type → dotted import path)
# ────────────────────────────────────────────────────────────────────────────
_BUILTIN_MODELS: Dict[str, str] = {
    "act":       "module.models.act.model.ACTModel",
    "diffusion": "module.models.diffusion.model.DiffusionPolicyModel",
    "tdmpc2":    "module.models.tdmpc2.model.TDMPC2Model",
    "pi0":       "module.models.pi0.model.Pi0Model",
    "custom":    "module.models.custom.model.CustomModel",
}


class ModelRegistry:
    """Central registry for LeRobot model plug-ins."""

    _registry: Dict[str, Type["BaseLeRobotModel"]] = {}

    # ------------------------------------------------------------------ #
    # Decorator
    # ------------------------------------------------------------------ #
    @classmethod
    def register(cls, name: str):
        """Decorator: @ModelRegistry.register("act")"""
        def _wrap(model_cls):
            cls._register(name, model_cls)
            return model_cls
        return _wrap

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _register(cls, name: str, model_cls: Type["BaseLeRobotModel"]) -> None:
        name = name.lower()
        if name in cls._registry:
            logger.warning("ModelRegistry: overwriting existing key '%s'", name)
        cls._registry[name] = model_cls
        logger.debug("ModelRegistry: registered '%s' → %s", name, model_cls)

    # ------------------------------------------------------------------ #
    # Auto-discovery
    # ------------------------------------------------------------------ #
    @classmethod
    def auto_discover(cls) -> None:
        """Import all built-in model modules so their @register decorators run."""
        for name, dotted_path in _BUILTIN_MODELS.items():
            module_path, _, _ = dotted_path.rpartition(".")
            try:
                importlib.import_module(module_path)
                logger.info("ModelRegistry: discovered '%s'", name)
            except ImportError as exc:
                logger.warning(
                    "ModelRegistry: could not import '%s' (%s). "
                    "Ensure optional dependencies are installed.",
                    name, exc,
                )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, name: str) -> Type["BaseLeRobotModel"]:
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def build(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "BaseLeRobotModel":
        """Convenience: get class + instantiate."""
        model_cls = cls.get(name)
        return model_cls(config=config or {}, **kwargs)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name.lower() in cls._registry

    def __repr__(self) -> str:
        return f"ModelRegistry({list(self._registry.keys())})"
