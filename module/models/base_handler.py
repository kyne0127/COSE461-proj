"""
module.models.base_handler
==========================
GenericInferenceService용 핸들러 추상 기반 클래스.

새 baseline을 서버에 올리려면 BaseHandler를 상속받아
handle() 메서드를 구현하고 @HandlerRegistry.register("name")을
붙이면 됩니다.

예시 (AmbRes):
    @HandlerRegistry.register("ambres")
    class AmbResHandler(BaseHandler):
        def setup(self, config): ...
        def handle(self, method, payload, tensors, session_id): ...
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Handler Registry
# ────────────────────────────────────────────────────────────────────────────

class HandlerRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """클래스 데코레이터: @HandlerRegistry.register("ambres")"""
        def decorator(handler_cls):
            if name in cls._registry:
                logger.warning("HandlerRegistry: '%s' already registered, overwriting", name)
            cls._registry[name] = handler_cls
            logger.debug("HandlerRegistry: registered '%s' → %s", name, handler_cls.__name__)
            return handler_cls
        return decorator

    @classmethod
    def build(cls, name: str, config: Optional[Dict[str, Any]] = None) -> "BaseHandler":
        if name not in cls._registry:
            raise KeyError(
                f"Handler '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        handler = cls._registry[name]()
        handler.setup(config or {})
        return handler

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def list_handlers(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def auto_discover(cls) -> None:
        """
        module/models/ 아래의 모든 패키지를 import해서
        @HandlerRegistry.register 데코레이터가 실행되도록 합니다.
        """
        import importlib
        import pkgutil
        import module.models as models_pkg

        for finder, mod_name, _ in pkgutil.walk_packages(
            path=models_pkg.__path__,
            prefix=models_pkg.__name__ + ".",
        ):
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                logger.debug("auto_discover: skipping %s (%s)", mod_name, e)


# ────────────────────────────────────────────────────────────────────────────
# Abstract Base Handler
# ────────────────────────────────────────────────────────────────────────────

class BaseHandler(ABC):
    """
    GenericInferenceService에 연결되는 핸들러 기반 클래스.

    서브클래스는 두 메서드를 구현합니다:
      - setup(config)  : 모델 로드 등 초기화 (서버 시작 시 1회 호출)
      - handle(...)    : 실제 추론 로직 (매 요청마다 호출)

    스트리밍이 필요하면 handle_stream()도 오버라이드합니다.
    """

    def setup(self, config: Dict[str, Any]) -> None:
        """초기화. 모델 로드, GPU 할당 등. 기본 구현은 no-op."""
        pass

    @abstractmethod
    def handle(
        self,
        method:     str,
        payload:    Dict[str, Any],
        tensors:    List[np.ndarray],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        단일 요청 처리.

        Args:
            method:     핸들러 내 메서드 이름 (e.g. "query", "respond", "reset")
            payload:    JSON에서 역직렬화된 dict (문자열 파라미터 등)
            tensors:    이미지 등 바이너리 텐서 리스트 (np.ndarray)
            session_id: 멀티턴 세션 ID (빈 문자열이면 stateless)

        Returns:
            JSON 직렬화 가능한 dict
        """
        ...

    def handle_stream(
        self,
        method:     str,
        payload:    Dict[str, Any],
        tensors:    List[np.ndarray],
        session_id: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        스트리밍 응답. 기본 구현은 handle()을 단일 청크로 래핑합니다.
        대용량 응답이나 점진적 출력이 필요하면 오버라이드하세요.
        """
        yield self.handle(method, payload, tensors, session_id)

    # ------------------------------------------------------------------ #
    # 편의 헬퍼
    # ------------------------------------------------------------------ #

    @staticmethod
    def ok(data: Dict[str, Any]) -> Dict[str, Any]:
        """성공 응답 dict 생성."""
        return {"success": True, **data}

    @staticmethod
    def err(message: str) -> Dict[str, Any]:
        """실패 응답 dict 생성."""
        return {"success": False, "error": message}

    @staticmethod
    def dumps(data: Dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def loads(json_str: str) -> Dict[str, Any]:
        if not json_str:
            return {}
        return json.loads(json_str)
