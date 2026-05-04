"""
module.server.generic_service
==============================
GenericInferenceService 서버 구현 (GPU 서버 전용).

핸들러 ID 기반으로 요청을 라우팅합니다.
새 baseline 추가 시 BaseHandler 구현체를 등록하기만 하면 됩니다.

흐름:
    클라이언트 → GenericRequest(handler_id, method, payload_json, tensors)
              → HandlerRegistry.get(handler_id).handle(method, payload, tensors, session_id)
              → GenericResponse(success, payload_json)
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from module.models.base_handler import BaseHandler, HandlerRegistry
from module.utils.logging import get_logger

logger = get_logger("server.generic_service")


# ────────────────────────────────────────────────────────────────────────────
# Handler Store  (thread-safe)
# ────────────────────────────────────────────────────────────────────────────

class HandlerStore:
    """로드된 핸들러 인스턴스를 handler_id로 관리."""

    def __init__(self) -> None:
        self._handlers: Dict[str, BaseHandler] = {}
        self._lock = threading.RLock()

    def get(self, handler_id: str) -> BaseHandler:
        with self._lock:
            if handler_id not in self._handlers:
                raise KeyError(f"Handler '{handler_id}' is not loaded")
            return self._handlers[handler_id]

    def load(self, handler_id: str, config: Dict[str, Any]) -> None:
        """핸들러를 빌드하고 store에 등록."""
        if not HandlerRegistry.is_registered(handler_id):
            HandlerRegistry.auto_discover()

        handler = HandlerRegistry.build(handler_id, config=config)
        with self._lock:
            self._handlers[handler_id] = handler
        logger.info("Handler '%s' loaded", handler_id)

    def unload(self, handler_id: str) -> None:
        with self._lock:
            self._handlers.pop(handler_id, None)
        logger.info("Handler '%s' unloaded", handler_id)

    def list_all(self) -> List[str]:
        with self._lock:
            return list(self._handlers.keys())


# ────────────────────────────────────────────────────────────────────────────
# GenericServicer
# ────────────────────────────────────────────────────────────────────────────

class GenericServicer:
    """
    gRPC GenericInferenceService 구현.
    proto-stub과 독립적으로 동작하며, grpc_server.py에서 래핑됩니다.
    """

    def __init__(self, store: Optional[HandlerStore] = None) -> None:
        self._store = store or HandlerStore()

    # ------------------------------------------------------------------ #
    # 핸들러 관리 (로컬 호출용 — gRPC 래퍼가 없어도 직접 쓸 수 있음)
    # ------------------------------------------------------------------ #

    def load_handler(self, handler_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._store.load(handler_id, config)
            return {"success": True, "message": f"Handler '{handler_id}' loaded"}
        except Exception as e:
            logger.error("Failed to load handler '%s': %s", handler_id, e)
            return {"success": False, "message": str(e)}

    def unload_handler(self, handler_id: str) -> Dict[str, Any]:
        self._store.unload(handler_id)
        return {"success": True}

    def list_handlers(self) -> List[str]:
        return self._store.list_all()

    # ------------------------------------------------------------------ #
    # 추론
    # ------------------------------------------------------------------ #

    def infer(
        self,
        handler_id:   str,
        method:       str,
        payload_json: str,
        tensors:      List[np.ndarray],
        session_id:   str = "",
    ) -> Dict[str, Any]:
        """
        단일 요청 처리.

        method="__load__" 는 예약어: 핸들러를 store에 로드/재초기화합니다.
        payload_json에 핸들러 config dict를 담아 보내면 됩니다.

        Returns:
            {"success": bool, "payload_json": str, "error": str, "session_id": str}
        """
        # ── 핸들러 로드 요청 (__load__ 예약 메서드) ───────────────────────
        if method == "__load__":
            config = json.loads(payload_json) if payload_json else {}
            result = self.load_handler(handler_id, config=config)
            return {
                "success":      result["success"],
                "payload_json": json.dumps(result, ensure_ascii=False),
                "error":        result.get("message", "") if not result["success"] else "",
                "session_id":   session_id,
            }

        try:
            handler = self._store.get(handler_id)
            payload = json.loads(payload_json) if payload_json else {}
            result  = handler.handle(method, payload, tensors, session_id)
            return {
                "success":      True,
                "payload_json": json.dumps(result, ensure_ascii=False),
                "error":        "",
                "session_id":   session_id,
            }
        except KeyError as e:
            msg = str(e)
            logger.warning("infer: handler not found — %s", msg)
            return {"success": False, "payload_json": "", "error": msg, "session_id": session_id}
        except Exception as e:
            logger.error("infer error [%s.%s]: %s", handler_id, method, e, exc_info=True)
            return {"success": False, "payload_json": "", "error": str(e), "session_id": session_id}

    def infer_stream(
        self,
        handler_id:   str,
        method:       str,
        payload_json: str,
        tensors:      List[np.ndarray],
        session_id:   str = "",
    ) -> Iterator[Dict[str, Any]]:
        """
        스트리밍 요청 처리. 각 청크를 dict로 yield합니다.
        """
        try:
            handler = self._store.get(handler_id)
            payload = json.loads(payload_json) if payload_json else {}
            for chunk in handler.handle_stream(method, payload, tensors, session_id):
                yield {
                    "success":      True,
                    "payload_json": json.dumps(chunk, ensure_ascii=False),
                    "error":        "",
                    "session_id":   session_id,
                }
        except Exception as e:
            logger.error("infer_stream error [%s.%s]: %s", handler_id, method, e, exc_info=True)
            yield {"success": False, "payload_json": "", "error": str(e), "session_id": session_id}
