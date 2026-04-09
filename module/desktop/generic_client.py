"""
module.desktop.generic_client
==============================
GenericInferenceService 클라이언트 (로컬에서 실행).

baseline 평가 스크립트에서 모델 추론 부분만 서버로 위임할 때 사용합니다.

사용법:
    from module.desktop.generic_client import GenericClient

    client = GenericClient(host="<runpod-ip>", port=50051)
    client.connect()

    # 핸들러 로드 (서버에서 모델 초기화)
    client.load_handler("ambres", config={"device": "cuda"})

    # 추론 요청
    result = client.infer(
        handler_id="ambres",
        method="query",
        payload={"task_description": "Put the banana in the drawer."},
        images={"cam": img_array},       # np.ndarray (H, W, C)
        session_id="sess_001",
    )
    # result: {"task_objects": [...], "task_ambiguous": True, ...}

    client.disconnect()
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, Iterator, List, Optional

import numpy as np

from module.utils.tensor_utils import numpy_to_proto
from module.utils.logging import get_logger

logger = get_logger("desktop.generic_client")


def _load_stubs():
    try:
        import sys
        import os
        proto_dir = os.path.join(os.path.dirname(__file__), "..", "proto")
        if proto_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(proto_dir))
        import lerobot_pb2      as pb2
        import lerobot_pb2_grpc as pb2_grpc
        return pb2, pb2_grpc
    except ImportError:
        raise ImportError(
            "gRPC stubs not found. Generate them with:\n"
            "  python -m grpc_tools.protoc -I module/proto "
            "--python_out=module/proto --grpc_python_out=module/proto "
            "module/proto/lerobot.proto"
        )


def _make_channel(host: str, port: int, use_tls: bool = False):
    import grpc
    address = f"{host}:{port}"
    opts = [
        ("grpc.max_send_message_length",    256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.keepalive_time_ms",          10_000),
        ("grpc.keepalive_timeout_ms",        5_000),
    ]
    if use_tls:
        return __import__("grpc").secure_channel(address, __import__("grpc").ssl_channel_credentials(), options=opts)
    return __import__("grpc").insecure_channel(address, options=opts)


def _images_to_protos(pb2, images: Dict[str, np.ndarray]) -> list:
    protos = []
    for name, arr in images.items():
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        td = numpy_to_proto(arr, name=name)
        protos.append(pb2.Tensor(
            shape=td["shape"], data=td["data"],
            dtype=td["dtype"], name=td["name"],
        ))
    return protos


# ────────────────────────────────────────────────────────────────────────────
# GenericClient
# ────────────────────────────────────────────────────────────────────────────

class GenericClient:
    """
    범용 baseline 추론 클라이언트.

    baseline 평가 스크립트는 이 클라이언트를 통해
    무거운 모델 추론(Molmo, SAM2, LLM 등)을 GPU 서버로 위임합니다.
    """

    def __init__(
        self,
        host:    str   = "localhost",
        port:    int   = 50051,
        use_tls: bool  = False,
        timeout: float = 60.0,
    ) -> None:
        self._host    = host
        self._port    = port
        self._use_tls = use_tls
        self._timeout = timeout
        self._channel = None
        self._stub    = None
        self._pb2     = None

    # ------------------------------------------------------------------ #
    # 연결
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        pb2, pb2_grpc = _load_stubs()
        self._pb2     = pb2
        self._channel = _make_channel(self._host, self._port, self._use_tls)
        self._stub    = pb2_grpc.GenericInferenceServiceStub(self._channel)
        logger.info("GenericClient connected to %s:%d", self._host, self._port)

    def disconnect(self) -> None:
        if self._channel:
            self._channel.close()
            logger.info("GenericClient disconnected")

    # ------------------------------------------------------------------ #
    # 핸들러 관리 (서버 측 모델 로드/언로드)
    # 핸들러 관리는 InferenceService의 LoadModel과 별도로
    # payload_json을 통해 컨벤션으로 처리합니다.
    # ------------------------------------------------------------------ #

    def load_handler(
        self,
        handler_id: str,
        config:     Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        서버에 핸들러를 로드하도록 요청합니다.
        내부적으로 method="__load__" 컨벤션을 사용합니다.
        """
        result = self.infer(
            handler_id=handler_id,
            method="__load__",
            payload=config or {},
        )
        success = result.get("success", False)
        if success:
            logger.info("Handler '%s' loaded on server", handler_id)
        else:
            logger.error("Failed to load handler '%s': %s", handler_id, result.get("error", ""))
        return success

    # ------------------------------------------------------------------ #
    # 추론
    # ------------------------------------------------------------------ #

    def infer(
        self,
        handler_id: str,
        method:     str,
        payload:    Optional[Dict[str, Any]] = None,
        images:     Optional[Dict[str, np.ndarray]] = None,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """
        단일 추론 요청.

        Args:
            handler_id: 서버에 등록된 핸들러 이름 (e.g. "ambres")
            method:     핸들러 메서드 (e.g. "query", "respond", "reset")
            payload:    JSON 직렬화 가능한 파라미터 dict
            images:     이미지 dict (cam_name → np.ndarray HWC)
            session_id: 멀티턴 세션 ID

        Returns:
            서버 응답 dict (payload_json 역직렬화)
        """
        pb2          = self._pb2
        tensor_protos = _images_to_protos(pb2, images or {})
        payload_json  = json.dumps(payload or {}, ensure_ascii=False)

        req = pb2.GenericRequest(
            handler_id=handler_id,
            method=method,
            payload_json=payload_json,
            tensors=tensor_protos,
            session_id=session_id,
        )
        resp = self._stub.Infer(req, timeout=self._timeout)

        if not resp.success:
            raise RuntimeError(
                f"GenericInfer [{handler_id}.{method}] failed: {resp.error}"
            )

        return json.loads(resp.payload_json) if resp.payload_json else {}

    def infer_stream(
        self,
        handler_id: str,
        method:     str,
        payload:    Optional[Dict[str, Any]] = None,
        images:     Optional[Dict[str, np.ndarray]] = None,
        session_id: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        스트리밍 추론 요청. 청크 단위로 결과를 yield합니다.
        """
        pb2           = self._pb2
        tensor_protos = _images_to_protos(pb2, images or {})
        payload_json  = json.dumps(payload or {}, ensure_ascii=False)

        req = pb2.GenericRequest(
            handler_id=handler_id,
            method=method,
            payload_json=payload_json,
            tensors=tensor_protos,
            session_id=session_id,
        )
        for resp in self._stub.InferStream(req, timeout=self._timeout):
            if not resp.success:
                raise RuntimeError(
                    f"GenericInferStream [{handler_id}.{method}] error: {resp.error}"
                )
            yield json.loads(resp.payload_json) if resp.payload_json else {}

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
