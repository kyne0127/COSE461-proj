"""
module.server.grpc_server
==========================
gRPC 서버 진입점 (GPU 서버에서 실행).

proto에서 생성된 스텁과 Servicer를 연결하고 서버를 시작합니다.
InferenceService + TrainingService + HealthService를 단일 포트로 서빙.

실행:
    python -m module.server.grpc_server --config config/server.yaml
    # 또는
    python scripts/run_server.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent import futures
from typing import Any, Dict, Iterator, Optional

import numpy as np

from module.server.inference_service import InferenceServicer, ModelStore
from module.server.train_service import TrainingServicer
from module.server.generic_service import GenericServicer, HandlerStore
from module.utils.tensor_utils import proto_to_numpy
from module.utils.logging import get_logger, setup_logging

logger = get_logger("server.grpc_server")


# ────────────────────────────────────────────────────────────────────────────
# gRPC stub adapter helpers (thin bridge: proto ↔ servicer)
# ────────────────────────────────────────────────────────────────────────────

def _load_grpc_stubs():
    proto_dir = os.path.join(os.path.dirname(__file__), "..", "proto")
    abs_proto  = os.path.abspath(proto_dir)
    if abs_proto not in sys.path:
        sys.path.insert(0, abs_proto)
    import lerobot_pb2      as pb2
    import lerobot_pb2_grpc as pb2_grpc
    return pb2, pb2_grpc


# ────────────────────────────────────────────────────────────────────────────
# gRPC Servicer wrappers (proto-aware)
# ────────────────────────────────────────────────────────────────────────────

def _make_inference_servicer_class(pb2, pb2_grpc, inference_svc: InferenceServicer):
    """Dynamically create a proto-bound InferenceServicer class."""

    class _InferenceServicer(pb2_grpc.InferenceServiceServicer):

        def LoadModel(self, request, context):
            result = inference_svc.load_model(
                model_type=request.model_type,
                model_id=request.model_id,
                checkpoint_path=request.checkpoint_path,
                config_yaml=request.config_yaml,
            )
            return pb2.LoadModelResponse(**result)

        def UnloadModel(self, request, context):
            result = inference_svc.unload_model(request.model_id)
            return pb2.UnloadModelResponse(**result)

        def ListModels(self, request, context):
            models = inference_svc.list_models()
            return pb2.ListModelsResponse(
                models=[pb2.ModelInfo(**m) for m in models]
            )

        def GetAction(self, request, context):
            # Decode images
            images: Dict[str, np.ndarray] = {}
            for img_proto in request.images:
                arr = proto_to_numpy(img_proto)
                images[img_proto.name or f"cam_{len(images)}"] = arr

            # Decode state
            state = proto_to_numpy(request.state) if request.state.data else None

            action = inference_svc.get_action(
                model_id=request.model_id,
                images=images,
                state=state,
                task_text=request.task_text,
                episode_id=request.episode_id,
                step=request.step,
            )
            action_flat = action.flatten().tolist()
            return pb2.ActionResponse(
                action=pb2.Tensor(
                    shape=list(action.shape),
                    data=action_flat,
                    dtype="float32",
                ),
                timestamp=time.time(),
            )

        def GetActionStream(self, request, context):
            images: Dict[str, np.ndarray] = {}
            for img_proto in request.images:
                arr = proto_to_numpy(img_proto)
                images[img_proto.name or f"cam_{len(images)}"] = arr
            state = proto_to_numpy(request.state) if request.state.data else None

            for action in inference_svc.get_action_stream(
                model_id=request.model_id,
                images=images,
                state=state,
                task_text=request.task_text,
                episode_id=request.episode_id,
                step=request.step,
            ):
                action_proto = pb2.Tensor(
                    shape=list(action.shape),
                    data=action.flatten().tolist(),
                    dtype="float32",
                )
                yield pb2.ActionChunk(
                    actions=[action_proto],
                    chunk_size=1,
                    timestamp=time.time(),
                )

    return _InferenceServicer


def _make_training_servicer_class(pb2, pb2_grpc, training_svc: TrainingServicer):

    class _TrainingServicer(pb2_grpc.TrainingServiceServicer):

        def StreamEpisode(self, request_iterator, context):
            frames = []
            for req in request_iterator:
                image_arrays = [proto_to_numpy(img) for img in req.images]
                image_names  = [img.name or f"cam_{i}" for i, img in enumerate(req.images)]
                frames.append({
                    "dataset_id":  req.dataset_id,
                    "episode_idx": req.episode_idx,
                    "frame_idx":   req.frame_idx,
                    "images":      image_arrays,
                    "image_names": image_names,
                    "state":       list(proto_to_numpy(req.state)) if req.state.data else [],
                    "action":      list(proto_to_numpy(req.action)) if req.action.data else [],
                    "done":        req.done,
                    "reward":      req.reward,
                })
            result = training_svc.stream_episode(iter(frames))
            return pb2.EpisodeUploadResponse(**result)

        def StartTraining(self, request, context):
            result = training_svc.start_training(
                model_type=request.model_type,
                dataset_id=request.dataset_id,
                config_yaml=request.config_yaml,
                output_dir=request.output_dir,
                resume_from=request.resume_from,
            )
            return pb2.TrainingJobResponse(**result)

        def GetTrainingStatus(self, request, context):
            status = training_svc.get_status(request.job_id)
            return pb2.TrainingStatusResponse(**status)

        def StreamTrainingLogs(self, request, context):
            for log in training_svc.stream_logs(request.job_id):
                yield pb2.TrainingLog(
                    job_id=log["job_id"],
                    level=log["level"],
                    message=log["message"],
                    timestamp=log["timestamp"],
                )

        def StopTraining(self, request, context):
            result = training_svc.stop_training(request.job_id)
            return pb2.TrainingJobResponse(**result)

        def ListJobs(self, request, context):
            jobs = training_svc.list_jobs(request.status_filter)
            return pb2.ListJobsResponse(
                jobs=[pb2.TrainingStatusResponse(**j) for j in jobs]
            )

    return _TrainingServicer


def _make_generic_servicer_class(pb2, pb2_grpc, generic_svc: GenericServicer):

    class _GenericServicer(pb2_grpc.GenericInferenceServiceServicer):

        def Infer(self, request, context):
            tensors = [
                np.array(t.data, dtype=np.float32).reshape(t.shape)
                for t in request.tensors
            ]
            result = generic_svc.infer(
                handler_id=request.handler_id,
                method=request.method,
                payload_json=request.payload_json,
                tensors=tensors,
                session_id=request.session_id,
            )
            return pb2.GenericResponse(
                success=result["success"],
                payload_json=result["payload_json"],
                error=result["error"],
                session_id=result["session_id"],
            )

        def InferStream(self, request, context):
            tensors = [
                np.array(t.data, dtype=np.float32).reshape(t.shape)
                for t in request.tensors
            ]
            for chunk in generic_svc.infer_stream(
                handler_id=request.handler_id,
                method=request.method,
                payload_json=request.payload_json,
                tensors=tensors,
                session_id=request.session_id,
            ):
                yield pb2.GenericResponse(
                    success=chunk["success"],
                    payload_json=chunk["payload_json"],
                    error=chunk["error"],
                    session_id=chunk["session_id"],
                )

    return _GenericServicer


def _make_health_servicer_class(pb2, pb2_grpc):

    class _HealthServicer(pb2_grpc.HealthServiceServicer):
        def Ping(self, request, context):
            gpu_info = ""
            gpu_mem  = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    dev = torch.cuda.current_device()
                    gpu_info = torch.cuda.get_device_name(dev)
                    gpu_mem  = torch.cuda.mem_get_info(dev)[1] / 1e9
            except Exception:
                pass
            return pb2.PingResponse(
                ok=True,
                server_id="lerobot-gpu-server",
                gpu_info=gpu_info,
                gpu_mem_gb=gpu_mem,
                timestamp=time.time(),
            )

    return _HealthServicer


# ────────────────────────────────────────────────────────────────────────────
# Server factory
# ────────────────────────────────────────────────────────────────────────────

class LeRobotGRPCServer:
    """
    All-in-one gRPC server hosting Inference + Training + Health services.

    Usage:
        server = LeRobotGRPCServer.from_config(config)
        server.start()
        server.wait_for_termination()
    """

    def __init__(
        self,
        host:        str  = "0.0.0.0",
        port:        int  = 50051,
        max_workers: int  = 10,
        data_root:   str  = "/data/lerobot",
        ckpt_root:   str  = "/checkpoints",
        train_workers: int = 2,
        preload_handlers: list = None,
    ) -> None:
        self._host             = host
        self._port             = port
        self._max_workers      = max_workers
        self._data_root        = data_root
        self._ckpt_root        = ckpt_root
        self._train_workers    = train_workers
        self._preload_handlers = preload_handlers or []
        self._server           = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LeRobotGRPCServer":
        srv = config.get("server", {})
        return cls(
            host=srv.get("host", "0.0.0.0"),
            port=srv.get("port", 50051),
            max_workers=srv.get("max_workers", 10),
            data_root=config.get("data_root", "/data/lerobot"),
            ckpt_root=config.get("checkpoint_root", "/checkpoints"),
            train_workers=srv.get("train_workers", 2),
            preload_handlers=config.get("handlers", []),
        )

    def start(self) -> None:
        import grpc

        pb2, pb2_grpc = _load_grpc_stubs()

        # Build servicers
        store          = ModelStore()
        inference_svc  = InferenceServicer(store=store)
        training_svc   = TrainingServicer(
            data_root=self._data_root,
            ckpt_root=self._ckpt_root,
            max_workers=self._train_workers,
        )
        handler_store  = HandlerStore()
        generic_svc    = GenericServicer(store=handler_store)

        InferenceCls = _make_inference_servicer_class(pb2, pb2_grpc, inference_svc)
        TrainingCls  = _make_training_servicer_class(pb2, pb2_grpc, training_svc)
        GenericCls   = _make_generic_servicer_class(pb2, pb2_grpc, generic_svc)
        HealthCls    = _make_health_servicer_class(pb2, pb2_grpc)

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers),
            options=[
                ("grpc.max_send_message_length",    256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )

        pb2_grpc.add_InferenceServiceServicer_to_server(InferenceCls(), self._server)
        pb2_grpc.add_TrainingServiceServicer_to_server(TrainingCls(), self._server)
        pb2_grpc.add_GenericInferenceServiceServicer_to_server(GenericCls(), self._server)
        pb2_grpc.add_HealthServiceServicer_to_server(HealthCls(), self._server)

        # ── 서버 시작 전 핸들러 프리로드 ────────────────────────────────────
        for h in self._preload_handlers:
            handler_id = h.get("handler_id", "")
            h_config   = h.get("config", {})
            if not handler_id:
                continue
            logger.info("Pre-loading handler '%s' ...", handler_id)
            result = generic_svc.load_handler(handler_id, config=h_config)
            if result["success"]:
                logger.info("Handler '%s' loaded: %s", handler_id, result["message"])
            else:
                logger.error("Handler '%s' failed to load: %s", handler_id, result["message"])

        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)
        self._server.start()
        logger.info("LeRobot gRPC server listening on %s", address)

    def stop(self, grace: float = 5.0) -> None:
        if self._server:
            self._server.stop(grace)
            logger.info("Server stopped")

    def wait_for_termination(self) -> None:
        if self._server:
            self._server.wait_for_termination()


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="LeRobot gRPC Server")
    parser.add_argument("--config", default="config/server.yaml")
    parser.add_argument("--host",   default=None)
    parser.add_argument("--port",   type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level, component="lerobot.server")

    config: Dict[str, Any] = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    if args.host:
        config.setdefault("server", {})["host"] = args.host
    if args.port:
        config.setdefault("server", {})["port"] = args.port

    server = LeRobotGRPCServer.from_config(config)
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down ...")
        server.stop()


if __name__ == "__main__":
    main()
