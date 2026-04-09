"""
module.desktop.grpc_client
===========================
gRPC 클라이언트 (데스크탑 → GPU 서버).

두 가지 클라이언트 제공:
  1. InferenceClient — 서버에서 실시간 action 예측 요청
  2. TrainingClient  — 에피소드 업로드 + 학습 트리거 + 상태 모니터링

protobuf stubs은 proto/lerobot.proto를 컴파일해서 생성:
    python -m grpc_tools.protoc -I module/proto \
        --python_out=module/proto \
        --grpc_python_out=module/proto \
        module/proto/lerobot.proto
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, Iterable, Iterator, Optional

import numpy as np

from module.utils.dataset import Episode, Frame
from module.utils.tensor_utils import numpy_to_proto
from module.utils.logging import get_logger

logger = get_logger("desktop.grpc_client")


# ────────────────────────────────────────────────────────────────────────────
# gRPC stub lazy loader — avoids hard dependency before proto compile
# ────────────────────────────────────────────────────────────────────────────

def _load_stubs():
    """Lazy import of generated gRPC stubs."""
    try:
        import sys, os
        # Add proto dir so generated stubs are importable
        proto_dir = os.path.join(os.path.dirname(__file__), "..", "proto")
        if proto_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(proto_dir))

        import lerobot_pb2       as pb2
        import lerobot_pb2_grpc  as pb2_grpc
        return pb2, pb2_grpc
    except ImportError:
        raise ImportError(
            "gRPC stubs not found. Generate them with:\n"
            "  pip install grpcio-tools\n"
            "  python -m grpc_tools.protoc -I module/proto "
            "--python_out=module/proto --grpc_python_out=module/proto "
            "module/proto/lerobot.proto"
        )


# ────────────────────────────────────────────────────────────────────────────
# Shared channel factory
# ────────────────────────────────────────────────────────────────────────────

def _make_channel(host: str, port: int, use_tls: bool = False):
    import grpc
    address = f"{host}:{port}"
    if use_tls:
        creds = grpc.ssl_channel_credentials()
        return grpc.secure_channel(address, creds)
    return grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length",    256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.keepalive_time_ms",          10_000),
            ("grpc.keepalive_timeout_ms",        5_000),
        ],
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Inference Client
# ────────────────────────────────────────────────────────────────────────────

class InferenceClient:
    """
    Real-time action prediction client.

    Usage:
        client = InferenceClient(host="192.168.1.100", port=50051)
        client.connect()

        # Load model on server
        client.load_model("act", model_id="run_001",
                          checkpoint_path="/checkpoints/act_run001")

        # Per-step inference
        action = client.get_action(
            model_id="run_001",
            images={"top": img_array},
            state=state_array,
        )

        client.disconnect()
    """

    def __init__(
        self,
        host:    str   = "localhost",
        port:    int   = 50051,
        use_tls: bool  = False,
        timeout: float = 5.0,
    ) -> None:
        self._host    = host
        self._port    = port
        self._use_tls = use_tls
        self._timeout = timeout
        self._channel = None
        self._stub    = None

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        pb2, pb2_grpc = _load_stubs()
        self._pb2      = pb2
        self._channel  = _make_channel(self._host, self._port, self._use_tls)
        self._stub     = pb2_grpc.InferenceServiceStub(self._channel)
        self._health   = pb2_grpc.HealthServiceStub(self._channel)
        logger.info("InferenceClient connected to %s:%d", self._host, self._port)

    def disconnect(self) -> None:
        if self._channel:
            self._channel.close()
            logger.info("InferenceClient disconnected")

    def ping(self) -> Dict[str, Any]:
        resp = self._health.Ping(
            self._pb2.PingRequest(client_id="desktop"),
            timeout=self._timeout,
        )
        return {
            "ok":         resp.ok,
            "server_id":  resp.server_id,
            "gpu_info":   resp.gpu_info,
            "gpu_mem_gb": resp.gpu_mem_gb,
        }

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model_type:      str,
        model_id:        str,
        checkpoint_path: str,
        config_yaml:     str = "",
    ) -> bool:
        req = self._pb2.LoadModelRequest(
            model_type=model_type,
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            config_yaml=config_yaml,
        )
        resp = self._stub.LoadModel(req, timeout=30.0)
        logger.info("LoadModel('%s'): %s — %s", model_id, resp.success, resp.message)
        return resp.success

    def unload_model(self, model_id: str) -> bool:
        resp = self._stub.UnloadModel(
            self._pb2.UnloadModelRequest(model_id=model_id),
            timeout=self._timeout,
        )
        return resp.success

    def list_models(self) -> list[Dict[str, str]]:
        resp = self._stub.ListModels(
            self._pb2.ListModelsRequest(), timeout=self._timeout
        )
        return [
            {"model_id": m.model_id, "model_type": m.model_type, "status": m.status}
            for m in resp.models
        ]

    # ------------------------------------------------------------------ #
    # Action prediction
    # ------------------------------------------------------------------ #

    def get_action(
        self,
        model_id:   str,
        images:     Dict[str, np.ndarray],
        state:      Optional[np.ndarray] = None,
        task_text:  str  = "",
        episode_id: int  = 0,
        step:       int  = 0,
    ) -> np.ndarray:
        """
        Send observation to server → receive predicted action.

        Args:
            model_id:   ID of the model to use (must be loaded)
            images:     dict of cam_name → (H, W, C) uint8 or float32 [0,1]
            state:      proprioceptive state vector
            task_text:  language description (for Pi0/VLA)
            episode_id: current episode index
            step:       current step index

        Returns:
            (action_dim,) float32 numpy array
        """
        pb2 = self._pb2

        # Build image tensors
        image_protos = []
        for cam_name, img in images.items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            td = numpy_to_proto(img, name=cam_name)
            image_protos.append(pb2.Tensor(
                shape=td["shape"], data=td["data"],
                dtype=td["dtype"], name=td["name"],
            ))

        # Build state tensor
        state_proto = None
        if state is not None:
            td = numpy_to_proto(state.astype(np.float32))
            state_proto = pb2.Tensor(
                shape=td["shape"], data=td["data"], dtype=td["dtype"]
            )

        req = pb2.ObservationRequest(
            model_id=model_id,
            images=image_protos,
            state=state_proto,
            task_text=task_text,
            episode_id=episode_id,
            step=step,
        )

        t0 = time.perf_counter()
        resp = self._stub.GetAction(req, timeout=self._timeout)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.debug("GetAction latency: %.1f ms", latency_ms)

        action = np.array(resp.action.data, dtype=np.float32)
        action = action.reshape(resp.action.shape)
        return action

    def get_action_stream(
        self,
        model_id:   str,
        images:     Dict[str, np.ndarray],
        state:      Optional[np.ndarray] = None,
        task_text:  str = "",
        episode_id: int = 0,
        step:       int = 0,
    ) -> Generator[np.ndarray, None, None]:
        """
        Streaming version: yields action chunks from ACT-style policies.
        Each yielded value is (chunk_size, action_dim).
        """
        pb2 = self._pb2
        image_protos = []
        for cam_name, img in images.items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            td = numpy_to_proto(img, name=cam_name)
            image_protos.append(pb2.Tensor(
                shape=td["shape"], data=td["data"],
                dtype=td["dtype"], name=td["name"],
            ))

        req = pb2.ObservationRequest(
            model_id=model_id, images=image_protos,
            task_text=task_text, episode_id=episode_id, step=step,
        )
        for chunk in self._stub.GetActionStream(req, timeout=30.0):
            for action_proto in chunk.actions:
                a = np.array(action_proto.data, dtype=np.float32).reshape(action_proto.shape)
                yield a

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ────────────────────────────────────────────────────────────────────────────
# 2. Training Client
# ────────────────────────────────────────────────────────────────────────────

class TrainingClient:
    """
    Dataset upload + training trigger client.

    Usage:
        client = TrainingClient(host="192.168.1.100", port=50051)
        client.connect()

        # Upload collected episodes
        for episode in buffer.completed_episodes:
            client.stream_episode(episode)

        # Start training
        job_id = client.start_training(
            model_type="act",
            dataset_id="my_dataset",
            config_yaml=open("config/models/act.yaml").read(),
        )

        # Monitor progress
        for log in client.stream_logs(job_id):
            print(log)
    """

    def __init__(
        self,
        host:    str  = "localhost",
        port:    int  = 50051,
        use_tls: bool = False,
    ) -> None:
        self._host    = host
        self._port    = port
        self._use_tls = use_tls
        self._channel = None
        self._stub    = None

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        pb2, pb2_grpc = _load_stubs()
        self._pb2     = pb2
        self._channel = _make_channel(self._host, self._port, self._use_tls)
        self._stub    = pb2_grpc.TrainingServiceStub(self._channel)
        logger.info("TrainingClient connected to %s:%d", self._host, self._port)

    def disconnect(self) -> None:
        if self._channel:
            self._channel.close()

    # ------------------------------------------------------------------ #
    # Episode upload
    # ------------------------------------------------------------------ #

    def stream_episode(self, episode: Episode) -> bool:
        """
        Stream a single episode frame-by-frame to the server.

        Returns True if upload succeeded.
        """
        pb2 = self._pb2

        def frame_generator() -> Iterator:
            for frame in episode.frames:
                image_protos = []
                for cam_name, img in frame.images.items():
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    td = numpy_to_proto(img, name=cam_name)
                    image_protos.append(pb2.Tensor(
                        shape=td["shape"], data=td["data"],
                        dtype=td["dtype"], name=td["name"],
                    ))

                state_td  = numpy_to_proto(frame.state)
                action_td = numpy_to_proto(frame.action)

                yield pb2.EpisodeFrame(
                    dataset_id=episode.dataset_id,
                    episode_idx=episode.episode_idx,
                    frame_idx=frame.frame_idx,
                    images=image_protos,
                    state=pb2.Tensor(**state_td),
                    action=pb2.Tensor(**action_td),
                    done=frame.done,
                    reward=frame.reward,
                )

        resp = self._stub.StreamEpisode(frame_generator(), timeout=300.0)
        logger.info(
            "Episode %d uploaded — %d frames → dataset '%s' [%s]",
            episode.episode_idx, resp.total_frames,
            resp.dataset_id, "OK" if resp.success else "FAILED",
        )
        return resp.success

    def upload_all_episodes(self, episodes: list[Episode]) -> int:
        """Upload multiple episodes. Returns count of successful uploads."""
        success = 0
        for i, ep in enumerate(episodes):
            logger.info("Uploading episode %d/%d ...", i + 1, len(episodes))
            if self.stream_episode(ep):
                success += 1
        logger.info("%d/%d episodes uploaded successfully", success, len(episodes))
        return success

    # ------------------------------------------------------------------ #
    # Training control
    # ------------------------------------------------------------------ #

    def start_training(
        self,
        model_type:  str,
        dataset_id:  str,
        config_yaml: str,
        output_dir:  str = "/checkpoints",
        resume_from: str = "",
    ) -> str:
        """
        Trigger training on the server.
        Returns job_id string for monitoring.
        """
        req = self._pb2.TrainingRequest(
            model_type=model_type,
            dataset_id=dataset_id,
            config_yaml=config_yaml,
            output_dir=output_dir,
            resume_from=resume_from,
        )
        resp = self._stub.StartTraining(req, timeout=10.0)
        if not resp.success:
            raise RuntimeError(f"StartTraining failed: {resp.message}")
        logger.info("Training job started — id=%s", resp.job_id)
        return resp.job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        req  = self._pb2.TrainingStatusRequest(job_id=job_id)
        resp = self._stub.GetTrainingStatus(req, timeout=5.0)
        return {
            "job_id":       resp.job_id,
            "status":       resp.status,
            "epoch":        resp.epoch,
            "total_epochs": resp.total_epochs,
            "loss":         resp.loss,
            "checkpoint":   resp.checkpoint,
            "elapsed_secs": resp.elapsed_secs,
        }

    def stop_training(self, job_id: str) -> bool:
        resp = self._stub.StopTraining(
            self._pb2.TrainingStatusRequest(job_id=job_id), timeout=5.0
        )
        return resp.success

    def stream_logs(self, job_id: str) -> Generator[str, None, None]:
        """Yield training log lines until training completes."""
        req = self._pb2.TrainingStatusRequest(job_id=job_id)
        for log in self._stub.StreamTrainingLogs(req, timeout=86400):
            yield f"[{log.level}] {log.message}"

    def list_jobs(self, status_filter: str = "") -> list[Dict[str, Any]]:
        resp = self._stub.ListJobs(
            self._pb2.ListJobsRequest(status_filter=status_filter),
            timeout=5.0,
        )
        return [
            {"job_id": j.job_id, "status": j.status, "loss": j.loss}
            for j in resp.jobs
        ]

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
