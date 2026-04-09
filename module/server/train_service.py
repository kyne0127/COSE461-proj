"""
module.server.train_service
============================
gRPC TrainingService 서버 구현 (GPU 서버 전용).

- 에피소드 스트리밍 수신 → DatasetWriter로 디스크 저장
- 학습 잡 큐 관리 (ThreadPool 기반)
- 실시간 학습 로그 스트리밍 (gRPC 서버 스트리밍)
- 체크포인트 자동 저장 및 학습 재개
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

import numpy as np

from module.models.base_model import BaseLeRobotModel, TrainingBatch
from module.utils.dataset import DatasetWriter, Episode, Frame
from module.utils.registry import ModelRegistry
from module.utils.logging import get_logger

logger = get_logger("server.train_service")


# ────────────────────────────────────────────────────────────────────────────
# Job status enum
# ────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    STOPPED = "stopped"


# ────────────────────────────────────────────────────────────────────────────
# Training job descriptor
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingJob:
    job_id:       str
    model_type:   str
    dataset_id:   str
    config:       Dict[str, Any]
    output_dir:   str
    resume_from:  str = ""

    status:       JobStatus = JobStatus.QUEUED
    epoch:        int       = 0
    total_epochs: int       = 0
    loss:         float     = float("nan")
    checkpoint:   str       = ""
    elapsed_secs: float     = 0.0
    message:      str       = ""
    started_at:   float     = field(default_factory=time.time)

    # Queue for log streaming
    log_queue:    queue.Queue = field(default_factory=queue.Queue)
    stop_flag:    threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id":       self.job_id,
            "status":       self.status.value,
            "epoch":        self.epoch,
            "total_epochs": self.total_epochs,
            "loss":         self.loss,
            "checkpoint":   self.checkpoint,
            "elapsed_secs": self.elapsed_secs,
            "message":      self.message,
        }

    def log(self, level: str, message: str) -> None:
        self.log_queue.put({
            "job_id":    self.job_id,
            "level":     level,
            "message":   message,
            "timestamp": time.time(),
        })
        getattr(logger, level.lower(), logger.info)("[%s] %s", self.job_id[:8], message)


# ────────────────────────────────────────────────────────────────────────────
# TrainingServicer
# ────────────────────────────────────────────────────────────────────────────

class TrainingServicer:
    """
    gRPC TrainingService implementation.

    Episode handling (stream_episode) is synchronous.
    Training jobs run in background threads.
    """

    def __init__(
        self,
        data_root:   str = "/data/lerobot",
        ckpt_root:   str = "/checkpoints",
        max_workers: int = 2,
    ) -> None:
        self._data_root  = Path(data_root)
        self._ckpt_root  = Path(ckpt_root)
        self._data_root.mkdir(parents=True, exist_ok=True)
        self._ckpt_root.mkdir(parents=True, exist_ok=True)

        self._jobs:       Dict[str, TrainingJob] = {}
        self._jobs_lock   = threading.RLock()
        self._job_queue:  queue.Queue = queue.Queue()

        # Dataset writers (one per dataset_id)
        self._writers:    Dict[str, DatasetWriter] = {}
        self._writer_lock = threading.RLock()

        # Worker pool
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"trainer-{i}")
            for i in range(max_workers)
        ]
        for w in self._workers:
            w.start()

        logger.info("TrainingServicer started (data=%s, ckpt=%s, workers=%d)",
                    data_root, ckpt_root, max_workers)

    # ------------------------------------------------------------------ #
    # Episode ingestion
    # ------------------------------------------------------------------ #

    def stream_episode(self, frames: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Receive a stream of EpisodeFrame dicts and write to disk.

        Args:
            frames: Iterable of frame dicts (from gRPC adapter)

        Returns:
            Dict with success, dataset_id, episode_idx, total_frames
        """
        first = True
        writer: Optional[DatasetWriter] = None
        episode: Optional[Episode] = None
        total_frames = 0

        for frame_dict in frames:
            dataset_id  = frame_dict["dataset_id"]
            episode_idx = frame_dict["episode_idx"]
            frame_idx   = frame_dict["frame_idx"]

            if first:
                writer  = self._get_writer(dataset_id)
                episode = Episode(
                    episode_idx=episode_idx,
                    dataset_id=dataset_id,
                )
                first = False

            # Convert image arrays
            images: Dict[str, np.ndarray] = {}
            for img_arr, img_name in zip(
                frame_dict.get("images", []),
                frame_dict.get("image_names", []),
            ):
                if isinstance(img_arr, np.ndarray):
                    images[img_name] = img_arr
                elif isinstance(img_arr, list):
                    images[img_name] = np.array(img_arr, dtype=np.float32)

            frame = Frame(
                frame_idx=frame_idx,
                images=images,
                state=np.array(frame_dict.get("state", []), dtype=np.float32),
                action=np.array(frame_dict.get("action", []), dtype=np.float32),
                reward=float(frame_dict.get("reward", 0.0)),
                done=bool(frame_dict.get("done", False)),
            )
            episode.append(frame)
            total_frames += 1

        if episode is None or writer is None:
            return {"success": False, "message": "No frames received"}

        writer.write_episode(episode)
        logger.info("Episode %d saved — %d frames — dataset '%s'",
                    episode.episode_idx, total_frames, dataset_id)

        return {
            "success":      True,
            "dataset_id":   dataset_id,
            "episode_idx":  episode.episode_idx,
            "total_frames": total_frames,
        }

    # ------------------------------------------------------------------ #
    # Training control
    # ------------------------------------------------------------------ #

    def start_training(
        self,
        model_type:  str,
        dataset_id:  str,
        config_yaml: str,
        output_dir:  str = "",
        resume_from: str = "",
    ) -> Dict[str, Any]:
        """Queue a training job. Returns immediately with job_id."""
        import yaml

        try:
            config = yaml.safe_load(config_yaml) or {}
        except Exception as e:
            return {"success": False, "job_id": "", "message": f"Invalid config YAML: {e}"}

        job_id    = str(uuid.uuid4())[:12]
        out_dir   = output_dir or str(self._ckpt_root / model_type / job_id)
        total_eps = config.get("num_epochs", 100)

        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            dataset_id=dataset_id,
            config=config,
            output_dir=out_dir,
            resume_from=resume_from,
            total_epochs=total_eps,
        )

        with self._jobs_lock:
            self._jobs[job_id] = job

        self._job_queue.put(job)
        logger.info("Training job %s queued (model=%s, dataset=%s)",
                    job_id, model_type, dataset_id)

        return {"success": True, "job_id": job_id, "message": "queued"}

    def get_status(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"job_id": job_id, "status": "not_found"}
        return job.to_dict()

    def stop_training(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"success": False, "message": "Job not found"}
        job.stop_flag.set()
        job.status = JobStatus.STOPPED
        return {"success": True, "message": "stop requested"}

    def stream_logs(self, job_id: str) -> Generator[Dict[str, Any], None, None]:
        """Generator that yields log dicts until training finishes."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            try:
                log = job.log_queue.get(timeout=1.0)
                yield log
            except queue.Empty:
                continue

        # Drain remaining logs
        while not job.log_queue.empty():
            try:
                yield job.log_queue.get_nowait()
            except queue.Empty:
                break

    def list_jobs(self, status_filter: str = "") -> List[Dict[str, Any]]:
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status.value == status_filter]
        return [j.to_dict() for j in jobs]

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        """Background thread that processes training jobs."""
        while True:
            job: TrainingJob = self._job_queue.get()
            self._run_job(job)

    def _run_job(self, job: TrainingJob) -> None:
        """Execute one training job end-to-end."""
        job.status = JobStatus.RUNNING
        t_start    = time.time()
        job.log("INFO", f"Training started — model={job.model_type} dataset={job.dataset_id}")

        try:
            # ── Load dataset ──────────────────────────────────────────────
            job.log("INFO", "Loading dataset ...")
            dataloader = self._build_dataloader(job)

            # ── Build model ───────────────────────────────────────────────
            job.log("INFO", f"Building model '{job.model_type}' ...")
            if not ModelRegistry.is_registered(job.model_type):
                ModelRegistry.auto_discover()

            cfg = dict(job.config)
            cfg.setdefault("device", "cuda" if self._gpu_available() else "cpu")
            model = ModelRegistry.build(job.model_type, config=cfg)

            ckpt = job.resume_from or self._find_latest_checkpoint(job)
            if ckpt:
                job.log("INFO", f"Resuming from {ckpt}")
                model.load_checkpoint(ckpt)
            else:
                # Load from scratch — create dummy checkpoint dir so load succeeds
                dummy_dir = Path(job.output_dir) / "init"
                dummy_dir.mkdir(parents=True, exist_ok=True)
                try:
                    model.load_checkpoint(str(dummy_dir))
                except Exception:
                    pass  # Random init is fine

            # ── Training loop ─────────────────────────────────────────────
            num_epochs   = job.config.get("num_epochs", 100)
            save_every   = job.config.get("save_every_epochs", 10)
            log_every    = job.config.get("log_every_steps", 50)
            job.total_epochs = num_epochs

            for epoch in range(1, num_epochs + 1):
                if job.stop_flag.is_set():
                    job.log("INFO", "Stop requested — exiting training loop")
                    break

                epoch_losses = []
                for step, batch in enumerate(dataloader):
                    if job.stop_flag.is_set():
                        break
                    loss_dict = model.train_step(batch)
                    total_loss = sum(loss_dict.values())
                    epoch_losses.append(total_loss)

                    if step % log_every == 0:
                        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                        job.log("INFO", f"epoch={epoch}/{num_epochs} step={step}  {loss_str}")

                job.epoch = epoch
                job.loss  = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                job.elapsed_secs = time.time() - t_start

                # Checkpoint
                if epoch % save_every == 0 or epoch == num_epochs:
                    ckpt_path = Path(job.output_dir) / f"epoch_{epoch:04d}"
                    model.save_checkpoint(str(ckpt_path))
                    job.checkpoint = str(ckpt_path)
                    job.log("INFO", f"Checkpoint saved → {ckpt_path}")

            job.status  = JobStatus.DONE
            job.message = "Training complete"
            job.log("INFO", f"✓ Done — final loss={job.loss:.4f}")

        except Exception as e:
            job.status  = JobStatus.FAILED
            job.message = str(e)
            job.log("ERROR", f"Training failed: {e}")
            logger.exception("Job %s failed", job.job_id)

    # ------------------------------------------------------------------ #
    # Dataset loading helpers
    # ------------------------------------------------------------------ #

    def _build_dataloader(self, job: TrainingJob):
        """
        Build a PyTorch DataLoader from the received dataset.
        Falls back to a simple numpy-based loader if torch is unavailable.
        """
        dataset_path = self._data_root / job.dataset_id
        batch_size   = job.config.get("batch_size", 32)

        try:
            return self._build_torch_dataloader(dataset_path, batch_size, job)
        except ImportError:
            logger.warning("torch not available — using numpy loader")
            return self._build_numpy_loader(dataset_path, batch_size)

    def _build_torch_dataloader(self, dataset_path: Path, batch_size: int, job: TrainingJob):
        import torch
        from torch.utils.data import Dataset, DataLoader

        class LeRobotLocalDataset(Dataset):
            def __init__(self, root: Path):
                self.root = root
                meta_path = root / "dataset_meta.json"
                if not meta_path.exists():
                    raise FileNotFoundError(f"Dataset not found: {meta_path}")
                with open(meta_path) as f:
                    meta = json.load(f)
                self._episodes = meta["episodes"]

                # Pre-load all frames
                self._states: List[np.ndarray]  = []
                self._actions: List[np.ndarray] = []
                for ep in self._episodes:
                    ep_dir = Path(ep["path"])
                    if (ep_dir / "states.npy").exists():
                        self._states.append(np.load(ep_dir / "states.npy"))
                        self._actions.append(np.load(ep_dir / "actions.npy"))

                if self._states:
                    self._all_states  = np.concatenate(self._states,  axis=0)
                    self._all_actions = np.concatenate(self._actions, axis=0)
                else:
                    self._all_states  = np.zeros((1, 1), dtype=np.float32)
                    self._all_actions = np.zeros((1, 1), dtype=np.float32)

            def __len__(self):
                return len(self._all_states)

            def __getitem__(self, idx):
                return {
                    "states":  torch.from_numpy(self._all_states[idx]),
                    "actions": torch.from_numpy(self._all_actions[idx]),
                }

        dataset = LeRobotLocalDataset(dataset_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=job.config.get("num_workers", 4),
            pin_memory=True,
        )

    def _build_numpy_loader(self, dataset_path: Path, batch_size: int):
        """Fallback loader using only numpy."""
        class NumpyLoader:
            def __init__(self, root: Path, bs: int):
                states_files  = sorted(root.rglob("states.npy"))
                actions_files = sorted(root.rglob("actions.npy"))
                if not states_files:
                    self._states  = np.zeros((bs, 1), dtype=np.float32)
                    self._actions = np.zeros((bs, 1), dtype=np.float32)
                else:
                    self._states  = np.concatenate([np.load(f) for f in states_files])
                    self._actions = np.concatenate([np.load(f) for f in actions_files])
                self._bs = bs

            def __iter__(self):
                n = len(self._states)
                idx = np.random.permutation(n)
                for start in range(0, n, self._bs):
                    b = idx[start:start + self._bs]
                    yield TrainingBatch(
                        states=self._states[b],
                        actions=self._actions[b],
                    )

        return NumpyLoader(dataset_path, batch_size)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_writer(self, dataset_id: str) -> DatasetWriter:
        with self._writer_lock:
            if dataset_id not in self._writers:
                self._writers[dataset_id] = DatasetWriter(
                    root=str(self._data_root), dataset_id=dataset_id
                )
            return self._writers[dataset_id]

    def _find_latest_checkpoint(self, job: TrainingJob) -> Optional[str]:
        out_dir = Path(job.output_dir)
        if not out_dir.exists():
            return None
        checkpoints = sorted(out_dir.glob("epoch_*"))
        return str(checkpoints[-1]) if checkpoints else None

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
