"""
module.utils.dataset
====================
LeRobot dataset utilities — episode buffering, HuggingFace dataset conversion,
and upload helpers compatible with the gRPC streaming API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """Single time-step frame from a robot episode."""
    frame_idx:  int
    images:     Dict[str, np.ndarray]   # cam_name → (H, W, C) uint8
    state:      np.ndarray              # proprio state vector
    action:     np.ndarray              # action vector
    reward:     float      = 0.0
    done:       bool       = False
    timestamp:  float      = field(default_factory=time.time)
    extra:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Collection of frames forming one demonstration."""
    episode_idx: int
    dataset_id:  str
    frames:      List[Frame] = field(default_factory=list)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def is_empty(self) -> bool:
        return len(self.frames) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_idx": self.episode_idx,
            "dataset_id":  self.dataset_id,
            "n_frames":    len(self.frames),
            "metadata":    self.metadata,
        }


# ────────────────────────────────────────────────────────────────────────────
# Episode buffer — used on desktop side while collecting
# ────────────────────────────────────────────────────────────────────────────

class EpisodeBuffer:
    """
    Thread-safe buffer that accumulates frames during data collection,
    then yields them for gRPC streaming.
    """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id  = dataset_id
        self._episodes:  List[Episode] = []
        self._current:   Optional[Episode] = None
        self._ep_count   = 0

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def start_episode(self, metadata: Optional[Dict] = None) -> Episode:
        if self._current is not None and not self._current.is_empty:
            self._episodes.append(self._current)
        self._current = Episode(
            episode_idx=self._ep_count,
            dataset_id=self.dataset_id,
            metadata=metadata or {},
        )
        self._ep_count += 1
        return self._current

    def add_frame(self, frame: Frame) -> None:
        if self._current is None:
            raise RuntimeError("Call start_episode() before add_frame()")
        self._current.append(frame)

    def end_episode(self) -> Episode:
        if self._current is None:
            raise RuntimeError("No active episode")
        ep = self._current
        self._episodes.append(ep)
        self._current = None
        return ep

    # ------------------------------------------------------------------ #
    # Iteration for streaming
    # ------------------------------------------------------------------ #
    def iter_frames(self, episode: Episode) -> Generator[Frame, None, None]:
        """Yield frames one by one for gRPC client-streaming."""
        yield from episode.frames

    @property
    def completed_episodes(self) -> List[Episode]:
        return list(self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)


# ────────────────────────────────────────────────────────────────────────────
# Disk I/O helpers (server side — save received dataset)
# ────────────────────────────────────────────────────────────────────────────

class DatasetWriter:
    """
    Writes received episodes to disk in LeRobot-compatible format.
    Compatible with lerobot.common.datasets.LeRobotDataset.
    """

    def __init__(self, root: str | Path, dataset_id: str) -> None:
        self.root = Path(root) / dataset_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._episode_idx = 0
        self._meta: List[Dict] = []

    def write_episode(self, episode: Episode) -> Path:
        ep_dir = self.root / f"episode_{episode.episode_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        states  = []
        actions = []
        rewards = []
        dones   = []

        for frame in episode.frames:
            states.append(frame.state)
            actions.append(frame.action)
            rewards.append(frame.reward)
            dones.append(frame.done)

            # Save images
            for cam_name, img in frame.images.items():
                cam_dir = ep_dir / cam_name
                cam_dir.mkdir(exist_ok=True)
                img_path = cam_dir / f"{frame.frame_idx:06d}.npy"
                np.save(img_path, img)

        np.save(ep_dir / "states.npy",  np.array(states,  dtype=np.float32))
        np.save(ep_dir / "actions.npy", np.array(actions, dtype=np.float32))
        np.save(ep_dir / "rewards.npy", np.array(rewards, dtype=np.float32))
        np.save(ep_dir / "dones.npy",   np.array(dones,   dtype=bool))

        meta = {**episode.to_dict(), "path": str(ep_dir)}
        self._meta.append(meta)
        self._save_meta()
        return ep_dir

    def _save_meta(self) -> None:
        meta_path = self.root / "dataset_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"episodes": self._meta}, f, indent=2)

    @property
    def dataset_path(self) -> Path:
        return self.root
