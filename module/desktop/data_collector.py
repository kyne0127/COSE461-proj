"""
module.desktop.data_collector
==============================
텔레오퍼레이션 기반 에피소드 수집기 (데스크탑 전용).

- 리더-팔로워 텔레오퍼레이션으로 시연(demonstration) 녹화
- 에피소드를 EpisodeBuffer에 저장
- gRPC 클라이언트를 통해 서버로 스트리밍 업로드
- 키보드 인터럽트 / 시간 제한 / 수동 종료 지원
"""

from __future__ import annotations

import select
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from module.desktop.robot_connector import RobotConnector, RawObservation
from module.utils.dataset import EpisodeBuffer, Frame, Episode
from module.utils.logging import get_logger

logger = get_logger("desktop.data_collector")


# ────────────────────────────────────────────────────────────────────────────
# Collection result
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CollectionStats:
    n_episodes:    int   = 0
    n_frames:      int   = 0
    elapsed_secs:  float = 0.0
    aborted:       bool  = False


# ────────────────────────────────────────────────────────────────────────────
# Data Collector
# ────────────────────────────────────────────────────────────────────────────

class DataCollector:
    """
    High-level controller for collecting robot demonstrations.

    Usage:
        collector = DataCollector(
            robot=connector,
            buffer=episode_buffer,
            fps=30,
            max_episode_steps=500,
        )
        stats = collector.collect_episodes(n_episodes=10)
    """

    def __init__(
        self,
        robot:             RobotConnector,
        buffer:            EpisodeBuffer,
        fps:               float = 30.0,
        max_episode_steps: int   = 500,
        warmup_steps:      int   = 10,
        on_frame:          Optional[Callable[[Frame], None]] = None,
        on_episode_end:    Optional[Callable[[Episode], None]] = None,
    ) -> None:
        self._robot             = robot
        self._buffer            = buffer
        self._fps               = fps
        self._max_steps         = max_episode_steps
        self._warmup_steps      = warmup_steps
        self._dt                = 1.0 / fps
        self._on_frame          = on_frame
        self._on_episode_end    = on_episode_end
        self._stop_event        = threading.Event()

        # Register SIGINT for graceful stop
        signal.signal(signal.SIGINT, self._handle_sigint)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stdin_has_input() -> bool:
        """Non-blocking check: True if ENTER was pressed on stdin."""
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(rlist)

    def collect_episodes(
        self,
        n_episodes:     int,
        task_text:      str = "",
        metadata:       Optional[Dict[str, Any]] = None,
        interactive:    bool = False,
    ) -> CollectionStats:
        """
        Collect `n_episodes` teleoperation demonstrations.

        Args:
            n_episodes: Number of episodes to record
            task_text:  Language description of the task (for VLA models)
            metadata:   Extra metadata stored with each episode

        Returns:
            CollectionStats with summary information
        """
        stats = CollectionStats()
        t_start = time.time()

        logger.info("Starting collection: %d episodes @ %.0f fps", n_episodes, self._fps)

        for ep_idx in range(n_episodes):
            if self._stop_event.is_set():
                logger.info("Collection stopped by user after %d episodes", ep_idx)
                stats.aborted = True
                break

            # Interactive mode: wait for ENTER before starting each episode
            if interactive:
                print(f"\n  [Episode {ep_idx + 1}/{n_episodes}] 초기 자세로 설정 후 ENTER ▶  (q+ENTER = 전체 종료)")
                try:
                    line = sys.stdin.readline()
                except EOFError:
                    break
                if line.strip().lower() == 'q':
                    break
                print(f"  ▶ 수집 중...  ENTER = 이 에피소드 종료  |  Ctrl+C = 전체 중단")

            logger.info("▶ Episode %d / %d  (task: '%s')", ep_idx + 1, n_episodes, task_text)
            ep_meta = {**(metadata or {}), "task": task_text, "ep_idx": ep_idx}

            try:
                episode = self._collect_one_episode(ep_meta, interactive=interactive)
                stats.n_episodes += 1
                stats.n_frames   += len(episode)
                logger.info(
                    "✓ Episode %d done — %d frames", ep_idx + 1, len(episode)
                )
                if interactive:
                    print(f"  ✓ Episode {ep_idx + 1} 완료 — {len(episode)} frames 저장됨")

                if self._on_episode_end:
                    self._on_episode_end(episode)

            except KeyboardInterrupt:
                logger.info("Interrupted during episode %d", ep_idx + 1)
                stats.aborted = True
                break
            except ConnectionError as e:
                logger.error(
                    "Episode %d — robot connection lost, skipping: %s", ep_idx + 1, e
                )
                print(
                    f"\n[collect] ⚠ Episode {ep_idx + 1} skipped (connection error). "
                    "Check USB/power cable on the arm.\n"
                )
                # Discard the partial episode from the buffer
                try:
                    self._buffer.end_episode()
                except Exception:
                    pass
            except Exception as e:
                logger.error("Episode %d failed: %s", ep_idx + 1, e)
                raise

        stats.elapsed_secs = time.time() - t_start
        logger.info(
            "Collection complete — %d eps / %d frames / %.1f s",
            stats.n_episodes, stats.n_frames, stats.elapsed_secs,
        )
        return stats

    def collect_one_episode_interactive(
        self, task_text: str = "", metadata: Optional[Dict] = None
    ) -> Episode:
        """Collect a single episode with interactive confirmation."""
        input("Press ENTER to start episode, Ctrl+C to abort...")
        return self._collect_one_episode(metadata or {"task": task_text})

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    def _collect_one_episode(self, metadata: Dict[str, Any], interactive: bool = False) -> Episode:
        """
        Inner loop for one teleoperation episode.
        Runs at self._fps until max_steps is reached or stop is signalled.
        In interactive mode, ENTER key ends the episode early.
        """
        episode = self._buffer.start_episode(metadata=metadata)
        step    = 0

        # Warmup — skip first N frames to let camera auto-exposure settle
        if self._warmup_steps > 0:
            logger.debug("Warmup: skipping %d steps", self._warmup_steps)
            for _ in range(self._warmup_steps):
                self._robot.teleop_step()

        while step < self._max_steps and not self._stop_event.is_set():
            # Interactive: ENTER ends this episode (consume the newline)
            if interactive and self._stdin_has_input():
                sys.stdin.readline()
                logger.info("Episode ended by user at step %d", step)
                break

            t0 = time.perf_counter()

            # Collect one step
            obs, action = self._robot.teleop_step()

            frame = Frame(
                frame_idx=step,
                images=obs.images,
                state=obs.state,
                action=action,
                timestamp=obs.timestamp,
            )
            self._buffer.add_frame(frame)

            if self._on_frame:
                self._on_frame(frame)

            step += 1

            # Rate limiting
            elapsed = time.perf_counter() - t0
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        return self._buffer.end_episode()

    # ------------------------------------------------------------------ #
    # Signal handling
    # ------------------------------------------------------------------ #

    def _handle_sigint(self, signum, frame) -> None:
        logger.info("Received SIGINT — finishing current episode then stopping")
        self._stop_event.set()

    def stop(self) -> None:
        """Programmatically stop collection after current episode."""
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Replay (for debugging)
    # ------------------------------------------------------------------ #

    def replay_episode(self, episode: Episode, speed: float = 1.0) -> None:
        """
        Replay a collected episode by sending saved actions back to the robot.
        Useful for verifying recorded demonstrations.

        Args:
            episode: Episode to replay
            speed:   Playback speed multiplier (1.0 = real-time)
        """
        logger.info("Replaying episode %d (%d frames)", episode.episode_idx, len(episode))
        dt = self._dt / speed

        for frame in episode.frames:
            t0 = time.perf_counter()
            self._robot.send_action(frame.action)
            elapsed = time.perf_counter() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        logger.info("Replay complete")
