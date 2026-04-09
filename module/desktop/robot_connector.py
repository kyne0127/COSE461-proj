"""
module.desktop.robot_connector
===============================
LeRobot 하드웨어 연결 레이어 (데스크탑 전용).

RTX 3060 데스크탑에서 로봇 하드웨어(Koch v1.1, SO-100 등)를
직접 연결하고 제어하는 모듈.

- 로봇 초기화 / 연결 / 해제
- 관측(observation) 수집: 카메라 + 고유감각(state)
- 행동(action) 실행
- 텔레오퍼레이션 지원 (수동 시연 수집용)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from module.utils.logging import get_logger

logger = get_logger("desktop.robot_connector")


# ────────────────────────────────────────────────────────────────────────────
# Robot types supported by LeRobot
# ────────────────────────────────────────────────────────────────────────────

SUPPORTED_ROBOTS = {
    "koch":        "lerobot.common.robot_devices.robots.koch.KochRobot",
    "koch_bimanual": "lerobot.common.robot_devices.robots.koch_bimanual.KochBimanualRobot",
    "so100":       "lerobot.common.robot_devices.robots.so100.So100Robot",
    "moss":        "lerobot.common.robot_devices.robots.moss.MossRobot",
    "lekiwi":      "lerobot.common.robot_devices.robots.lekiwi.LeKiwiRobot",
    "aloha":       "lerobot.common.robot_devices.robots.aloha.AlohaRobot",
}


# ────────────────────────────────────────────────────────────────────────────
# Observation dataclass (desktop-native, no proto dependency)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RawObservation:
    """Raw sensor data snapshot from the robot."""
    images:     Dict[str, np.ndarray]  # cam_name → (H, W, C) uint8
    state:      np.ndarray             # joint positions / velocities
    timestamp:  float = field(default_factory=time.time)
    step:       int   = 0
    extra:      Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Main connector
# ────────────────────────────────────────────────────────────────────────────

class RobotConnector:
    """
    Thin wrapper around a LeRobot robot device.

    Usage:
        connector = RobotConnector.from_config(config["robot"])
        connector.connect()

        obs = connector.get_observation()
        connector.send_action(action_array)

        connector.disconnect()
    """

    def __init__(
        self,
        robot_type:    str,
        robot_config:  Dict[str, Any],
        fps:           float = 30.0,
    ) -> None:
        self._robot_type   = robot_type
        self._robot_config = robot_config
        self._fps          = fps
        self._robot        = None
        self._connected    = False
        self._step_count   = 0
        self._dt           = 1.0 / fps

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RobotConnector":
        return cls(
            robot_type=config["robot_type"],
            robot_config=config.get("robot_config", {}),
            fps=config.get("fps", 30.0),
        )

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Initialize and connect to the robot hardware."""
        logger.info("Connecting to robot type='%s'", self._robot_type)
        self._robot = self._build_robot()

        try:
            self._robot.connect()
            self._connected = True
            logger.info("Robot connected successfully")
        except Exception as e:
            logger.error("Failed to connect to robot: %s", e)
            raise

    def disconnect(self) -> None:
        """Safely disconnect from the robot."""
        if self._robot is not None and self._connected:
            try:
                self._robot.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
            finally:
                self._connected = False

    @contextmanager
    def session(self) -> Generator[None, None, None]:
        """Context manager for robot session."""
        self.connect()
        try:
            yield
        finally:
            self.disconnect()

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def get_observation(self) -> RawObservation:
        """
        Capture one observation from all cameras and joints.
        Returns a RawObservation with images and state vector.
        """
        self._check_connected()
        obs_dict = self._robot.capture_observation()

        images: Dict[str, np.ndarray] = {}
        state:  Optional[np.ndarray]  = None

        for key, value in obs_dict.items():
            if "image" in key or "camera" in key:
                # LeRobot returns (C, H, W) float32 — convert to (H, W, C) uint8
                if isinstance(value, np.ndarray):
                    if value.shape[0] in (1, 3, 4) and value.ndim == 3:
                        value = (value.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                cam_name = key.split(".")[-1]
                images[cam_name] = value
            elif "observation.state" in key or "state" in key:
                if isinstance(value, np.ndarray):
                    state = value.astype(np.float32)

        if state is None:
            state = np.zeros(1, dtype=np.float32)

        self._step_count += 1
        return RawObservation(
            images=images,
            state=state,
            step=self._step_count,
        )

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def send_action(self, action: np.ndarray) -> None:
        """
        Send action to the robot.
        action: (action_dim,) float32 array in normalized [-1, 1] range
                (LeRobot handles denormalization internally)
        """
        self._check_connected()
        import torch
        action_tensor = torch.from_numpy(action.astype(np.float32))
        self._robot.send_action(action_tensor)

    # ------------------------------------------------------------------ #
    # Teleoperation (for data collection)
    # ------------------------------------------------------------------ #

    def get_teleop_action(self) -> np.ndarray:
        """
        Read current leader arm position for teleoperation.
        Returns: (action_dim,) float32 array
        """
        self._check_connected()
        if not hasattr(self._robot, "teleop_step"):
            raise NotImplementedError(
                f"Robot type '{self._robot_type}' does not support teleoperation"
            )
        obs, action = self._robot.teleop_step(record_data=True)
        return action["action"].numpy().astype(np.float32)

    def teleop_step(self) -> tuple[RawObservation, np.ndarray]:
        """Single teleoperation step: returns (observation, action)."""
        self._check_connected()
        obs_dict, action_dict = self._robot.teleop_step(record_data=True)

        images: Dict[str, np.ndarray] = {}
        state: np.ndarray = np.zeros(1, dtype=np.float32)

        for key, value in obs_dict.items():
            if "image" in key or "camera" in key:
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    if value.shape[0] in (1, 3, 4):
                        value = (value.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                images[key.split(".")[-1]] = value
            elif "state" in key:
                state = value.numpy().astype(np.float32) if hasattr(value, "numpy") else np.array(value, dtype=np.float32)

        self._step_count += 1
        obs = RawObservation(images=images, state=state, step=self._step_count)
        action = action_dict["action"].numpy().astype(np.float32)
        return obs, action

    # ------------------------------------------------------------------ #
    # Robot info
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def robot_type(self) -> str:
        return self._robot_type

    def get_info(self) -> Dict[str, Any]:
        return {
            "robot_type": self._robot_type,
            "connected":  self._connected,
            "fps":        self._fps,
            "step_count": self._step_count,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_robot(self):
        """Dynamically import and instantiate the robot class."""
        import importlib

        if self._robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unknown robot type '{self._robot_type}'. "
                f"Supported: {list(SUPPORTED_ROBOTS.keys())}"
            )

        dotted = SUPPORTED_ROBOTS[self._robot_type]
        module_path, _, class_name = dotted.rpartition(".")

        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not load robot class '{dotted}'. "
                f"Make sure lerobot is installed.\n{e}"
            ) from e

        return cls(**self._robot_config)

    def _check_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

    def __repr__(self) -> str:
        return (
            f"RobotConnector(type={self._robot_type}, "
            f"connected={self._connected}, fps={self._fps})"
        )
