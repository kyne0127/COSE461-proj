"""
module.desktop.robot_connector
===============================
LeRobot 0.5.x 하드웨어 연결 레이어 (데스크탑 전용).

RTX 3060 데스크탑에서 SO-ARM100/101 등 로봇 하드웨어를
직접 연결하고 제어하는 모듈.

lerobot 0.5.x API 기반:
  - Follower(robot): lerobot.robots.so_follower.SOFollower
  - Leader(teleop):  lerobot.teleoperators.so_leader.SOLeader
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
# Supported robot types → (follower path, leader path)
# lerobot 0.5.x 기준
# ────────────────────────────────────────────────────────────────────────────

SUPPORTED_ROBOTS = {
    "so100": {
        "follower_module": "lerobot.robots.so_follower.so_follower",
        "follower_class":  "SOFollower",
        "follower_config_module": "lerobot.robots.so_follower.config_so_follower",
        "follower_config_class":  "SOFollowerRobotConfig",
        "leader_module": "lerobot.teleoperators.so_leader.so_leader",
        "leader_class":  "SOLeader",
        "leader_config_module": "lerobot.teleoperators.so_leader.config_so_leader",
        "leader_config_class":  "SOLeaderTeleopConfig",
        "robot_id": "so100_follower",
        "teleop_id": "so100_leader",
    },
    "so101": {
        "follower_module": "lerobot.robots.so_follower.so_follower",
        "follower_class":  "SOFollower",
        "follower_config_module": "lerobot.robots.so_follower.config_so_follower",
        "follower_config_class":  "SOFollowerRobotConfig",
        "leader_module": "lerobot.teleoperators.so_leader.so_leader",
        "leader_class":  "SOLeader",
        "leader_config_module": "lerobot.teleoperators.so_leader.config_so_leader",
        "leader_config_class":  "SOLeaderTeleopConfig",
        "robot_id": "so101_follower",
        "teleop_id": "so101_leader",
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Observation dataclass (desktop-native, no proto dependency)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RawObservation:
    """Raw sensor data snapshot from the robot."""
    images:    Dict[str, np.ndarray]  # cam_name → (H, W, C) uint8
    state:     np.ndarray             # joint positions (degrees or normalized)
    state_keys: List[str] = field(default_factory=list)  # motor key names
    timestamp: float = field(default_factory=time.time)
    step:      int   = 0
    extra:     Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Main connector
# ────────────────────────────────────────────────────────────────────────────

class RobotConnector:
    """
    Wrapper around lerobot 0.5.x SOFollower + SOLeader.

    desktop.yaml 기준 robot_config 포맷:
        robot_type: "so101"
        robot_config:
          follower_arms:
            main:
              port: "/dev/ttyUSB1"
          leader_arms:
            main:
              port: "/dev/ttyUSB0"
          cameras:
            top:
              camera_index: 0
              fps: 30
              width: 640
              height: 480

    Usage:
        connector = RobotConnector.from_config(config["robot"])
        connector.connect()
        obs = connector.get_observation()
        connector.send_action(action_array, state_keys)
        connector.disconnect()
    """

    def __init__(
        self,
        robot_type:   str,
        robot_config: Dict[str, Any],
        fps:          float = 30.0,
    ) -> None:
        if robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unknown robot type '{robot_type}'. "
                f"Supported: {list(SUPPORTED_ROBOTS.keys())}"
            )
        self._robot_type   = robot_type
        self._robot_config = robot_config
        self._fps          = fps
        self._spec         = SUPPORTED_ROBOTS[robot_type]

        self._followers: Dict[str, Any] = {}
        self._leaders: Dict[str, Any] = {}
        self._follower_order: List[str] = []
        self._leader_order: List[str] = []
        self._connected    = False
        self._step_count   = 0
        self._state_keys: List[str] = []   # populated after first observation

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

    def connect(self, calibrate: bool = True) -> None:
        """Initialize and connect follower + leader arms to hardware."""
        logger.info("Connecting to robot type='%s'", self._robot_type)
        allow_partial_followers = bool(self._robot_config.get("allow_partial_followers", False))
        follower_ports = self._get_follower_ports()
        leader_ports = self._get_leader_ports()

        self._followers = self._build_followers(follower_ports)
        self._leaders = self._build_leaders(leader_ports)
        self._follower_order = list(follower_ports.keys())
        self._leader_order = list(leader_ports.keys())

        connected_followers: List[str] = []
        connected_leaders: List[str] = []

        for arm_name in self._follower_order:
            follower = self._followers[arm_name]
            try:
                follower.connect(calibrate=calibrate)
                connected_followers.append(arm_name)
                logger.info("Follower arm '%s' connected", arm_name)
            except Exception as e:
                logger.error("Failed to connect follower '%s': %s", arm_name, e)
                if not allow_partial_followers:
                    for n in connected_followers:
                        try:
                            self._followers[n].disconnect()
                        except Exception:
                            pass
                    self._followers = {}
                    self._leaders = {}
                    raise

        if not connected_followers:
            raise RuntimeError("No follower arm connected")

        # Keep only connected followers if partial mode is enabled.
        self._followers = {n: self._followers[n] for n in connected_followers}
        self._follower_order = connected_followers

        for arm_name in self._leader_order:
            leader = self._leaders[arm_name]
            try:
                leader.connect(calibrate=calibrate)
                connected_leaders.append(arm_name)
                logger.info("Leader arm '%s' connected", arm_name)
            except Exception as e:
                logger.warning("Failed to connect leader '%s' (teleop unavailable): %s", arm_name, e)

        # Keep only connected leaders for teleop.
        self._leaders = {n: self._leaders[n] for n in connected_leaders}
        self._leader_order = connected_leaders

        self._connected = True
        logger.info(
            "RobotConnector ready: followers=%s leaders=%s",
            self._follower_order,
            self._leader_order,
        )

    def disconnect(self) -> None:
        """Safely disconnect all arms."""
        for arm_name, follower in self._followers.items():
            try:
                follower.disconnect()
                logger.info("Follower '%s' disconnected", arm_name)
            except Exception as e:
                logger.warning("Error disconnecting follower '%s': %s", arm_name, e)

        for arm_name, leader in self._leaders.items():
            try:
                leader.disconnect()
                logger.info("Leader '%s' disconnected", arm_name)
            except Exception as e:
                logger.warning("Error disconnecting leader '%s': %s", arm_name, e)

        self._followers = {}
        self._leaders = {}
        self._follower_order = []
        self._leader_order = []

        self._connected = False

    @contextmanager
    def session(self, calibrate: bool = True) -> Generator[None, None, None]:
        self.connect(calibrate=calibrate)
        try:
            yield
        finally:
            self.disconnect()

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def get_observation(self) -> RawObservation:
        """
        Read one observation from follower arm + cameras.

        lerobot 0.5.x get_observation() returns:
          { "shoulder_pan.pos": float, ..., "cam_name": ndarray(H,W,C) }
        """
        self._check_connected()
        images: Dict[str, np.ndarray] = {}
        state_vals: List[float] = []
        state_keys: List[str] = []

        multi_arm = len(self._followers) > 1
        for arm_name in self._follower_order:
            obs_dict = self._followers[arm_name].get_observation()

            for key, val in obs_dict.items():
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    # Camera image: already (H, W, C) uint8 in lerobot 0.5.x
                    img = val
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    img_key = f"{arm_name}/{key}" if multi_arm else key
                    images[img_key] = img
                elif key.endswith(".pos") or "state" in key:
                    state_key = f"{arm_name}:{key}" if multi_arm else key
                    state_keys.append(state_key)
                    state_vals.append(float(val))

        self._state_keys = state_keys
        state = np.array(state_vals, dtype=np.float32) if state_vals else np.zeros(1, dtype=np.float32)

        self._step_count += 1
        return RawObservation(
            images=images,
            state=state,
            state_keys=state_keys,
            step=self._step_count,
        )

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: np.ndarray,
        state_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Send action to follower arm.

        action: (action_dim,) float32 — joint positions (degrees or normalized)
        state_keys: motor key names e.g. ["shoulder_pan.pos", ...]
                    defaults to last observed keys
        """
        self._check_connected()
        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
        keys = state_keys or self._state_keys
        if not keys:
            raise ValueError("state_keys unknown — call get_observation() first")
        if len(action_vec) != len(keys):
            raise ValueError(
                f"Action length ({len(action_vec)}) != len(state_keys) ({len(keys)})"
            )

        if len(self._followers) == 1:
            follower = self._followers[self._follower_order[0]]
            action_dict: Dict[str, float] = {
                k: float(v) for k, v in zip(keys, action_vec)
            }
            follower.send_action(action_dict)
            return

        # Multi-arm: keys must be arm-prefixed (e.g. right:shoulder_pan.pos)
        per_arm: Dict[str, Dict[str, float]] = {n: {} for n in self._follower_order}
        for key, val in zip(keys, action_vec):
            if ":" not in key:
                raise ValueError(
                    f"Multi-arm action key must be prefixed as '<arm>:<joint>' but got '{key}'"
                )
            arm_name, joint_key = key.split(":", 1)
            if arm_name not in per_arm:
                raise ValueError(f"Unknown arm '{arm_name}' in key '{key}'")
            per_arm[arm_name][joint_key] = float(val)

        for arm_name in self._follower_order:
            if per_arm[arm_name]:
                self._followers[arm_name].send_action(per_arm[arm_name])

    # ------------------------------------------------------------------ #
    # Teleoperation (data collection)
    # ------------------------------------------------------------------ #

    def get_teleop_action(self) -> np.ndarray:
        """
        Read current leader arm position.
        Returns: (action_dim,) float32 array (joint positions in degrees)
        """
        self._check_connected()
        if not self._leaders:
            raise RuntimeError("Leader arm not connected — teleoperation unavailable")

        multi_arm = len(self._leaders) > 1
        action_vals: List[float] = []
        action_keys: List[str] = []
        for arm_name in self._leader_order:
            action_dict = self._leaders[arm_name].get_action()
            for key, val in action_dict.items():
                k = f"{arm_name}:{key}" if multi_arm else key
                action_keys.append(k)
                action_vals.append(float(val))

        self._state_keys = action_keys
        return np.array(action_vals, dtype=np.float32)

    def teleop_step(self) -> tuple[RawObservation, np.ndarray]:
        """
        Single teleoperation step:
          1. Read leader position (desired action)
          2. Send to follower
          3. Read follower observation
        Returns: (observation, action)
        """
        self._check_connected()
        if not self._leaders:
            raise RuntimeError("Leader arm not connected — teleoperation unavailable")

        multi_arm = len(self._leaders) > 1
        action_vals: List[float] = []
        action_keys: List[str] = []

        # 1. Read each leader and map to same-name follower.
        for arm_name in self._leader_order:
            action_dict = self._leaders[arm_name].get_action()

            # 2. Send to matching follower arm, if present.
            follower = self._followers.get(arm_name)
            if follower is not None:
                follower.send_action(action_dict)

            for key, val in action_dict.items():
                k = f"{arm_name}:{key}" if multi_arm else key
                action_keys.append(k)
                action_vals.append(float(val))

        self._state_keys = action_keys

        # 3. Read follower observation
        obs = self.get_observation()
        action = np.array(action_vals, dtype=np.float32)
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

    @property
    def state_keys(self) -> List[str]:
        return self._state_keys

    def get_info(self) -> Dict[str, Any]:
        return {
            "robot_type":  self._robot_type,
            "connected":   self._connected,
            "fps":         self._fps,
            "step_count":  self._step_count,
            "state_keys":  self._state_keys,
            "followers":   self._follower_order,
            "leaders":     self._leader_order,
            "has_leader":  bool(self._leaders),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_followers(self, ports: Dict[str, str]) -> Dict[str, Any]:
        """Build one SOFollower instance per follower arm."""
        followers: Dict[str, Any] = {}
        multi_arm = len(ports) > 1
        for idx, (arm_name, port) in enumerate(ports.items()):
            # Attach shared cameras only to the first follower to avoid
            # opening identical camera devices multiple times.
            include_cameras = idx == 0
            followers[arm_name] = self._build_follower(
                arm_name=arm_name,
                port=port,
                multi_arm=multi_arm,
                include_cameras=include_cameras,
            )
        return followers

    def _build_follower(
        self,
        arm_name: str,
        port: str,
        multi_arm: bool = False,
        include_cameras: bool = False,
    ):
        """Build lerobot 0.5.x SOFollower with optional camera config."""
        import importlib
        spec = self._spec

        # Load follower class
        mod = importlib.import_module(spec["follower_module"])
        FollowerCls = getattr(mod, spec["follower_class"])

        # Load config class
        cfg_mod = importlib.import_module(spec["follower_config_module"])
        FollowerConfigCls = getattr(cfg_mod, spec["follower_config_class"])

        cameras = self._build_camera_configs() if include_cameras else {}
        # lerobot bi-arm calibration CLI stores per-arm files as dual_right/dual_left.
        follower_id = f"dual_{arm_name}" if multi_arm else spec["robot_id"]

        # Instantiate config — only pass known fields
        try:
            cfg = FollowerConfigCls(
                port=port,
                id=follower_id,
                cameras=cameras,
            )
        except TypeError:
            # Fallback if 'id' not accepted
            cfg = FollowerConfigCls(port=port, cameras=cameras)

        return FollowerCls(cfg)

    def _build_leaders(self, ports: Dict[str, str]) -> Dict[str, Any]:
        """Build one SOLeader instance per leader arm."""
        leaders: Dict[str, Any] = {}
        multi_arm = len(ports) > 1
        for arm_name, port in ports.items():
            leaders[arm_name] = self._build_leader(
                arm_name=arm_name,
                port=port,
                multi_arm=multi_arm,
            )
        return leaders

    def _build_leader(self, arm_name: str, port: str, multi_arm: bool = False):
        """Build lerobot 0.5.x SOLeader."""
        import importlib
        spec = self._spec

        mod = importlib.import_module(spec["leader_module"])
        LeaderCls = getattr(mod, spec["leader_class"])

        cfg_mod = importlib.import_module(spec["leader_config_module"])
        LeaderConfigCls = getattr(cfg_mod, spec["leader_config_class"])

        # lerobot bi-arm calibration CLI stores per-arm files as dual_right/dual_left.
        teleop_id = f"dual_{arm_name}" if multi_arm else spec["teleop_id"]
        try:
            cfg = LeaderConfigCls(port=port, id=teleop_id)
        except TypeError:
            cfg = LeaderConfigCls(port=port)

        return LeaderCls(cfg)

    def _get_follower_ports(self) -> Dict[str, str]:
        """Extract follower arm -> USB port mapping from robot_config."""
        cfg = self._robot_config
        # Flat style: robot_config: {port: /dev/ttyUSB1}
        if "port" in cfg:
            return {"main": cfg["port"]}

        # Preferred style: robot_config: {follower_arms: {right: {port: ...}, ...}}
        if "follower_arms" in cfg:
            out: Dict[str, str] = {}
            for arm_name, arm_cfg in cfg["follower_arms"].items():
                if "port" not in arm_cfg:
                    raise ValueError(f"Missing 'port' for follower arm '{arm_name}'")
                out[arm_name] = arm_cfg["port"]
            if out:
                return out

        raise ValueError(
            "Cannot find follower ports in robot_config. "
            "Set robot_config.port or robot_config.follower_arms.<name>.port"
        )

    def _get_leader_ports(self) -> Dict[str, str]:
        """Extract leader arm -> USB port mapping from robot_config."""
        cfg = self._robot_config

        # Flat style: robot_config: {leader_port: /dev/ttyUSB0}
        if "leader_port" in cfg:
            return {"main": cfg["leader_port"]}

        if "leader_arms" in cfg:
            out: Dict[str, str] = {}
            for arm_name, arm_cfg in cfg["leader_arms"].items():
                if "port" not in arm_cfg:
                    raise ValueError(f"Missing 'port' for leader arm '{arm_name}'")
                out[arm_name] = arm_cfg["port"]
            if out:
                return out

        raise ValueError(
            "Cannot find leader ports in robot_config. "
            "Set robot_config.leader_port or robot_config.leader_arms.<name>.port"
        )

    def _build_camera_configs(self) -> Dict[str, Any]:
        """Convert old-style camera dict to lerobot 0.5.x OpenCVCameraConfig."""
        from lerobot.cameras.opencv import OpenCVCameraConfig

        cam_cfgs: Dict[str, Any] = {}
        cameras = self._robot_config.get("cameras", {})
        for cam_name, cam_cfg in cameras.items():
            idx = cam_cfg.get("camera_index", 0)
            cam_cfgs[cam_name] = OpenCVCameraConfig(
                index_or_path=idx,
                fps=cam_cfg.get("fps"),
                width=cam_cfg.get("width"),
                height=cam_cfg.get("height"),
            )
        return cam_cfgs

    def _check_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

    def __repr__(self) -> str:
        return (
            f"RobotConnector(type={self._robot_type}, "
            f"connected={self._connected}, fps={self._fps})"
        )
