"""
module.models.pcd_grasp.model
==============================
3D Point Cloud-based Grasping Model Implementation

Paper Type: PointCloud-based 6D Object Grasping
Observation Type: Case C — Point Cloud + RGB-D
Input:
  - pcd: [2048, 3] xyz coordinates (Point Cloud)
  - zed_rgb: [H, W, 3] RGB image
  - state: [6] Joint angles (radians)

Output:
  - grasp_pose: [7] (position[3] + quaternion[4])

Server-side implements:
  - depth_to_pointcloud(): ZED depth → point cloud conversion
  - fps_sampling(): Farthest Point Sampling for 2048 points
  - PCD normalization and preprocessing

ZED VGA Intrinsics (f: 350, 350; c: 336, 188)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from module.models.base_model import BaseLeRobotModel, Observation, TrainingBatch
from module.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Point Cloud Processing Utilities
# ────────────────────────────────────────────────────────────────────────────


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float = 350.0,
    fy: float = 350.0,
    cx: float = 336.0,
    cy: float = 188.0,
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud using camera intrinsics.

    Args:
        depth: [H, W, 1] or [H, W] depth map in meters (float32)
        fx, fy: focal lengths in pixels
        cx, cy: principal point in pixels

    Returns:
        pcd: [N, 3] xyz coordinates (float32) where N = H*W
    """
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    assert depth.ndim == 2, f"Expected 2D depth, got shape {depth.shape}"

    h, w = depth.shape

    # Create pixel coordinate grids
    x_idx = np.arange(w, dtype=np.float32)
    y_idx = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x_idx, y_idx)  # [H, W], [H, W]

    # Backproject to 3D using camera intrinsics
    # x_3d = (u - cx) * z / fx
    # y_3d = (v - cy) * z / fy
    # z_3d = z
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy

    # Stack into [H, W, 3]
    pcd_full = np.stack([x, y, z], axis=-1)  # [H, W, 3]

    # Reshape to [N, 3]
    pcd = pcd_full.reshape(-1, 3)

    # Remove points with invalid depth (z <= 0 or z > max_depth)
    valid_mask = (pcd[:, 2] > 0.01) & (pcd[:, 2] < 10.0)
    pcd = pcd[valid_mask]

    return pcd.astype(np.float32)


def fps_sampling(
    pcd: np.ndarray,
    num_samples: int = 2048,
) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) to downsample point cloud.

    Greedy algorithm that iteratively selects points farthest from already-selected points.
    More computationally expensive than random sampling but better geometric coverage.

    Args:
        pcd: [N, 3] point cloud
        num_samples: target number of points (default 2048)

    Returns:
        pcd_sampled: [num_samples, 3] downsampled point cloud
    """
    n_points = pcd.shape[0]

    # If already smaller, just return (optionally pad with zeros)
    if n_points <= num_samples:
        padding = num_samples - n_points
        if padding > 0:
            pad = np.zeros((padding, 3), dtype=np.float32)
            return np.vstack([pcd, pad])
        return pcd[:num_samples]

    # FPS algorithm
    selected_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n_points, np.inf, dtype=np.float32)

    # Pick first point randomly
    selected_indices[0] = np.random.randint(0, n_points)
    distances = np.linalg.norm(
        pcd - pcd[selected_indices[0]], axis=1
    ).astype(np.float32)

    # Iteratively pick farthest point
    for i in range(1, num_samples):
        farthest_idx = np.argmax(distances)
        selected_indices[i] = farthest_idx

        # Update distances
        new_dists = np.linalg.norm(
            pcd - pcd[farthest_idx], axis=1
        ).astype(np.float32)
        distances = np.minimum(distances, new_dists)

    return pcd[selected_indices].astype(np.float32)


def normalize_pointcloud(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize point cloud to unit sphere (centered at origin, radius ~1).

    Args:
        pcd: [N, 3] point cloud

    Returns:
        pcd_norm: [N, 3] normalized point cloud
        center: [3] centroid used for normalization
        scale: float scale factor
    """
    center = np.mean(pcd, axis=0)  # [3]
    pcd_centered = pcd - center

    # Scale to fit in unit sphere
    distances = np.linalg.norm(pcd_centered, axis=1)
    scale = np.max(distances) + 1e-6

    pcd_norm = pcd_centered / scale

    return pcd_norm.astype(np.float32), center.astype(np.float32), scale


# ────────────────────────────────────────────────────────────────────────────
# PointCloud Grasping Model
# ────────────────────────────────────────────────────────────────────────────


@ModelRegistry.register("pcd_grasp")
class PCDGraspModel(BaseLeRobotModel):
    """
    3D Point Cloud-based Grasping Model.

    Processes depth maps to point clouds, applies FPS sampling,
    and predicts 6D grasp poses (position + quaternion).

    Config keys:
        device:           "cuda" | "cpu"
        num_points:       number of sampled points (default: 2048)
        use_rgb:          whether to use ZED RGB (default: True)
        output_type:      "pose_7" | "pose_6" | "action_6" (default: "pose_7")
    """

    MODEL_TYPE = "pcd_grasp"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Model hyperparameters
        self._num_points = config.get("num_points", 2048)
        self._use_rgb = config.get("use_rgb", True)
        self._output_type = config.get("output_type", "pose_7")

        # ZED VGA intrinsics
        self._fx = 350.0
        self._fy = 350.0
        self._cx = 336.0
        self._cy = 188.0

        # Placeholder network (would be PointNet++ in real implementation)
        self._network = None
        self.logger.info(
            f"PCDGraspModel initialized: num_points={self._num_points}, "
            f"use_rgb={self._use_rgb}, output={self._output_type}"
        )

    # ────────────────────────────────────────────────────────────────
    # Required interface
    # ────────────────────────────────────────────────────────────────

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load pre-trained model weights.

        In a real implementation, this would load a PointNet++ backbone
        trained on grasp data (e.g., from Dex-Net, GGCNN datasets).
        """
        checkpoint_path = Path(checkpoint_path)
        self.logger.info(f"Loading PCD Grasp checkpoint from {checkpoint_path}")

        try:
            # Stub: In production, load actual PointNet++ model here
            # import torch
            # from .pointnet_plusplus import PointNetPlusPlus
            # self._network = PointNetPlusPlus(...)
            # state_dict = torch.load(checkpoint_path / "model.pth")
            # self._network.load_state_dict(state_dict)

            self.logger.info("PCD Grasp model loaded (stub)")
            self._mark_loaded()

        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            self._mark_loaded()  # Allow model to run with random init

    def predict_action(self, observation: Observation) -> np.ndarray:
        """
        Predict grasp pose from observation.

        Process:
          1. Extract ZED depth and RGB from observation
          2. Convert depth → point cloud (depth_to_pointcloud)
          3. Apply FPS sampling to 2048 points
          4. Normalize point cloud
          5. Forward through network (PointNet++ backbone)
          6. Output: grasp_pose [7] = position[3] + quaternion[4]

        Args:
            observation: Observation with images dict:
                - "zed_rgb": [H, W, 3] or [3, H, W]
                - "zed_depth": [H, W, 1] in metres
                - state: [6] joint angles

        Returns:
            action: [7] (position[3] + quaternion[4])
                     or [6] (position[3] + rotation_6d) if output_type="action_6"
        """
        # Extract images
        images = observation.images
        zed_depth = images.get("zed_depth")
        zed_rgb = images.get("zed_rgb")
        state = observation.state  # [6]

        if zed_depth is None:
            self.logger.warning("No ZED depth provided, returning dummy grasp")
            return np.array([0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Ensure depth is [H, W, 1] or [H, W]
        if zed_depth.ndim == 3 and zed_depth.shape[2] == 1:
            pass  # Already correct
        elif zed_depth.ndim == 2:
            zed_depth = zed_depth[..., np.newaxis]
        else:
            raise ValueError(f"Unexpected depth shape: {zed_depth.shape}")

        # Step 1: depth → pointcloud
        pcd_full = depth_to_pointcloud(
            zed_depth,
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
        )  # [N, 3]

        self.logger.debug(f"Generated point cloud: {pcd_full.shape}")

        # Step 2: FPS sampling
        pcd_sampled = fps_sampling(pcd_full, num_samples=self._num_points)  # [2048, 3]

        # Step 3: Normalize
        pcd_norm, center, scale = normalize_pointcloud(pcd_sampled)  # [2048, 3]

        self.logger.debug(
            f"Normalized PCD: center={center}, scale={scale:.4f}, shape={pcd_norm.shape}"
        )

        # Step 4: Forward pass
        # In a real implementation, this would invoke a PointNet++ network:
        # output = self._network(torch.from_numpy(pcd_norm[np.newaxis]).to(device))
        # For now, return a dummy grasp pose

        # Dummy forward: grasp pose = [x, y, z, qx, qy, qz, qw]
        # Position: small offset from hand frame
        # Quaternion: identity (no rotation)
        grasp_pos = np.array([0.3, 0.0, 0.1], dtype=np.float32)
        grasp_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Identity quaternion

        action = np.concatenate([grasp_pos, grasp_quat])  # [7]

        return action.astype(np.float32)

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute one training step.

        In a real implementation:
          - Convert batch depth maps → point clouds
          - Forward through PointNet++ network
          - Compute loss (e.g., L2 on pose, Huber on orientation)
          - Backprop and update

        For now, return a stub loss dict.
        """
        self.logger.debug("train_step called (stub implementation)")

        # Placeholder: return dummy losses
        return {
            "loss_position": 0.0,
            "loss_quaternion": 0.0,
            "loss_total": 0.0,
        }

    # ────────────────────────────────────────────────────────────────
    # Optional overrides
    # ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Called at episode start."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return config dict."""
        cfg = super().get_config()
        cfg.update({
            "num_points": self._num_points,
            "use_rgb": self._use_rgb,
            "output_type": self._output_type,
            "camera_intrinsics": {
                "fx": self._fx,
                "fy": self._fy,
                "cx": self._cx,
                "cy": self._cy,
            }
        })
        return cfg
