"""
module.models.rgbd_diffusion
=============================
RGB-D Diffusion Policy model implementation.

Exports:
    - RGBDDiffusionPolicyModel: Main model class
    - depth_mm_to_meters: Utility to convert ZED depth from mm to metres
    - create_rgbd_tensor: Utility to concatenate RGB and depth
"""

from .model import (
    RGBDDiffusionPolicyModel,
    create_rgbd_tensor,
    depth_mm_to_meters,
)

__all__ = [
    "RGBDDiffusionPolicyModel",
    "depth_mm_to_meters",
    "create_rgbd_tensor",
]
