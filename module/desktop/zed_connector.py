"""
module.desktop.zed_connector
=============================
ZED Mini RGB-D Capture and USB WebCam Synchronization

Captures synchronized RGB and Depth frames from ZED Mini camera,
along with RGB from USB wrist-mounted camera.

Supports two backends:
  1. pyzed (ZED SDK) — preferred, full depth support
  2. OpenCV fallback — uses generic USB camera with simulated depth

Output specifications:
  - ZED RGB: [H, W, 3] uint8 → float32 [0,1]
  - ZED Depth: [H, W, 1] float32 (millimeters from SDK → converted to metres)
  - USB RGB: [H, W, 3] uint8 → float32 [0,1]

Dependencies:
    pip install opencv-python pyzed  (ZED SDK optional but recommended)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from module.utils.logging import get_logger

logger = get_logger("desktop.zed_connector")

# Try to import ZED SDK
try:
    import pyzed.sl as sl
    _ZED_SDK_AVAILABLE = True
except ImportError:
    _ZED_SDK_AVAILABLE = False
    logger.warning("pyzed (ZED SDK) not installed. Falling back to OpenCV + simulated depth.")


class ZEDCapture:
    """
    ZED Mini camera capture wrapper.

    Supports both pyzed (ZED SDK) and OpenCV fallback modes.
    """

    def __init__(
        self,
        resolution: str = "VGA",       # HD720 | VGA (360p)
        fps: int = 30,
        depth_mode: str = "PERFORMANCE",  # ULTRA | QUALITY | PERFORMANCE
        depth_min_dist: float = 0.3,   # metres
        depth_max_dist: float = 3.0,   # metres
        use_zed_sdk: bool = True,
        fallback_device_index: int = 0,  # OpenCV fallback camera index
    ) -> None:
        """
        Initialize ZED capture.

        Args:
            resolution: "VGA" (640x360) or "HD720" (1280x720)
            fps: Capture FPS
            depth_mode: Depth processing quality
            depth_min_dist: Minimum depth range (metres)
            depth_max_dist: Maximum depth range (metres)
            use_zed_sdk: Try to use ZED SDK if available
        """
        self._resolution = resolution
        self._fps = fps
        self._depth_mode = depth_mode
        self._depth_min = depth_min_dist
        self._depth_max = depth_max_dist
        self._use_zed_sdk = use_zed_sdk and _ZED_SDK_AVAILABLE
        self._fallback_device_index = fallback_device_index

        self._camera = None
        self._initialized = False

        # Resolution mapping
        self._res_map = {
            "VGA": (640, 360),
            "HD720": (1280, 720),
        }

        if resolution not in self._res_map:
            raise ValueError(f"Unsupported resolution: {resolution}")

        self._width, self._height = self._res_map[resolution]

        if self._use_zed_sdk:
            self._init_zed_sdk()
        else:
            self._init_opencv_fallback()

    def _init_zed_sdk(self) -> None:
        """Initialize ZED SDK."""
        try:
            self._camera = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = (
                sl.RESOLUTION.VGA if self._resolution == "VGA"
                else sl.RESOLUTION.HD720
            )
            init_params.camera_fps = self._fps
            depth_mode_map = {
                "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
                "QUALITY":     sl.DEPTH_MODE.QUALITY,
                "ULTRA":       sl.DEPTH_MODE.ULTRA,
                "NEURAL":      sl.DEPTH_MODE.NEURAL,
            }
            init_params.depth_mode = depth_mode_map.get(self._depth_mode, sl.DEPTH_MODE.NEURAL)
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = self._depth_min
            init_params.depth_maximum_distance = self._depth_max

            status = self._camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                logger.error("Failed to open ZED camera: %s", status)
                self._use_zed_sdk = False
                self._init_opencv_fallback()
            else:
                logger.info("ZED SDK initialized: %dx%d @ %dfps", self._width, self._height, self._fps)
                self._initialized = True

        except Exception as e:
            logger.warning("ZED SDK initialization failed, falling back to OpenCV: %s", e)
            self._use_zed_sdk = False
            self._init_opencv_fallback()

    def _init_opencv_fallback(self) -> None:
        """Initialize OpenCV fallback (simulated depth)."""
        self._camera = cv2.VideoCapture(self._fallback_device_index)
        if not self._camera.isOpened():
            raise RuntimeError("Failed to open camera with OpenCV")

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._camera.set(cv2.CAP_PROP_FPS, self._fps)
        self._initialized = True
        logger.info("OpenCV camera fallback initialized (simulated depth)")

    def capture_synchronized(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture synchronized RGB and Depth frames.

        Returns:
            rgb: [H, W, 3] uint8
            depth: [H, W, 1] float32 (metres)
        """
        if not self._initialized:
            raise RuntimeError("ZED capture not initialized")

        if self._use_zed_sdk:
            return self._capture_zed_sdk()
        else:
            return self._capture_opencv_fallback()

    def _capture_zed_sdk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture from ZED SDK."""
        runtime_params = sl.RuntimeParameters()
        if self._camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Capture RGB from left camera (BGRA → RGB)
            image = sl.Mat()
            self._camera.retrieve_image(image, sl.VIEW.LEFT)
            bgra = image.get_data()                          # [H, W, 4] uint8
            rgb = bgra[..., 2::-1].copy()                   # BGRA → RGB, [H, W, 3]

            # Capture Depth (metres, set by coordinate_units=METER)
            depth_map = sl.Mat()
            self._camera.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            depth_m = depth_map.get_data().astype(np.float32)
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_m = np.expand_dims(depth_m, axis=-1)      # [H, W, 1]

            return rgb, depth_m
        else:
            logger.error("ZED grab failed")
            raise RuntimeError("ZED grab failed")

    def _capture_opencv_fallback(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture from OpenCV (simulated depth)."""
        ret, frame = self._camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from OpenCV camera")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Simulate depth: fill with mid-range values (1.5m)
        depth_m = np.full((self._height, self._width, 1), 1.5, dtype=np.float32)

        return rgb, depth_m

    def release(self) -> None:
        """Release camera resources."""
        if self._camera is not None:
            if self._use_zed_sdk:
                try:
                    self._camera.close()
                except Exception as e:
                    logger.warning("Error closing ZED camera: %s", e)
            else:
                self._camera.release()
            logger.info("ZED camera released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class USBCapture:
    """
    USB camera capture wrapper (wrist-mounted RGB camera).

    Typically used for ego-view on SOArm-101 end-effector.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """
        Initialize USB camera.

        Args:
            device_index: Camera device index
            width: Frame width
            height: Frame height
            fps: Capture FPS
        """
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._camera = None
        self._initialized = False

        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize USB camera."""
        self._camera = cv2.VideoCapture(self._device_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"Failed to open USB camera {self._device_index}")

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._camera.set(cv2.CAP_PROP_FPS, self._fps)
        self._initialized = True
        logger.info("USB camera initialized: %dx%d @ %dfps (device %d)",
                    self._width, self._height, self._fps, self._device_index)

    def capture(self) -> np.ndarray:
        """
        Capture RGB frame.

        Returns:
            rgb: [H, W, 3] uint8
        """
        if not self._initialized:
            raise RuntimeError("USB camera not initialized")

        ret, frame = self._camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from USB camera")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb

    def release(self) -> None:
        """Release camera resources."""
        if self._camera is not None:
            self._camera.release()
            logger.info("USB camera released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class SyncCapture:
    """
    Synchronized ZED RGB-D + USB RGB capture.

    Coordinates frame capture from both cameras and returns
    aligned observations ready for model inference.
    """

    def __init__(
        self,
        zed_resolution: str = "VGA",
        zed_fps: int = 30,
        zed_depth_mode: str = "PERFORMANCE",
        zed_depth_min: float = 0.3,
        zed_depth_max: float = 3.0,
        usb_device_index: int = 0,
        usb_width: int = 640,
        usb_height: int = 480,
        usb_fps: int = 30,
    ) -> None:
        """
        Initialize synchronized capture.

        Args:
            zed_*: ZED Mini configuration
            usb_*: USB camera configuration
        """
        self._zed = ZEDCapture(
            resolution=zed_resolution,
            fps=zed_fps,
            depth_mode=zed_depth_mode,
            depth_min_dist=zed_depth_min,
            depth_max_dist=zed_depth_max,
        )
        self._usb = USBCapture(
            device_index=usb_device_index,
            width=usb_width,
            height=usb_height,
            fps=usb_fps,
        )

    def capture(self) -> dict[str, np.ndarray]:
        """
        Capture synchronized frames from both cameras.

        Returns:
            dict with keys:
                - "zed_rgb": [H, W, 3] uint8
                - "zed_depth": [H, W, 1] float32 (metres)
                - "usb_rgb": [H, W, 3] uint8
        """
        zed_rgb, zed_depth = self._zed.capture_synchronized()
        usb_rgb = self._usb.capture()

        return {
            "zed_rgb": zed_rgb,
            "zed_depth": zed_depth,
            "usb_rgb": usb_rgb,
        }

    def capture_normalized(self) -> dict[str, np.ndarray]:
        """
        Capture and normalize frames to float32 [0,1].

        Returns:
            dict with keys:
                - "zed_rgb": [H, W, 3] float32 [0,1]
                - "zed_depth": [H, W, 1] float32 (metres)
                - "usb_rgb": [H, W, 3] float32 [0,1]
        """
        frames = self.capture()
        frames["zed_rgb"] = frames["zed_rgb"].astype(np.float32) / 255.0
        frames["usb_rgb"] = frames["usb_rgb"].astype(np.float32) / 255.0
        return frames

    def release(self) -> None:
        """Release both cameras."""
        self._zed.release()
        self._usb.release()
        logger.info("SyncCapture released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
