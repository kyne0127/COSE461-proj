"""
src.trigger.dinov2_monitor
==========================
DINOv2-based lightweight scene-change monitor.

At G₀ time, reference features are extracted from intent-relevant image crops
(target bbox and destination bbox).  Each subsequent frame is scored against
these references using cosine distance.  A high score means the scene region
has changed — which may signal that AmbResVLM should be re-invoked.

Phase-specific monitoring (spec §3):
  pre_pick  (t₀ → C1): Monitor DESTINATION bbox only.
            The robot arm moves toward the target during this phase, so the
            target region contains arm-motion noise.  Destination changes are
            unexpected and worth flagging.
  pre_place (C1 → C2): Monitor both TARGET bbox and DESTINATION bbox.
            - Target region: confirms object was picked (should be empty).
              Sudden re-appearance or a different object would be anomalous.
            - Destination region: watches for occupation before placement.
            Score = max(target_score, dest_score).

Note: §4.3 of the spec only monitors bbox_dest for both phases; this
implementation corrects that to align with the §3 rationale.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate / bbox helpers
# ---------------------------------------------------------------------------

BBOX_SIZE_DEFAULT = 80  # pixels; spec §2.2 recommends 80


def coord_to_bbox(
    coord: list[int] | list[float],
    img_shape: list[int] | tuple[int, ...],
    size: int = BBOX_SIZE_DEFAULT,
) -> list[int]:
    """Convert a pointing coordinate to a square bounding box.

    Args:
        coord:     [x, y] pixel coordinate (column, row).
        img_shape: [H, W] or (H, W[, C]).
        size:      Side length of the square crop (pixels).

    Returns:
        [x1, y1, x2, y2] clamped to image bounds.
    """
    x, y = int(coord[0]), int(coord[1])
    h, w = int(img_shape[0]), int(img_shape[1])
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    return [x1, y1, x2, y2]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D feature vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # treat as identical when one is zero
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# DINOv2Monitor
# ---------------------------------------------------------------------------

class DINOv2Monitor:
    """
    Computes scene-change score between G₀ reference frame and live frames.

    Args:
        frame_0:    Reference frame at t₀ (np.ndarray, uint8 or float [0,1], HxWxC).
        g0:         G₀ dict with "target" / "destination" / "image_shape".
        model:      Any object with an `encode(crop: np.ndarray) -> np.ndarray` method.
                    Expected to return an L2-normalised embedding vector.
        threshold:  Change-score threshold θ (spec §2.4 default: 0.15).
        bbox_size:  Square crop size around G₀ coord (pixels).
    """

    def __init__(
        self,
        frame_0: np.ndarray,
        g0: dict[str, Any],
        model: Any,
        threshold: float = 0.15,
        bbox_size: int = BBOX_SIZE_DEFAULT,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self._bbox_size = bbox_size

        img_shape = g0.get("image_shape") or list(frame_0.shape[:2])

        # ── Target region ────────────────────────────────────────────────────
        target_coord = (g0.get("target") or {}).get("coord")
        if target_coord is not None:
            self.bbox_target: list[int] | None = coord_to_bbox(
                target_coord, img_shape, bbox_size
            )
            self.f0_target: np.ndarray | None = self._extract(frame_0, self.bbox_target)
        else:
            self.bbox_target = None
            self.f0_target = None

        # ── Destination region ───────────────────────────────────────────────
        dest_coord = (g0.get("destination") or {}).get("coord")
        if dest_coord is not None:
            self.bbox_dest: list[int] | None = coord_to_bbox(
                dest_coord, img_shape, bbox_size
            )
            self.f0_dest: np.ndarray | None = self._extract(frame_0, self.bbox_dest)
        else:
            self.bbox_dest = None
            self.f0_dest = None

        self._last_score_target: float = 0.0
        self._last_score_dest: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_score(
        self,
        frame_t: np.ndarray,
        phase: Literal["pre_pick", "pre_place"],
    ) -> float:
        """
        Compute scene-change score for the current frame.

        pre_pick  → destination-only score  (arm near target → target noisy)
        pre_place → max(target score, destination score)

        Returns:
            float in [0, 1].  Higher = more change detected.
        """
        if phase == "pre_pick":
            return self._score_dest(frame_t)

        elif phase == "pre_place":
            s_tgt = self._score_target(frame_t)
            s_dst = self._score_dest(frame_t)
            return max(s_tgt, s_dst)

        else:
            raise ValueError(f"Unknown phase: {phase!r}. Expected 'pre_pick' or 'pre_place'.")

    @property
    def last_scores(self) -> dict[str, float]:
        return {
            "target":      self._last_score_target,
            "destination": self._last_score_dest,
        }

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _score_target(self, frame_t: np.ndarray) -> float:
        if self.f0_target is None or self.bbox_target is None:
            return 0.0
        ft = self._extract(frame_t, self.bbox_target)
        score = 1.0 - _cosine_sim(self.f0_target, ft)
        self._last_score_target = score
        return score

    def _score_dest(self, frame_t: np.ndarray) -> float:
        if self.f0_dest is None or self.bbox_dest is None:
            return 0.0
        ft = self._extract(frame_t, self.bbox_dest)
        score = 1.0 - _cosine_sim(self.f0_dest, ft)
        self._last_score_dest = score
        return score

    def _extract(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Crop the region and run the DINOv2 encoder."""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(1, dtype=np.float32)
        return self.model.encode(crop)


# ---------------------------------------------------------------------------
# DINOv2 model loader (HuggingFace transformers)
# ---------------------------------------------------------------------------

class DINOv2Encoder:
    """
    Wraps a HuggingFace DINOv2 model to provide the `encode(crop)` interface
    expected by DINOv2Monitor.

    Args:
        model_name: "vit_s" → facebook/dinov2-small (384-d)
                    "vit_b" → facebook/dinov2-base  (768-d)
        device:     "cuda" or "cpu"
    """

    _MODEL_IDS: dict[str, str] = {
        "vit_s": "facebook/dinov2-small",
        "vit_b": "facebook/dinov2-base",
    }

    def __init__(self, model_name: str = "vit_s", device: str = "cuda") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model_id = self._MODEL_IDS[model_name]
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device)
        self._model.eval()
        self._device = device
        self._torch = torch

    def encode(self, crop: np.ndarray) -> np.ndarray:
        """Return an L2-normalised CLS-token embedding for the given crop.

        Args:
            crop: np.ndarray, shape (H, W, 3), uint8 [0,255] or float [0,1].
        """
        from PIL import Image as _PILImage

        if crop.dtype != np.uint8:
            crop = (crop * 255.0).clip(0, 255).astype(np.uint8)

        pil_img = _PILImage.fromarray(crop)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token embedding
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-9)


def load_dinov2(model_name: str = "vit_s", device: str = "cuda") -> DINOv2Encoder:
    """Convenience factory matching the spec §2.5 model selection table."""
    return DINOv2Encoder(model_name=model_name, device=device)


# ---------------------------------------------------------------------------
# Null model (for offline testing without GPU)
# ---------------------------------------------------------------------------

class _RandomEmbeddingModel:
    """Drop-in test model that returns random unit-norm embeddings."""

    def __init__(self, dim: int = 384, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._dim = dim

    def encode(self, crop: np.ndarray) -> np.ndarray:
        v = self._rng.standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
