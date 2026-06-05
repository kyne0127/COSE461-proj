"""DetectionBundle: shared detection results computed once per trial.

All baselines (B1~B4) and VLM judge methods consume this bundle so that
every method operates on identical detection results — making comparisons fair.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from extraction.ambres_g0_extractor import extract_g0
from monitoring.consistency_monitor import get_checkpoint_detections


def _gdino_detect(gdino: Any, image_path: str | Path, labels: list[str]) -> dict[str, list]:
    """Run GroundingDINO detection → {label: [[x,y], ...]} format."""
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    return gdino.detect(pil, labels)


@dataclass
class DetectionBundle:
    """Trial당 1회 계산되어 모든 메서드가 공유하는 detection 결과."""
    objects: list[str]
    g0: dict[str, Any]
    det_t0: dict[str, list]
    det_c1: dict[str, list]
    det_c2: dict[str, list]

    def det_at(self, checkpoint: str) -> dict[str, list]:
        if checkpoint == "C1":
            return self.det_c1
        if checkpoint == "C2":
            return self.det_c2
        raise ValueError(f"Unknown checkpoint {checkpoint!r}")


def compute_detection_bundle(
    initial_img: str | Path,
    c1_img: str | Path,
    c2_img: str | Path,
    task: str,
    handler: Any,
    *,
    sample_id: str = "trial",
    gdino: Any = None,
) -> DetectionBundle:
    """Run detections once for a trial. Uses DINO if provided, else Molmo."""
    g0 = extract_g0(
        initial_img,
        task,
        handler=handler,
        session_id=f"{sample_id}_bundle_g0",
        allow_ambiguous=True,
    )
    target_label = g0["target"]["label"]
    dest_label = g0["destination"]["label"]
    labels = [target_label, dest_label]

    det_t0 = {
        target_label: g0["target"].get("coords") or [],
        dest_label: g0["destination"].get("coords") or [],
    }

    if gdino is not None:
        det_c1 = _gdino_detect(gdino, c1_img, labels)
        det_c2 = _gdino_detect(gdino, c2_img, labels)
    else:
        det_c1 = get_checkpoint_detections(
            c1_img, g0, handler, session_id=f"{sample_id}_bundle_c1"
        )
        det_c2 = get_checkpoint_detections(
            c2_img, g0, handler, session_id=f"{sample_id}_bundle_c2"
        )

    return DetectionBundle(
        objects=labels,
        g0=g0,
        det_t0=det_t0,
        det_c1=det_c1,
        det_c2=det_c2,
    )
