"""ENSEMBLE baseline: GDino + Molmo B+A hybrid.

B approach (role-based specialization):
  - GDino: binary present/absent oracle (reliable for disappear detection)
  - Molmo: coordinate-based spatial precision

A approach (t0-relative delta):
  - Normalizes against each detector's own t0 baseline
  - Cancels out Molmo's systematic over-counting

Decision flow:
  1. GDino t0=0  → force_map risk → Molmo-only fallback
  2. GDino Ck=0 AND Molmo Ck=0 → STOP (both confirm disappear)
  3. GDino Ck=0 BUT Molmo Ck>0 → ASK  (detectors disagree)
  4. GDino Ck=1 AND Molmo AMBIGUOUS → use GDino coord to override CLEAR
  5. Otherwise → Molmo coordinate-based result
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from baselines.common import BaselineResult
from monitoring.consistency_monitor import (
    Decision,
    check_grounding_ensemble,
    get_checkpoint_detections,
)

METHOD_NAME = "ENSEMBLE"


def _load_pil(path: str | Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def run_ensemble(
    g0: dict[str, Any],
    initial_img: str | Path,
    checkpoint_img: str | Path,
    *,
    checkpoint: str = "C1",
    handler: Any,
    gdino: Any,
    threshold: float = 50.0,
    session_id: str = "ensemble_checkpoint",
) -> BaselineResult:
    """Run B+A ensemble at one checkpoint.

    Args:
        g0: Initial grounding from extract_g0.
        initial_img: t0 image path.
        checkpoint_img: Checkpoint image path (C1 or C2).
        checkpoint: "C1" or "C2".
        handler: AmbResHandler (Molmo) for checkpoint detection.
        gdino: GroundingDINO instance for detection.
        threshold: Pixel distance threshold.
        session_id: Session identifier.

    Returns:
        BaselineResult with ENSEMBLE method name.
    """
    if checkpoint not in ("C1", "C2"):
        raise ValueError(f"checkpoint must be 'C1' or 'C2', got {checkpoint!r}")

    labels = list(dict.fromkeys([g0["target"]["label"], g0["destination"]["label"]]))

    # ── GDino at t0 ──────────────────────────────────────────────────────────
    pil_t0 = _load_pil(initial_img)
    gdino_t0 = gdino.detect(pil_t0, labels)

    # ── GDino at Ck ──────────────────────────────────────────────────────────
    pil_ck = _load_pil(checkpoint_img)
    gdino_ck = gdino.detect(pil_ck, labels)

    # ── Molmo at Ck ──────────────────────────────────────────────────────────
    molmo_ck = get_checkpoint_detections(
        checkpoint_img, g0, handler, session_id=session_id
    )

    # ── Ensemble decision ─────────────────────────────────────────────────────
    state, decision = check_grounding_ensemble(
        g0,
        molmo_ck,
        gdino_t0,
        gdino_ck,
        checkpoint,
        threshold,
    )

    role = "target" if checkpoint == "C1" else "destination"
    label = g0[role]["label"]
    gdino_t0_n = len(gdino_t0.get(label) or [])
    gdino_ck_n = len(gdino_ck.get(label) or [])
    molmo_ck_n = len([
        c for c in (molmo_ck.get(label) or [])
        if isinstance(c, (list, tuple)) and len(c) == 2
    ])

    reason = (
        f"{role} '{label}': "
        f"gdino t0={gdino_t0_n} ck={gdino_ck_n}, "
        f"molmo ck={molmo_ck_n} → {state.value}"
    )

    return BaselineResult(
        method=METHOD_NAME,
        decision=decision,
        reason=reason,
        question=(
            f"Ensemble anomaly detected for {role} '{label}'. Please clarify."
            if decision != Decision.CONTINUE
            else ""
        ),
        raw_output={
            "gdino_t0": {k: len(v) for k, v in gdino_t0.items()},
            "gdino_ck": {k: len(v) for k, v in gdino_ck.items()},
            "molmo_ck": {k: len([c for c in v if len(c) == 2])
                         for k, v in molmo_ck.items()},
        },
        metadata={
            "checkpoint": checkpoint,
            "session_id": session_id,
            "uses_checkpoint": True,
            "stores_g0": True,
            "uses_coord": True,
            "uses_gdino": True,
            "threshold": threshold,
            "state": state.value,
            "gdino_t0_count": gdino_t0_n,
            "gdino_ck_count": gdino_ck_n,
            "molmo_ck_count": molmo_ck_n,
        },
    )
