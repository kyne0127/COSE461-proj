"""Shared pytest configuration, fixtures, and test-log hooks.

Responsibilities:
  1. Tensorflow stub — applied at import time so real-model tests don't fail
     on the conditional `import tensorflow` inside Molmo's preprocessing.
  2. Timestamped log file — each pytest run creates logs/pytest_YYYYMMDD_HHMMSS.log
     with the full terminal output (via pytest-capturing hooks).
  3. real_handler fixture — module-scoped AmbResHandler for integration tests.
"""
from __future__ import annotations

import datetime
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Path constants  (conftest lives in tests/, so parents[1] = project root)
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parents[1]   # COSE461-proj
SRC_ROOT    = REPO_ROOT / "src"                     # COSE461-proj/src
AMBRES_ROOT = REPO_ROOT.parent / "AmbRes"           # workspace/AmbRes
LOGS_DIR    = REPO_ROOT / "logs"

# pytest.ini already adds src/ via pythonpath, but keep this for direct runs
for _p in (SRC_ROOT, REPO_ROOT, AMBRES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Tensorflow stub — must run before any transformers/peft import
# ---------------------------------------------------------------------------

def _stub_tensorflow() -> None:
    if "tensorflow" in sys.modules:
        return
    tf = ModuleType("tensorflow")
    tf.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)
    tf.Tensor   = type("Tensor",   (), {})
    tf.Variable = type("Variable", (), {})
    sys.modules["tensorflow"] = tf

_stub_tensorflow()

# ---------------------------------------------------------------------------
# Integration test constants
# ---------------------------------------------------------------------------

REAL_IMAGE_MARKER = AMBRES_ROOT / "assets" / "images" / "5rhU25AdQW4jADxhp8EYuq.jpeg"
REAL_IMAGE_BLOCK  = AMBRES_ROOT / "assets" / "images" / "real_0.png"
REAL_CKPT         = "43qazb3XcrZF5rZWnjRPVm"   # CKPT.REAL

TASK_MARKER = "move the marker next to the sprite bottle"
TASK_BLOCK  = "place the blue cup on the table"

# ---------------------------------------------------------------------------
# Timestamped log file — copies pytest_latest.log after each run
# ---------------------------------------------------------------------------

def pytest_sessionstart(session) -> None:
    """Record run start time for the timestamped log."""
    session._run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Copy logs/pytest_latest.log → logs/pytest_YYYYMMDD_HHMMSS.log."""
    latest = LOGS_DIR / "pytest_latest.log"
    if not latest.exists():
        return
    ts  = getattr(session, "_run_timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    dst = LOGS_DIR / f"pytest_{ts}.log"
    shutil.copy2(latest, dst)


# ---------------------------------------------------------------------------
# GPU guard helper
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ---------------------------------------------------------------------------
# Module-scoped real handler — load model once per integration test module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_handler():
    """Live AmbResHandler with finetune model (CKPT.REAL).

    Auto-skips when:
      - No CUDA device available
      - Checkpoint directory not found
    """
    if not _cuda_available():
        pytest.skip("No CUDA device — integration test requires GPU")

    ckpt_dir = AMBRES_ROOT / "ckpt" / REAL_CKPT
    if not ckpt_dir.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_dir}")

    from module.models.ambres.handler import AmbResHandler

    handler = AmbResHandler()
    handler.setup({
        "model_type":    "finetune",
        "adapter_ckpt":  REAL_CKPT,
        "use_detection": False,
        "use_sam":       False,
    })
    return handler
