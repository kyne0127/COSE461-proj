"""
module.utils.tensor_utils
=========================
Helpers for converting between numpy/torch tensors and gRPC Tensor messages.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Lazy import to avoid hard torch dep on desktop
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# numpy ↔ proto Tensor
# ────────────────────────────────────────────────────────────────────────────

def numpy_to_proto(arr: np.ndarray, name: str = "") -> dict:
    """
    Convert numpy array to a dict matching proto Tensor schema.
    (Used before proto stubs are generated — works with grpc stubs too.)
    """
    arr = arr.astype(np.float32)
    return {
        "shape": list(arr.shape),
        "data":  arr.flatten().tolist(),
        "dtype": "float32",
        "name":  name,
    }


def proto_to_numpy(proto_tensor) -> np.ndarray:
    """Convert proto Tensor message → numpy float32 array."""
    shape = list(proto_tensor.shape)
    data  = np.array(proto_tensor.data, dtype=np.float32)
    return data.reshape(shape)


# ────────────────────────────────────────────────────────────────────────────
# torch ↔ proto Tensor
# ────────────────────────────────────────────────────────────────────────────

def torch_to_proto(tensor, name: str = "") -> dict:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed")
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    return numpy_to_proto(arr, name=name)


def proto_to_torch(proto_tensor, device: str = "cpu"):
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed")
    import torch
    arr = proto_to_numpy(proto_tensor)
    return torch.from_numpy(arr).to(device)


# ────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ────────────────────────────────────────────────────────────────────────────

def stack_proto_tensors(proto_tensors: List) -> np.ndarray:
    """Stack a list of proto tensors along a new batch dimension."""
    arrays = [proto_to_numpy(t) for t in proto_tensors]
    return np.stack(arrays, axis=0)
