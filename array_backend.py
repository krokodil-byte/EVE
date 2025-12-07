"""
Array Backend - CuPy with NumPy fallback
Transparent GPU acceleration without changing pipeline logic
"""

import sys

# Try CuPy first (GPU), fallback to NumPy (CPU)
try:
    import cupy as xp
    BACKEND = "cupy"
    GPU_AVAILABLE = True
    print("[Backend] Using CuPy - GPU acceleration enabled", file=sys.stderr)
except ImportError:
    import numpy as xp
    BACKEND = "numpy"
    GPU_AVAILABLE = False
    print("[Backend] Using NumPy - CPU mode", file=sys.stderr)


def to_cpu(array):
    """Convert array to CPU (NumPy)"""
    if BACKEND == "cupy":
        return xp.asnumpy(array)
    return array


def to_gpu(array):
    """Convert array to GPU (CuPy) if available"""
    if BACKEND == "cupy":
        return xp.asarray(array)
    return array


# Export for use in other modules
__all__ = ['xp', 'BACKEND', 'GPU_AVAILABLE', 'to_cpu', 'to_gpu']
