"""
Array Backend - CuPy with NumPy fallback
Transparent GPU acceleration without changing pipeline logic
"""

import sys
import os

# Check user preference for backend
# Priority: ENV var > .eve_backend file > auto
BACKEND_PREFERENCE = os.environ.get('EVE_BACKEND', None)

if BACKEND_PREFERENCE is None:
    # Try to read from .eve_backend file
    try:
        with open('.eve_backend', 'r') as f:
            BACKEND_PREFERENCE = f.read().strip().lower()
    except FileNotFoundError:
        BACKEND_PREFERENCE = 'auto'
else:
    BACKEND_PREFERENCE = BACKEND_PREFERENCE.lower()

# Try CuPy first (GPU), fallback to NumPy (CPU)
if BACKEND_PREFERENCE == 'cpu':
    # Force CPU mode
    import numpy as xp
    BACKEND = "numpy"
    GPU_AVAILABLE = False
    print("[Backend] Forced CPU mode (NumPy)", file=sys.stderr)
else:
    # Auto mode - try GPU first
    try:
        import cupy as xp
        BACKEND = "cupy"
        GPU_AVAILABLE = True
        print("[Backend] Using CuPy - GPU acceleration enabled ✓", file=sys.stderr)
    except ImportError:
        import numpy as xp
        BACKEND = "numpy"
        GPU_AVAILABLE = False
        print("[Backend] Using NumPy - CPU mode (CuPy not installed)", file=sys.stderr)


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
