# src/utils/ops.py

# This file is a placeholder for M1-specific optimized operations or general utilities.
# For example, functions that could leverage Metal Performance Shaders via PyTorch,
# or other M1-specific hardware features if accessible from Python.

import numpy as np

def ensure_contiguous_float32(array: np.ndarray) -> np.ndarray:
    """
    Ensures a NumPy array is C-contiguous and of type float32.
    This is often beneficial for performance with libraries like FAISS
    and for M1 SIMD operations.
    """
    if not array.flags['C_CONTIGUOUS'] or array.dtype != np.float32:
        return np.ascontiguousarray(array, dtype=np.float32)
    return array

# Add other M1-specific or general utility functions here as needed.
# For instance, functions for managing memory-mapped files,
# or wrappers for Metal-accelerated computations if you were to implement them.
