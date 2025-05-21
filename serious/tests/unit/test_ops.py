# tests/unit/test_ops.py
import numpy as np
import pytest

# Add src to Python path
import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.utils.ops import ensure_contiguous_float32

def test_ensure_contiguous_float32_already_correct():
    arr = np.array([1, 2, 3], dtype=np.float32)
    assert arr.flags['C_CONTIGUOUS']
    processed_arr = ensure_contiguous_float32(arr)
    assert np.array_equal(processed_arr, arr)
    assert processed_arr is arr # Should return same object if no change needed

def test_ensure_contiguous_float32_needs_dtype_conversion():
    arr = np.array([1, 2, 3], dtype=np.int32)
    processed_arr = ensure_contiguous_float32(arr)
    assert processed_arr.dtype == np.float32
    assert np.array_equal(processed_arr, np.array([1., 2., 3.], dtype=np.float32))
    assert processed_arr is not arr

def test_ensure_contiguous_float32_needs_contiguous_conversion():
    base_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    arr = base_arr[:, ::2]  # Creates a non-contiguous view
    assert not arr.flags['C_CONTIGUOUS']
    
    processed_arr = ensure_contiguous_float32(arr)
    assert processed_arr.flags['C_CONTIGUOUS']
    assert processed_arr.dtype == np.float32
    assert np.array_equal(processed_arr, np.array([[0., 2.], [3., 5.]], dtype=np.float32))
    assert processed_arr is not arr

def test_ensure_contiguous_float32_needs_both_conversions():
    base_arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    arr = base_arr[:, ::2]  # Non-contiguous view, non-float32 dtype
    
    processed_arr = ensure_contiguous_float32(arr)
    assert processed_arr.flags['C_CONTIGUOUS']
    assert processed_arr.dtype == np.float32
    assert np.array_equal(processed_arr, np.array([[0., 2.], [3., 5.]], dtype=np.float32))
    assert processed_arr is not arr

