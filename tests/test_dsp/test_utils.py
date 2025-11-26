"""Tests for dsp.utils module."""

import numpy as np
import pytest

from qconduit.dsp.utils import (
    check_1d_array,
    design_frequency_grid,
    freqz,
    next_pow2,
)


def test_next_pow2():
    """Test next_pow2 function."""
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(4) == 4
    assert next_pow2(5) == 8
    assert next_pow2(15) == 16
    assert next_pow2(16) == 16
    assert next_pow2(17) == 32
    assert next_pow2(0) == 1
    assert next_pow2(-1) == 1


def test_check_1d_array():
    """Test check_1d_array validation."""
    # Valid 1D array
    x = np.array([1.0, 2.0, 3.0])
    result = check_1d_array(x)
    assert result.ndim == 1
    assert result.dtype == float
    np.testing.assert_array_equal(result, x)

    # List input
    result = check_1d_array([1.0, 2.0, 3.0])
    assert result.ndim == 1
    assert result.dtype == float

    # Scalar
    result = check_1d_array(5.0)
    assert result.ndim == 1
    assert len(result) == 1

    # 2D array should raise
    with pytest.raises(ValueError, match="Expected 1D array"):
        check_1d_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # NaN should raise
    with pytest.raises(ValueError, match="NaN"):
        check_1d_array(np.array([1.0, np.nan, 3.0]))

    # Inf should raise
    with pytest.raises(ValueError, match="Inf"):
        check_1d_array(np.array([1.0, np.inf, 3.0]))


def test_freqz_fir():
    """Test freqz for FIR filter."""
    # Simple FIR: b = [1, 0.5], a = [1]
    b = np.array([1.0, 0.5])
    a = np.array([1.0])

    w, h = freqz(b, a, worN=64, fs=1.0)

    assert len(w) == 33  # rfft returns n//2 + 1
    assert len(h) == 33
    assert np.allclose(w[0], 0.0)
    assert np.allclose(w[-1], 0.5)  # Nyquist

    # DC response should be sum of coefficients
    dc_gain = np.sum(b)
    assert abs(h[0] - dc_gain) < 1e-10


def test_freqz_iir():
    """Test freqz for IIR filter."""
    # Simple IIR: b = [1], a = [1, -0.5]
    b = np.array([1.0])
    a = np.array([1.0, -0.5])

    w, h = freqz(b, a, worN=64, fs=1.0)

    assert len(w) == 33
    assert len(h) == 33
    assert np.iscomplexobj(h)


def test_design_frequency_grid():
    """Test design_frequency_grid."""
    n_fft = 1024
    fs = 1000.0

    w = design_frequency_grid(n_fft, fs)

    assert len(w) == n_fft // 2 + 1
    assert np.allclose(w[0], 0.0)
    assert np.allclose(w[-1], fs / 2.0)  # Nyquist

