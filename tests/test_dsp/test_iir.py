"""Tests for dsp.iir module."""

import numpy as np
import pytest

from qconduit.dsp.iir import butterworth, cheby1, lfilter, sosfilt, tf2sos
from qconduit.dsp.utils import freqz


def test_butterworth_lowpass():
    """Test Butterworth lowpass design."""
    order = 4
    cutoff = 0.2  # Normalized frequency
    fs = 1.0

    b, a = butterworth(order, cutoff, fs=fs, btype="lowpass")

    assert len(b) == order + 1
    assert len(a) == order + 1

    # Check DC gain (should be ~1 for lowpass)
    dc_gain = np.sum(b) / np.sum(a)
    assert abs(dc_gain - 1.0) < 0.1


def test_butterworth_highpass():
    """Test Butterworth highpass design."""
    order = 4
    cutoff = 0.3
    fs = 1.0

    b, a = butterworth(order, cutoff, fs=fs, btype="highpass")

    assert len(b) == order + 1
    assert len(a) == order + 1

    # Highpass should have zero DC gain
    dc_gain = np.sum(b) / np.sum(a)
    assert abs(dc_gain) < 0.1


def test_butterworth_bandpass():
    """Test Butterworth bandpass design."""
    order = 4
    cutoff = (0.2, 0.4)
    fs = 1.0

    b, a = butterworth(order, cutoff, fs=fs, btype="bandpass")

    assert len(b) == 2 * order + 1
    assert len(a) == 2 * order + 1


def test_cheby1_lowpass():
    """Test Chebyshev Type I lowpass design."""
    order = 4
    rp = 1.0  # 1 dB ripple
    cutoff = 0.2
    fs = 1.0

    b, a = cheby1(order, rp, cutoff, fs=fs, btype="lowpass")

    assert len(b) == order + 1
    assert len(a) == order + 1


def test_tf2sos():
    """Test transfer function to SOS conversion."""
    # Simple 2nd order filter
    b = np.array([1.0, 0.5, 0.25])
    a = np.array([1.0, -0.5, 0.25])

    sos = tf2sos(b, a)

    assert sos.ndim == 2
    assert sos.shape[1] == 6
    assert sos.shape[0] >= 1

    # Check normalization (a0 should be 1.0)
    assert np.allclose(sos[:, 3], 1.0)


def test_sosfilt():
    """Test SOS filtering."""
    # Design filter
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")
    sos = tf2sos(b, a)

    # Test signal
    np.random.seed(42)
    x = np.random.randn(100)

    y_sos = sosfilt(sos, x)
    y_direct = lfilter(b, a, x)

    # Should be approximately equal (within numerical tolerance)
    np.testing.assert_array_almost_equal(y_sos, y_direct, decimal=8)


def test_lfilter():
    """Test direct lfilter."""
    # Simple first-order filter
    b = np.array([1.0, 0.5])
    a = np.array([1.0, -0.5])

    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Impulse

    y = lfilter(b, a, x)

    assert len(y) == len(x)
    # Impulse response should decay
    assert abs(y[-1]) < abs(y[0])


def test_butterworth_frequency_response():
    """Test Butterworth frequency response."""
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    w, h = freqz(b, a, worN=256, fs=1.0)

    # DC gain should be ~1
    assert abs(h[0] - 1.0) < 0.1

    # At cutoff (0.2), magnitude should be ~0.707 (-3dB)
    cutoff_idx = int(0.2 * len(w) * 2)
    if cutoff_idx < len(h):
        mag_at_cutoff = abs(h[cutoff_idx])
        assert abs(mag_at_cutoff - 0.707) < 0.2  # Allow some tolerance


def test_cheby1_frequency_response():
    """Test Chebyshev frequency response."""
    b, a = cheby1(4, 1.0, 0.2, fs=1.0, btype="lowpass")

    w, h = freqz(b, a, worN=256, fs=1.0)

    # DC gain should be reasonable
    assert abs(h[0]) > 0.5


def test_iir_errors():
    """Test IIR design error handling."""
    # Invalid order
    with pytest.raises(ValueError):
        butterworth(0, 0.2)

    # Invalid cutoff for bandpass
    with pytest.raises(ValueError):
        butterworth(4, 0.2, btype="bandpass")

    # Invalid ripple
    with pytest.raises(ValueError):
        cheby1(4, -1.0, 0.2)


def test_sosfilt_stability():
    """Test SOS filter stability."""
    # High-order filter (should use SOS automatically)
    b, a = butterworth(10, 0.2, fs=1.0, btype="lowpass")
    sos = tf2sos(b, a)

    # Long signal
    x = np.ones(1000)

    y = sosfilt(sos, x)

    # Should be stable (no NaN or Inf)
    assert np.all(np.isfinite(y))
    # Step response should approach some finite value (DC gain may not be exactly 1)
    assert np.isfinite(y[-1])
    assert abs(y[-1]) < 10.0  # Reasonable bound

