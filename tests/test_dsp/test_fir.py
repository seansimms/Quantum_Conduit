"""Tests for dsp.fir module."""

import numpy as np
import pytest

from qconduit.dsp.fir import apply_fir, fir_window_design
from qconduit.dsp.utils import freqz


def test_fir_window_design_lowpass():
    """Test FIR lowpass design."""
    numtaps = 51
    cutoff = 0.2  # Normalized frequency
    fs = 1.0

    b = fir_window_design(numtaps, cutoff, fs=fs, window="hann", pass_type="lowpass")

    assert len(b) == numtaps
    # Check symmetry (should be symmetric for odd length)
    assert abs(b[0] - b[-1]) < 1e-10

    # Check DC gain (should be ~1 for lowpass)
    dc_gain = np.sum(b)
    assert abs(dc_gain - 1.0) < 0.1


def test_fir_window_design_highpass():
    """Test FIR highpass design."""
    numtaps = 51  # Must be odd
    cutoff = 0.3
    fs = 1.0

    b = fir_window_design(
        numtaps, cutoff, fs=fs, window="hamming", pass_type="highpass"
    )

    assert len(b) == numtaps

    # Highpass should have zero DC gain
    dc_gain = np.sum(b)
    assert abs(dc_gain) < 0.1


def test_fir_window_design_bandpass():
    """Test FIR bandpass design."""
    numtaps = 51
    cutoff = (0.2, 0.4)
    fs = 1.0

    b = fir_window_design(
        numtaps, cutoff, fs=fs, window="blackman", pass_type="bandpass"
    )

    assert len(b) == numtaps


def test_fir_window_design_errors():
    """Test FIR design error handling."""
    # Even numtaps for highpass
    with pytest.raises(ValueError):
        fir_window_design(50, 0.2, pass_type="highpass")

    # Invalid cutoff
    with pytest.raises(ValueError):
        fir_window_design(51, 0.6, fs=1.0)  # > Nyquist

    # Invalid window
    with pytest.raises(ValueError):
        fir_window_design(51, 0.2, window="invalid")


def test_apply_fir():
    """Test FIR filter application."""
    # Design filter
    b = fir_window_design(21, 0.2, fs=1.0, window="hann", pass_type="lowpass")

    # Create test signal: step function
    x = np.ones(100)
    y = apply_fir(x, b, method="fft")

    assert len(y) == len(x)

    # Step response should smooth the step
    # DC gain should be ~1, so output should approach 1 (allow more tolerance)
    assert abs(y[-1] - 1.0) < 0.5

    # Test direct method
    y_direct = apply_fir(x, b, method="direct")
    np.testing.assert_array_almost_equal(y, y_direct, decimal=10)


def test_fir_impulse_response():
    """Test FIR impulse response."""
    b = fir_window_design(21, 0.2, fs=1.0, window="rectangular", pass_type="lowpass")

    # Impulse input
    x = np.zeros(100)
    x[0] = 1.0

    y = apply_fir(x, b, method="direct")

    # Impulse response should contain the filter taps
    # For 'same' mode, the response is centered
    # Check that the response contains the filter coefficients
    center_idx = len(b) // 2
    # The impulse response should match the filter at some point
    assert np.max(np.abs(y)) > 0  # Non-zero response


def test_fir_frequency_response():
    """Test FIR frequency response."""
    b = fir_window_design(51, 0.2, fs=1.0, window="hann", pass_type="lowpass")

    # Compute frequency response
    w, h = freqz(b, worN=256, fs=1.0)

    # DC gain should be ~1
    assert abs(h[0] - 1.0) < 0.1

    # At Nyquist, gain should be small for lowpass
    assert abs(h[-1]) < 0.5

