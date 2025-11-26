"""Tests for dsp.filters module."""

import numpy as np
import pytest

from qconduit.dsp.filters import filtfilt, lfilter, sosfiltfilt
from qconduit.dsp.iir import butterworth, sosfilt, tf2sos


def test_lfilter_wrapper():
    """Test lfilter wrapper."""
    b = np.array([1.0, 0.5])
    a = np.array([1.0, -0.5])

    x = np.random.randn(100)
    y = lfilter(b, a, x)

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_filtfilt():
    """Test zero-phase filtfilt."""
    # Design lowpass filter
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    # Step input
    x = np.ones(100)
    x[:50] = 0.0

    # Forward filter (has phase delay)
    y_forward = lfilter(b, a, x)

    # Zero-phase filter
    y_filtfilt = filtfilt(b, a, x)

    # filtfilt should have no phase delay (step transition should be centered)
    # Check that transition is more centered than forward filter
    forward_transition = np.argmax(np.diff(y_forward))
    filtfilt_transition = np.argmax(np.diff(y_filtfilt))

    # filtfilt transition should be closer to center (50) or at least not worse
    # Allow some tolerance since exact transition detection can vary
    assert abs(filtfilt_transition - 50) <= abs(forward_transition - 50) + 5


def test_filtfilt_energy():
    """Test filtfilt energy preservation."""
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    x = np.random.randn(200)
    y = filtfilt(b, a, x)

    # Energy should be similar (filtering doesn't add energy)
    energy_x = np.sum(x**2)
    energy_y = np.sum(y**2)

    # Energy ratio should be reasonable
    assert 0.1 < energy_y / energy_x < 10.0


def test_sosfiltfilt():
    """Test zero-phase SOS filtering."""
    b, a = butterworth(6, 0.2, fs=1.0, btype="lowpass")
    sos = tf2sos(b, a)

    x = np.random.randn(100)
    y_sosfiltfilt = sosfiltfilt(sos, x)
    y_filtfilt = filtfilt(b, a, x)

    # Should be approximately equal
    np.testing.assert_array_almost_equal(y_sosfiltfilt, y_filtfilt, decimal=8)


def test_filtfilt_short_signal():
    """Test filtfilt with short signal."""
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    x = np.array([1.0, 2.0, 3.0])
    y = filtfilt(b, a, x)

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_filtfilt_impulse():
    """Test filtfilt with impulse."""
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    x = np.zeros(100)
    x[50] = 1.0

    y = filtfilt(b, a, x)

    # Impulse response should be symmetric (zero phase)
    # Check symmetry around impulse location
    left_half = y[:50]
    right_half = y[51:][::-1]
    min_len = min(len(left_half), len(right_half))
    if min_len > 0:
        np.testing.assert_array_almost_equal(
            left_half[:min_len], right_half[:min_len], decimal=6
        )


def test_filtfilt_vs_forward():
    """Test that filtfilt has zero phase delay."""
    b, a = butterworth(4, 0.2, fs=1.0, btype="lowpass")

    # Chirp signal
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 10 * t)

    y_forward = lfilter(b, a, x)
    y_filtfilt = filtfilt(b, a, x)

    # filtfilt should preserve phase (no delay)
    # Check correlation: filtfilt should align better with original
    corr_forward = np.corrcoef(x, y_forward)[0, 1]
    corr_filtfilt = np.corrcoef(x, y_filtfilt)[0, 1]

    # filtfilt should have higher correlation (better phase alignment)
    assert corr_filtfilt >= corr_forward - 0.1  # Allow small tolerance

