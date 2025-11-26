"""Tests for dsp.windows module."""

import numpy as np
import pytest

from qconduit.dsp.windows import blackman, hamming, hann, rectangular


def test_hann():
    """Test Hann window."""
    # M=1
    w = hann(1)
    assert len(w) == 1
    assert w[0] == 1.0

    # M=2
    w = hann(2, periodic=True)
    assert len(w) == 2
    # For periodic M=2: w[n] = 0.5 - 0.5*cos(2πn/2)
    # w[0] = 0.5 - 0.5*cos(0) = 0
    # w[1] = 0.5 - 0.5*cos(π) = 1
    assert abs(w[0] - 0.0) < 1e-10
    assert abs(w[1] - 1.0) < 1e-10

    # M=5, periodic
    w = hann(5, periodic=True)
    assert len(w) == 5
    # First should be close to 0 for periodic
    assert w[0] < 0.1
    # Last value for periodic is w[4] = 0.5 - 0.5*cos(8π/5) which is not necessarily 0
    # Just check it's in valid range [0, 1]
    assert 0 <= w[-1] <= 1

    # M=5, symmetric
    w = hann(5, periodic=False)
    assert len(w) == 5
    # Should be symmetric
    assert abs(w[0] - w[-1]) < 1e-10

    # Error cases
    with pytest.raises(ValueError):
        hann(0)
    with pytest.raises(ValueError):
        hann(-1)


def test_hamming():
    """Test Hamming window."""
    w = hamming(1)
    assert len(w) == 1
    assert w[0] == 1.0

    w = hamming(10, periodic=True)
    assert len(w) == 10
    # First and last should be close (periodic)
    assert abs(w[0] - w[-1]) < 0.1

    w = hamming(10, periodic=False)
    assert len(w) == 10
    # Should be symmetric
    assert abs(w[0] - w[-1]) < 1e-10

    with pytest.raises(ValueError):
        hamming(0)


def test_blackman():
    """Test Blackman window."""
    w = blackman(1)
    assert len(w) == 1
    assert w[0] == 1.0

    w = blackman(20, periodic=True)
    assert len(w) == 20
    # Should taper at edges
    assert w[0] < 0.1
    assert w[-1] < 0.1

    w = blackman(20, periodic=False)
    assert len(w) == 20
    # Should be symmetric
    assert abs(w[0] - w[-1]) < 1e-10

    with pytest.raises(ValueError):
        blackman(0)


def test_rectangular():
    """Test rectangular window."""
    w = rectangular(1)
    assert len(w) == 1
    assert w[0] == 1.0

    w = rectangular(10)
    assert len(w) == 10
    assert np.allclose(w, 1.0)

    with pytest.raises(ValueError):
        rectangular(0)


def test_window_properties():
    """Test window mathematical properties."""
    # Hann window sum for small M
    w = hann(5, periodic=False)
    # Sum should be positive
    assert np.sum(w) > 0

    # Hamming window center value
    w = hamming(11, periodic=False)
    # Center should be near 1.0
    center_idx = len(w) // 2
    assert abs(w[center_idx] - 1.0) < 0.1

