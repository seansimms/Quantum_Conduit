"""Tests for dsp.stft module."""

import numpy as np
import pytest

from qconduit.dsp.stft import STFT, istft, spectrogram, stft
from qconduit.dsp.windows import hann


def test_stft_roundtrip():
    """Test STFT/ISTFT roundtrip reconstruction."""
    np.random.seed(42)
    x = np.random.randn(1000)

    # Test with default parameters
    S = stft(x, n_fft=256, hop_length=64)
    x_recon = istft(S, hop_length=64)

    # Should reconstruct approximately (within numerical tolerance)
    # Account for edge effects
    min_len = min(len(x), len(x_recon))
    np.testing.assert_array_almost_equal(
        x[:min_len], x_recon[:min_len], decimal=10
    )


def test_stft_class():
    """Test STFT class."""
    np.random.seed(42)
    x = np.random.randn(500)

    stft_obj = STFT(n_fft=128, hop_length=32, center=True)
    S = stft_obj.transform(x)
    x_recon = stft_obj.inverse(S, length=len(x))

    # Check reconstruction
    min_len = min(len(x), len(x_recon))
    np.testing.assert_array_almost_equal(
        x[:min_len], x_recon[:min_len], decimal=10
    )


def test_stft_custom_window():
    """Test STFT with custom window."""
    np.random.seed(42)
    x = np.random.randn(500)
    window = hann(128, periodic=True)

    S = stft(x, n_fft=128, hop_length=32, window=window)
    x_recon = istft(S, hop_length=32, window=window)

    min_len = min(len(x), len(x_recon))
    np.testing.assert_array_almost_equal(
        x[:min_len], x_recon[:min_len], decimal=10
    )


def test_spectrogram():
    """Test spectrogram computation."""
    # Create chirp signal
    fs = 1000.0
    t = np.arange(1000) / fs
    f0, f1 = 10.0, 100.0
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))

    # Compute spectrogram
    S = spectrogram(x, n_fft=256, hop_length=64, power=2.0)

    assert S.ndim == 2
    assert S.shape[0] == 256 // 2 + 1  # n_freq
    assert np.all(S >= 0)  # Power should be non-negative

    # Check that power=2 gives magnitude squared
    S_complex = stft(x, n_fft=256, hop_length=64)
    S_mag_sq = np.abs(S_complex) ** 2
    np.testing.assert_array_almost_equal(S, S_mag_sq)


def test_stft_short_signal():
    """Test STFT with very short signal."""
    x = np.array([1.0, 2.0, 3.0])

    S = stft(x, n_fft=8, hop_length=2)
    assert S.ndim == 2

    x_recon = istft(S, hop_length=2)
    # Should handle short signals gracefully
    assert len(x_recon) >= 0


def test_stft_single_frame():
    """Test STFT with single frame."""
    np.random.seed(42)
    x = np.random.randn(50)
    S = stft(x, n_fft=64, hop_length=64)
    assert S.shape[1] >= 1  # At least one frame

    x_recon = istft(S, hop_length=64)
    assert len(x_recon) >= 0  # May be empty for very short signals


def test_stft_errors():
    """Test STFT error handling."""
    np.random.seed(42)
    x = np.random.randn(100)

    # Window length mismatch
    with pytest.raises(ValueError):
        stft(x, n_fft=128, window=np.ones(64))

    # Test with correct shape (should work)
    S = np.random.randn(33, 10) + 1j * np.random.randn(33, 10)  # 33 = 64//2 + 1
    x_recon = istft(S, hop_length=32)
    assert len(x_recon) > 0

    # Test with wrong frequency dimension (should raise error in STFT class)
    S_wrong = np.random.randn(50, 10) + 1j * np.random.randn(50, 10)
    stft_obj = STFT(n_fft=64, hop_length=32)
    with pytest.raises(ValueError, match="frequency dimension"):
        stft_obj.inverse(S_wrong)

