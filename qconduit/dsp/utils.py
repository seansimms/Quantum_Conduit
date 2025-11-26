"""Utility functions for signal processing.

Provides helper routines for input validation, frequency analysis, and
numerical utilities.
"""

from typing import Tuple

import numpy as np


def check_1d_array(x) -> np.ndarray:
    """Validate and cast input to 1D float64 array.

    Args:
        x: Input array-like object.

    Returns:
        1D float64 numpy array.

    Raises:
        ValueError: If input is not 1D, contains NaN, or contains Inf.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D array")
    if np.any(np.isnan(arr)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError("Input contains Inf values")
    return arr


def next_pow2(n: int) -> int:
    """Return the next power-of-two >= n.

    Args:
        n: Positive integer.

    Returns:
        Smallest power-of-two >= n. Returns 1 if n <= 0.
    """
    if n <= 0:
        return 1
    if n & (n - 1) == 0:  # Already a power of 2
        return n
    return 1 << (n - 1).bit_length()


def freqz(
    b: np.ndarray, a: np.ndarray = np.array([1.0]), worN: int = 512, fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frequency response of a digital filter.

    For FIR filters (a=1), uses FFT of impulse response.
    For IIR filters, samples the DTFT using FFT of long impulse response.

    Args:
        b: Numerator coefficients (FIR or IIR).
        a: Denominator coefficients (default: [1.0] for FIR).
        worN: Number of frequency points (default: 512).
        fs: Sampling frequency in Hz (default: 1.0).

    Returns:
        Tuple (w, h) where:
        - w: Frequency array in Hz (0 to fs/2).
        - h: Complex frequency response H(e^(jw)).
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    # Ensure 1D arrays
    if b.ndim == 0:
        b = b.reshape(1)
    if a.ndim == 0:
        a = a.reshape(1)

    # For FIR (a = [1.0]), use FFT directly
    if len(a) == 1 and np.allclose(a, 1.0):
        # Zero-pad to worN points
        h_fft = np.fft.rfft(b, n=worN)
        w = np.fft.rfftfreq(worN, 1.0 / fs)
        return w, h_fft

    # For IIR, compute impulse response via long convolution
    # Generate impulse
    impulse_len = max(512, len(b) + len(a) * 4)
    impulse = np.zeros(impulse_len)
    impulse[0] = 1.0

    # Filter impulse to get impulse response
    # Use direct difference equation for stability
    h_imp = np.zeros(impulse_len, dtype=float)
    # Initialize state
    x_buf = np.zeros(len(b))
    y_buf = np.zeros(len(a) - 1)

    for n in range(impulse_len):
        # Shift input buffer
        x_buf = np.roll(x_buf, 1)
        x_buf[0] = impulse[n]

        # Compute output: y[n] = sum(b[i]*x[n-i]) - sum(a[j]*y[n-j])
        y_n = np.dot(b, x_buf)
        if len(y_buf) > 0:
            y_n -= np.dot(a[1:], y_buf)

        h_imp[n] = y_n

        # Shift output buffer
        if len(y_buf) > 0:
            y_buf = np.roll(y_buf, 1)
            y_buf[0] = y_n

    # Take FFT of impulse response (real input -> complex output)
    h_fft = np.fft.rfft(h_imp, n=worN)
    w = np.fft.rfftfreq(worN, 1.0 / fs)

    return w, h_fft


def design_frequency_grid(n_fft: int, fs: float) -> np.ndarray:
    """Generate frequency grid for FFT analysis.

    Args:
        n_fft: FFT size.
        fs: Sampling frequency in Hz.

    Returns:
        Frequency array in Hz from 0 to fs/2 (Nyquist).
    """
    return np.fft.rfftfreq(n_fft, 1.0 / fs)

