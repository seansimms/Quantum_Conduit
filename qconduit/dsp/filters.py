"""Convenience filter application functions.

Provides lfilter, filtfilt (zero-phase filtering), and sosfiltfilt
wrappers for easy signal filtering.
"""

import numpy as np

from .iir import lfilter as iir_lfilter
from .iir import sosfilt
from .utils import check_1d_array


def lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter signal using IIR filter (direct form).

    Convenience wrapper for iir.lfilter.

    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.
        x: Input signal (1D array).

    Returns:
        Filtered signal.
    """
    return iir_lfilter(b, a, x)


def filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering via forward-backward filtering.

    Applies filter forward, then backward to eliminate phase distortion.
    Uses symmetric extension at edges.

    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.
        x: Input signal (1D array).

    Returns:
        Zero-phase filtered signal (same length as input).
    """
    x = check_1d_array(x)
    b = check_1d_array(b)
    a = check_1d_array(a)

    # Determine edge padding length (use filter length)
    edge = max(len(b), len(a)) * 3

    # Symmetric extension at edges
    if len(x) > edge:
        # Reflect edges
        x_padded = np.concatenate(
            [
                x[edge:0:-1],  # Reflected start
                x,
                x[-2 : -(edge + 2) : -1],  # Reflected end
            ]
        )
    else:
        # Signal too short, just pad with zeros
        x_padded = np.pad(x, edge, mode="constant", constant_values=0.0)

    # Forward filter
    y_forward = lfilter(b, a, x_padded)

    # Backward filter (reverse signal, keep coefficients same)
    y_backward = lfilter(b, a, y_forward[::-1])

    # Reverse again to get correct orientation
    y = y_backward[::-1]

    # Remove padding
    if len(x) > edge:
        y = y[edge:-edge]
    else:
        y = y[edge : edge + len(x)]

    return y


def sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering using second-order sections.

    Applies SOS filter forward then backward for zero phase.

    Args:
        sos: Second-order sections array of shape (n_sections, 6).
        x: Input signal (1D array).

    Returns:
        Zero-phase filtered signal.
    """
    x = check_1d_array(x)
    sos = np.asarray(sos, dtype=float)

    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"SOS must be shape (n_sections, 6), got {sos.shape}")

    # Determine edge padding
    edge = sos.shape[0] * 3

    # Symmetric extension
    if len(x) > edge:
        x_padded = np.concatenate(
            [
                x[edge:0:-1],
                x,
                x[-2 : -(edge + 2) : -1],
            ]
        )
    else:
        x_padded = np.pad(x, edge, mode="constant", constant_values=0.0)

    # Forward filter
    y_forward = sosfilt(sos, x_padded)

    # Backward filter (reverse SOS and signal)
    sos_rev = sos[::-1, :].copy()
    # Reverse coefficients within each section
    sos_rev[:, [0, 1, 2]] = sos_rev[:, [2, 1, 0]]
    sos_rev[:, [3, 4, 5]] = sos_rev[:, [5, 4, 3]]

    y_backward = sosfilt(sos_rev, y_forward[::-1])
    y = y_backward[::-1]

    # Remove padding
    if len(x) > edge:
        y = y[edge:-edge]
    else:
        y = y[edge : edge + len(x)]

    return y

