"""Window functions for signal processing.

Implements canonical window functions: Hann, Hamming, Blackman, and rectangular.
All windows follow textbook definitions and support periodic and symmetric forms.
"""

import numpy as np


def hann(M: int, periodic: bool = True) -> np.ndarray:
    """Generate Hann (Hanning) window.

    Hann window: w[n] = 0.5 - 0.5 * cos(2πn/(M-1)) for symmetric
                 w[n] = 0.5 - 0.5 * cos(2πn/M) for periodic

    Args:
        M: Window length (must be positive integer).
        periodic: If True, generate periodic form for FFT use (default: True).

    Returns:
        Window array of length M, dtype float64.

    Raises:
        ValueError: If M <= 0.
    """
    if M <= 0:
        raise ValueError(f"Window length M must be positive, got {M}")
    if M == 1:
        return np.array([1.0], dtype=float)

    n = np.arange(M, dtype=float)
    if periodic:
        # Periodic form: w[n] = 0.5 - 0.5 * cos(2πn/M)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / M)
    else:
        # Symmetric form: w[n] = 0.5 - 0.5 * cos(2πn/(M-1))
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))

    return w


def hamming(M: int, periodic: bool = True) -> np.ndarray:
    """Generate Hamming window.

    Hamming window: w[n] = 0.54 - 0.46 * cos(2πn/(M-1)) for symmetric
                    w[n] = 0.54 - 0.46 * cos(2πn/M) for periodic

    Args:
        M: Window length (must be positive integer).
        periodic: If True, generate periodic form for FFT use (default: True).

    Returns:
        Window array of length M, dtype float64.

    Raises:
        ValueError: If M <= 0.
    """
    if M <= 0:
        raise ValueError(f"Window length M must be positive, got {M}")
    if M == 1:
        return np.array([1.0], dtype=float)

    n = np.arange(M, dtype=float)
    if periodic:
        # Periodic form: w[n] = 0.54 - 0.46 * cos(2πn/M)
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / M)
    else:
        # Symmetric form: w[n] = 0.54 - 0.46 * cos(2πn/(M-1))
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (M - 1))

    return w


def blackman(M: int, periodic: bool = True) -> np.ndarray:
    """Generate Blackman window.

    Blackman window: w[n] = 0.42 - 0.5*cos(2πn/(M-1)) + 0.08*cos(4πn/(M-1)) for symmetric
                     w[n] = 0.42 - 0.5*cos(2πn/M) + 0.08*cos(4πn/M) for periodic

    Args:
        M: Window length (must be positive integer).
        periodic: If True, generate periodic form for FFT use (default: True).

    Returns:
        Window array of length M, dtype float64.

    Raises:
        ValueError: If M <= 0.
    """
    if M <= 0:
        raise ValueError(f"Window length M must be positive, got {M}")
    if M == 1:
        return np.array([1.0], dtype=float)

    n = np.arange(M, dtype=float)
    if periodic:
        # Periodic form
        w = (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * n / M)
            + 0.08 * np.cos(4.0 * np.pi * n / M)
        )
    else:
        # Symmetric form
        w = (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
            + 0.08 * np.cos(4.0 * np.pi * n / (M - 1))
        )

    return w


def rectangular(M: int) -> np.ndarray:
    """Generate rectangular (boxcar) window.

    Rectangular window: w[n] = 1.0 for all n.

    Args:
        M: Window length (must be positive integer).

    Returns:
        Window array of length M, dtype float64, all ones.

    Raises:
        ValueError: If M <= 0.
    """
    if M <= 0:
        raise ValueError(f"Window length M must be positive, got {M}")
    return np.ones(M, dtype=float)

