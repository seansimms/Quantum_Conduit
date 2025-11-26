"""FIR filter design using window method.

Implements window method for designing lowpass, highpass, bandpass, and
bandstop FIR filters, plus efficient application via convolution.
"""

from typing import Tuple, Union

import numpy as np

from .utils import check_1d_array
from .windows import blackman, hamming, hann, rectangular


def fir_window_design(
    numtaps: int,
    cutoff: Union[float, Tuple[float, float]],
    fs: float = 1.0,
    window: str = "hann",
    pass_type: str = "lowpass",
) -> np.ndarray:
    """Design FIR filter using window method.

    Designs FIR filter by windowing ideal sinc-based impulse response.

    Args:
        numtaps: Number of filter taps (must be odd for highpass/bandstop).
        cutoff: Cutoff frequency in Hz (single value for low/highpass,
               tuple (low, high) for bandpass/bandstop).
        fs: Sampling frequency in Hz (default: 1.0).
        window: Window type: "hann", "hamming", "blackman", "rectangular"
                (default: "hann").
        pass_type: Filter type: "lowpass", "highpass", "bandpass", "bandstop"
                   (default: "lowpass").

    Returns:
        FIR filter coefficients b (length numtaps).

    Raises:
        ValueError: If parameters are invalid.
    """
    if numtaps <= 0:
        raise ValueError(f"numtaps must be positive, got {numtaps}")

    if pass_type in ("highpass", "bandstop") and numtaps % 2 == 0:
        raise ValueError(
            f"numtaps must be odd for {pass_type} filters, got {numtaps}"
        )

    # Normalize cutoff frequencies
    if isinstance(cutoff, (tuple, list)):
        cutoff_low, cutoff_high = cutoff
        cutoff_low = cutoff_low / fs
        cutoff_high = cutoff_high / fs
        if cutoff_low <= 0 or cutoff_high >= 0.5 or cutoff_low >= cutoff_high:
            raise ValueError(
                f"Invalid cutoff frequencies: ({cutoff_low*fs}, {cutoff_high*fs})"
            )
    else:
        cutoff_norm = cutoff / fs
        if cutoff_norm <= 0 or cutoff_norm >= 0.5:
            raise ValueError(f"Invalid cutoff frequency: {cutoff}")

    # Generate window
    window_funcs = {
        "hann": hann,
        "hamming": hamming,
        "blackman": blackman,
        "rectangular": rectangular,
    }
    if window not in window_funcs:
        raise ValueError(f"Unknown window: {window}")
    win = window_funcs[window](numtaps, periodic=False)

    # Design ideal impulse response
    n = np.arange(numtaps, dtype=float)
    center = (numtaps - 1) / 2.0

    if pass_type == "lowpass":
        # Ideal lowpass: h[n] = 2*fc * sinc(2*fc*(n - center))
        fc = cutoff_norm
        h_ideal = 2.0 * fc * np.sinc(2.0 * fc * (n - center))

    elif pass_type == "highpass":
        # Ideal highpass: h[n] = delta[n] - 2*fc * sinc(2*fc*(n - center))
        fc = cutoff_norm
        h_ideal = -2.0 * fc * np.sinc(2.0 * fc * (n - center))
        h_ideal[int(center)] += 1.0

    elif pass_type == "bandpass":
        # Ideal bandpass: difference of two lowpass filters
        fc_low = cutoff_low
        fc_high = cutoff_high
        h_low = 2.0 * fc_low * np.sinc(2.0 * fc_low * (n - center))
        h_high = 2.0 * fc_high * np.sinc(2.0 * fc_high * (n - center))
        h_ideal = h_high - h_low

    else:  # bandstop
        # Ideal bandstop: sum of lowpass and highpass
        fc_low = cutoff_low
        fc_high = cutoff_high
        h_low = 2.0 * fc_low * np.sinc(2.0 * fc_low * (n - center))
        h_high = 2.0 * fc_high * np.sinc(2.0 * fc_high * (n - center))
        h_ideal = h_low - h_high
        h_ideal[int(center)] += 1.0

    # Apply window
    b = h_ideal * win

    # Normalize for unity DC gain (for lowpass/bandpass)
    if pass_type in ("lowpass", "bandpass"):
        dc_gain = np.sum(b)
        if abs(dc_gain) > 1e-10:
            b = b / dc_gain

    return b


def apply_fir(x: np.ndarray, b: np.ndarray, method: str = "fft") -> np.ndarray:
    """Apply FIR filter to signal.

    Filters input signal using FIR coefficients via convolution.

    Args:
        x: Input signal (1D array).
        b: FIR filter coefficients (1D array).
        method: Convolution method: "fft" or "direct" (default: "fft").

    Returns:
        Filtered signal (same length as input, using 'same' mode).
    """
    x = check_1d_array(x)
    b = check_1d_array(b)

    if method == "fft":
        from .conv import fft_convolve

        return fft_convolve(x, b, mode="same")
    elif method == "direct":
        from .conv import convolve

        return convolve(x, b, mode="same")
    else:
        raise ValueError(f"Unknown method: {method}")

