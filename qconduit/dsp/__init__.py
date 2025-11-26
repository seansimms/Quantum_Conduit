"""Deterministic Signal Processing Suite (G30).

This package provides canonical deterministic DSP primitives:
- Convolution and correlation (direct and FFT-based)
- FFT-based filtering with overlap-add/overlap-save
- STFT/ISTFT and spectrograms with windowing
- Window functions (Hann, Hamming, Blackman, rectangular)
- FIR filter design (window method) and application
- IIR filter design (Butterworth, Chebyshev I) and stable filtering

All functions are deterministic, NumPy-first, typed, and numerically stable.
"""

from .conv import (
    convolve,
    correlate,
    fft_convolve,
    overlap_add_filter,
    overlap_save_filter,
)
from .filters import filtfilt, lfilter, sosfiltfilt
from .fir import apply_fir, fir_window_design
from .iir import butterworth, cheby1, sosfilt, tf2sos
from .stft import STFT, istft, spectrogram, stft
from .utils import check_1d_array, design_frequency_grid, freqz, next_pow2
from .windows import blackman, hamming, hann, rectangular

__all__ = [
    # Utils
    "check_1d_array",
    "next_pow2",
    "freqz",
    "design_frequency_grid",
    # Windows
    "hann",
    "hamming",
    "blackman",
    "rectangular",
    # Convolution
    "convolve",
    "correlate",
    "fft_convolve",
    "overlap_add_filter",
    "overlap_save_filter",
    # STFT
    "STFT",
    "stft",
    "istft",
    "spectrogram",
    # FIR
    "fir_window_design",
    "apply_fir",
    # IIR
    "butterworth",
    "cheby1",
    "tf2sos",
    "sosfilt",
    # Filters
    "lfilter",
    "filtfilt",
    "sosfiltfilt",
]

