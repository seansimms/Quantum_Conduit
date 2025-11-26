"""Short-Time Fourier Transform (STFT) and spectrogram.

Implements STFT, inverse STFT, and spectrogram computation with proper
windowing and overlap-add reconstruction.
"""

from typing import Optional

import numpy as np

from .utils import check_1d_array
from .windows import hann


class STFT:
    """Short-Time Fourier Transform processor.

    Encapsulates STFT parameters and provides transform/inverse methods
    with proper windowing and overlap-add reconstruction.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        window: Optional[np.ndarray] = None,
        center: bool = True,
    ):
        """Initialize STFT processor.

        Args:
            n_fft: FFT size (default: 1024).
            hop_length: Hop size between frames (default: n_fft // 4).
            window: Analysis window array (default: Hann window of length n_fft).
            center: If True, pad signal symmetrically at start/end (default: True).
        """
        if n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {n_fft}")
        self.n_fft = n_fft

        if hop_length is None:
            hop_length = n_fft // 4
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        self.hop_length = hop_length

        if window is None:
            window = hann(n_fft, periodic=True)
        else:
            window = check_1d_array(window)
            if len(window) != n_fft:
                raise ValueError(
                    f"Window length ({len(window)}) must equal n_fft ({n_fft})"
                )
        self.window = window
        self.center = center

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Compute STFT of input signal.

        Args:
            x: Input signal (1D array).

        Returns:
            Complex spectrogram of shape (n_freq, n_frames) where
            n_freq = n_fft // 2 + 1 and n_frames is number of frames.
        """
        x = check_1d_array(x)

        # Pad if center=True
        if self.center:
            pad_width = self.n_fft // 2
            x = np.pad(x, pad_width, mode="constant", constant_values=0.0)

        # Compute number of frames
        n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
        if len(x) < self.n_fft:
            n_frames = 1

        # Initialize spectrogram
        n_freq = self.n_fft // 2 + 1
        S = np.zeros((n_freq, n_frames), dtype=complex)

        # Process each frame
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            if end > len(x):
                # Last frame: zero-pad if needed
                frame = np.zeros(self.n_fft, dtype=float)
                frame[: len(x) - start] = x[start:]
            else:
                frame = x[start:end]

            # Apply window
            frame_windowed = frame * self.window

            # Compute FFT
            S[:, i] = np.fft.rfft(frame_windowed, n=self.n_fft)

        return S

    def inverse(self, S: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """Inverse STFT to reconstruct time signal.

        Uses overlap-add reconstruction with proper window normalization.

        Args:
            S: Complex spectrogram of shape (n_freq, n_frames).
            length: Desired output length (default: inferred from frames).

        Returns:
            Reconstructed time signal (1D array).
        """
        if S.ndim != 2:
            raise ValueError(f"Spectrogram must be 2D, got {S.ndim}D")
        n_freq, n_frames = S.shape

        if n_freq != self.n_fft // 2 + 1:
            raise ValueError(
                f"Spectrogram frequency dimension ({n_freq}) does not match "
                f"n_fft ({self.n_fft})"
            )

        # Determine output length
        if length is None:
            length = (n_frames - 1) * self.hop_length + self.n_fft
        if self.center:
            # Account for padding
            pad_width = self.n_fft // 2
            length = length - 2 * pad_width

        # Initialize output with extra space for overlap
        output_len = (n_frames - 1) * self.hop_length + self.n_fft
        x_recon = np.zeros(output_len, dtype=float)

        # Window sum for normalization (overlap-add compensation)
        win_sums = np.zeros(output_len, dtype=float)

        # Reconstruct each frame
        for i in range(n_frames):
            start = i * self.hop_length

            # Inverse FFT
            frame = np.fft.irfft(S[:, i], n=self.n_fft)

            # Apply synthesis window (same as analysis window)
            frame_windowed = frame * self.window

            # Add to output
            end = start + self.n_fft
            x_recon[start:end] += frame_windowed
            win_sums[start:end] += self.window**2

        # Normalize by window sum (avoid division by zero)
        win_sums = np.maximum(win_sums, 1e-10)
        x_recon = x_recon / win_sums

        # Remove center padding if applied
        if self.center:
            pad_width = self.n_fft // 2
            x_recon = x_recon[pad_width:-pad_width] if pad_width > 0 else x_recon

        # Crop to desired length
        if length is not None and len(x_recon) > length:
            x_recon = x_recon[:length]

        return x_recon


def stft(
    x: np.ndarray,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
    center: bool = True,
) -> np.ndarray:
    """Compute Short-Time Fourier Transform.

    Convenience function for STFT computation.

    Args:
        x: Input signal (1D array).
        n_fft: FFT size (default: 1024).
        hop_length: Hop size between frames (default: n_fft // 4).
        window: Analysis window array (default: Hann window).
        center: If True, pad signal symmetrically (default: True).

    Returns:
        Complex spectrogram of shape (n_freq, n_frames).
    """
    stft_obj = STFT(n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    return stft_obj.transform(x)


def istft(
    S: np.ndarray,
    hop_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
    length: Optional[int] = None,
) -> np.ndarray:
    """Inverse Short-Time Fourier Transform.

    Convenience function for inverse STFT.

    Args:
        S: Complex spectrogram of shape (n_freq, n_frames).
        hop_length: Hop size (must match STFT, inferred from S if None).
        window: Synthesis window (must match STFT analysis window).
        length: Desired output length (default: inferred).

    Returns:
        Reconstructed time signal (1D array).
    """
    if S.ndim != 2:
        raise ValueError(f"Spectrogram must be 2D, got {S.ndim}D")
    n_freq, n_frames = S.shape

    # Infer n_fft from frequency dimension
    n_fft = (n_freq - 1) * 2

    if hop_length is None:
        hop_length = n_fft // 4

    if window is None:
        window = hann(n_fft, periodic=True)

    stft_obj = STFT(n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    return stft_obj.inverse(S, length=length)


def spectrogram(
    x: np.ndarray,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
    power: float = 2.0,
) -> np.ndarray:
    """Compute magnitude spectrogram.

    Args:
        x: Input signal (1D array).
        n_fft: FFT size (default: 1024).
        hop_length: Hop size between frames (default: n_fft // 4).
        window: Analysis window array (default: Hann window).
        power: Power to raise magnitude (default: 2.0 for power spectrogram).

    Returns:
        Magnitude spectrogram of shape (n_freq, n_frames).
    """
    S = stft(x, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    return np.abs(S) ** power

