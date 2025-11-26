"""Convolution and correlation operations.

Implements direct and FFT-based convolution/correlation, plus overlap-add
and overlap-save methods for efficient block-based filtering.
"""

from typing import Optional

import numpy as np

from .utils import check_1d_array, next_pow2


def convolve(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """Direct convolution of two 1D arrays.

    Computes y[n] = sum_k x[k] * h[n-k] using direct O(N*M) method.

    Args:
        x: First input signal (1D array).
        h: Second input signal (impulse response, 1D array).
        mode: Output mode: "full", "same", or "valid" (default: "full").

    Returns:
        Convolved signal.

    Raises:
        ValueError: If mode is not one of the supported modes.
    """
    x = check_1d_array(x)
    h = check_1d_array(h)

    if mode not in ("full", "same", "valid"):
        raise ValueError(f"Unsupported mode: {mode}")

    # Use numpy's convolve for correctness
    result = np.convolve(x, h, mode=mode)
    return result


def correlate(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """Cross-correlation of two 1D arrays.

    Correlation is equivalent to convolution with reversed second argument:
    correlate(x, h) = convolve(x, h[::-1])

    Args:
        x: First input signal (1D array).
        h: Second input signal (1D array).
        mode: Output mode: "full", "same", or "valid" (default: "full").

    Returns:
        Cross-correlated signal.
    """
    x = check_1d_array(x)
    h = check_1d_array(h)
    return convolve(x, h[::-1], mode=mode)


def fft_convolve(
    x: np.ndarray, h: np.ndarray, mode: str = "full", n_fft: Optional[int] = None
) -> np.ndarray:
    """FFT-based convolution using zero-padding.

    Computes convolution via FFT for efficiency on large signals.
    Uses rfft/irfft for real signals.

    Args:
        x: First input signal (1D array).
        h: Second input signal (impulse response, 1D array).
        mode: Output mode: "full", "same", or "valid" (default: "full").
        n_fft: FFT size (default: next power-of-two >= len(x) + len(h) - 1).

    Returns:
        Convolved signal (real-valued).
    """
    x = check_1d_array(x)
    h = check_1d_array(h)

    if mode not in ("full", "same", "valid"):
        raise ValueError(f"Unsupported mode: {mode}")

    len_x = len(x)
    len_h = len(h)

    # Determine output length
    if mode == "full":
        out_len = len_x + len_h - 1
    elif mode == "same":
        out_len = len_x
    else:  # valid
        out_len = len_x - len_h + 1

    # Determine FFT size
    if n_fft is None:
        min_fft = len_x + len_h - 1
        n_fft = next_pow2(min_fft)
    else:
        if n_fft < len_x + len_h - 1:
            raise ValueError(
                f"n_fft ({n_fft}) must be >= len(x) + len(h) - 1 ({len_x + len_h - 1})"
            )

    # Zero-pad and compute FFT
    x_padded = np.zeros(n_fft, dtype=float)
    h_padded = np.zeros(n_fft, dtype=float)
    x_padded[:len_x] = x
    h_padded[:len_h] = h

    # Use rfft for real signals
    X = np.fft.rfft(x_padded)
    H = np.fft.rfft(h_padded)
    Y = X * H
    y_full = np.fft.irfft(Y, n=n_fft)

    # Remove small imaginary residuals
    y_full = np.real_if_close(y_full, tol=1e-12)

    # Crop to desired mode
    if mode == "full":
        return y_full[:out_len]
    elif mode == "same":
        # Center the output
        start = (len(y_full) - out_len) // 2
        return y_full[start : start + out_len]
    else:  # valid
        start = len_h - 1
        return y_full[start : start + out_len]


def overlap_add_filter(
    x: np.ndarray, h: np.ndarray, block_len: Optional[int] = None
) -> np.ndarray:
    """Overlap-add method for block-based filtering.

    Splits input into blocks, filters each block, and adds overlapping outputs.

    Args:
        x: Input signal (1D array).
        h: Filter impulse response (1D array).
        block_len: Block length (default: max(1024, next_pow2(len(h)))).

    Returns:
        Filtered signal (same length as input).
    """
    x = check_1d_array(x)
    h = check_1d_array(h)

    len_h = len(h)
    if block_len is None:
        block_len = max(1024, next_pow2(len_h))
    else:
        block_len = max(block_len, len_h)

    # Determine FFT size for block convolution
    n_fft = next_pow2(block_len + len_h - 1)

    # Initialize output
    output = np.zeros(len(x) + len_h - 1, dtype=float)

    # Process blocks
    i = 0
    while i < len(x):
        # Extract block
        block = x[i : i + block_len]
        if len(block) == 0:
            break

        # Zero-pad block if needed
        if len(block) < block_len:
            block_padded = np.zeros(block_len, dtype=float)
            block_padded[: len(block)] = block
            block = block_padded

        # Filter block using FFT convolution
        block_filtered = fft_convolve(block, h, mode="full", n_fft=n_fft)

        # Add to output with overlap (ensure we don't go out of bounds)
        end_idx = min(i + len(block_filtered), len(output))
        block_len_actual = end_idx - i
        output[i:end_idx] += block_filtered[:block_len_actual]

        i += block_len

    # Return same length as input (crop to 'same' mode)
    if len(output) > len(x):
        start = (len(output) - len(x)) // 2
        return output[start : start + len(x)]
    return output[: len(x)]


def overlap_save_filter(
    x: np.ndarray, h: np.ndarray, block_len: Optional[int] = None
) -> np.ndarray:
    """Overlap-save method for block-based filtering.

    Splits input into overlapping blocks, filters each, and saves non-overlapping
    portions of output.

    Args:
        x: Input signal (1D array).
        h: Filter impulse response (1D array).
        block_len: Block length (default: reasonable size based on filter length).

    Returns:
        Filtered signal (same length as input).
    """
    x = check_1d_array(x)
    h = check_1d_array(h)

    len_h = len(h)
    if block_len is None:
        # Choose reasonable block size
        block_len = max(256, min(4 * len_h, 2048))
        block_len = next_pow2(block_len)
    else:
        block_len = max(block_len, len_h)

    # Overlap length is len_h - 1
    overlap = len_h - 1
    step = block_len - overlap

    # Determine FFT size for block convolution
    n_fft = next_pow2(block_len + len_h - 1)

    # Pad input at the beginning with zeros (overlap samples)
    x_padded = np.zeros(len(x) + overlap, dtype=float)
    x_padded[overlap:] = x

    # Initialize output array (pre-allocate for efficiency)
    total_output_samples = len(x)
    output = np.zeros(total_output_samples, dtype=float)
    output_idx = 0

    # Process blocks
    i = 0
    while i < len(x_padded) and output_idx < total_output_samples:
        # Extract block (with overlap from previous block)
        end_idx = min(i + block_len, len(x_padded))
        block = x_padded[i:end_idx]

        if len(block) < len_h:
            break

        # Zero-pad to block_len if needed (for last block)
        if len(block) < block_len:
            block_padded = np.zeros(block_len, dtype=float)
            block_padded[: len(block)] = block
            block = block_padded

        # Filter block using full convolution
        block_filtered = fft_convolve(block, h, mode="full", n_fft=n_fft)

        # For overlap-save: discard first 'overlap' samples (corrupted by circular conv)
        # Save next 'step' samples (valid linear convolution results)
        if len(block_filtered) > overlap:
            save_start = overlap
            save_end = min(overlap + step, len(block_filtered))
            save_len = save_end - save_start

            # Copy to output
            copy_len = min(save_len, total_output_samples - output_idx)
            output[output_idx : output_idx + copy_len] = block_filtered[
                save_start : save_start + copy_len
            ]
            output_idx += copy_len

        i += step
        if i >= len(x_padded):
            break

    return output

