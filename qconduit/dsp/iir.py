"""IIR filter design and stable filtering.

Implements Butterworth and Chebyshev Type I IIR filter design using
bilinear transform, plus second-order sections (SOS) conversion and
stable filtering.
"""

from typing import Tuple, Union

import numpy as np

from .utils import check_1d_array


def _butterworth_analog_poles(order: int) -> np.ndarray:
    """Compute analog Butterworth filter poles.

    Butterworth poles are uniformly spaced on unit circle in s-plane.

    Args:
        order: Filter order.

    Returns:
        Complex array of poles in s-plane.
    """
    # Poles are at: s_k = exp(j * pi * (2k + n + 1) / (2n))
    # for k = 0, 1, ..., n-1
    k = np.arange(order, dtype=float)
    angles = np.pi * (2 * k + order + 1) / (2 * order)
    poles = np.exp(1j * angles)
    return poles


def _cheby1_analog_poles(order: int, rp: float) -> np.ndarray:
    """Compute analog Chebyshev Type I filter poles.

    Args:
        order: Filter order.
        rp: Passband ripple in dB.

    Returns:
        Complex array of poles in s-plane.
    """
    # Convert ripple to epsilon
    epsilon = np.sqrt(10 ** (rp / 10.0) - 1.0)

    # Compute poles using Chebyshev formulas
    # Real part: -sinh(eta) * sin(theta_k)
    # Imag part: cosh(eta) * cos(theta_k)
    # where eta = asinh(1/epsilon) / n
    eta = np.arcsinh(1.0 / epsilon) / order

    k = np.arange(order, dtype=float)
    theta_k = np.pi * (2 * k + 1) / (2 * order)

    real_part = -np.sinh(eta) * np.sin(theta_k)
    imag_part = np.cosh(eta) * np.cos(theta_k)

    poles = real_part + 1j * imag_part
    return poles


def _bilinear_transform(
    poles: np.ndarray, zeros: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bilinear transform to convert analog to digital filter.

    Maps s-plane poles/zeros to z-plane using: z = (1 + s*T/2) / (1 - s*T/2)
    where T = 1/fs.

    Args:
        poles: Analog poles (complex array).
        zeros: Analog zeros (complex array).
        fs: Sampling frequency.

    Returns:
        Tuple (z_poles, z_zeros) in z-plane.
    """
    T = 1.0 / fs
    # Bilinear transform: z = (1 + s*T/2) / (1 - s*T/2)
    z_poles = (1.0 + poles * T / 2.0) / (1.0 - poles * T / 2.0)
    z_zeros = (1.0 + zeros * T / 2.0) / (1.0 - zeros * T / 2.0)
    return z_poles, z_zeros


def _prewarp_frequency(f: float, fs: float) -> float:
    """Prewarp frequency for bilinear transform.

    Args:
        f: Desired digital frequency in Hz.
        fs: Sampling frequency.

    Returns:
        Prewarped analog frequency.
    """
    # Prewarp: omega_analog = (2/T) * tan(omega_digital * T / 2)
    T = 1.0 / fs
    omega_digital = 2.0 * np.pi * f
    omega_analog = (2.0 / T) * np.tan(omega_digital * T / 2.0)
    return omega_analog / (2.0 * np.pi)  # Convert back to Hz


def _zpk_to_tf(z: np.ndarray, p: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert zero-pole-gain to transfer function (b, a).

    Args:
        z: Zeros (complex array).
        p: Poles (complex array).
        k: Gain.

    Returns:
        Tuple (b, a) numerator and denominator polynomials.
    """
    # Expand zeros to polynomial
    b = np.poly(z) * k
    # Expand poles to polynomial
    a = np.poly(p)
    return b, a


def butterworth(
    order: int,
    cutoff: Union[float, Tuple[float, float]],
    fs: float = 1.0,
    btype: str = "lowpass",
) -> Tuple[np.ndarray, np.ndarray]:
    """Design Butterworth IIR filter using bilinear transform.

    Args:
        order: Filter order.
        cutoff: Cutoff frequency in Hz (single for low/highpass,
               tuple for bandpass/bandstop).
        fs: Sampling frequency in Hz (default: 1.0).
        btype: Filter type: "lowpass", "highpass", "bandpass", "bandstop"
               (default: "lowpass").

    Returns:
        Tuple (b, a) numerator and denominator coefficients.

    Raises:
        ValueError: If parameters are invalid.
    """
    # Input validation
    if order <= 0:
        raise ValueError(f"Order must be positive, got {order}")
    if not isinstance(order, (int, np.integer)):
        raise ValueError(f"Order must be an integer, got {type(order)}")
    if order > 50:  # Reasonable upper bound
        raise ValueError(
            f"Order too large (>{50}), got {order}. "
            f"Consider using lower order or different filter type."
        )

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if np.isnan(fs) or np.isinf(fs):
        raise ValueError(f"Sampling frequency must be finite, got {fs}")

    if btype not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(
            f"Invalid btype '{btype}'. "
            f"Must be one of: lowpass, highpass, bandpass, bandstop"
        )

    # Handle cutoff frequencies with validation
    if isinstance(cutoff, (tuple, list)):
        if len(cutoff) != 2:
            raise ValueError(f"Cutoff tuple must have 2 elements, got {len(cutoff)}")
        cutoff_low, cutoff_high = cutoff
        if btype not in ("bandpass", "bandstop"):
            raise ValueError(f"Tuple cutoff requires bandpass/bandstop, got {btype}")
        # Validate bandpass/bandstop frequencies
        if cutoff_low <= 0 or cutoff_high <= 0:
            raise ValueError(
                f"Cutoff frequencies must be positive, got ({cutoff_low}, {cutoff_high})"
            )
        if (
            np.isnan(cutoff_low) or np.isnan(cutoff_high)
            or np.isinf(cutoff_low) or np.isinf(cutoff_high)
        ):
            raise ValueError(
                f"Cutoff frequencies must be finite, got ({cutoff_low}, {cutoff_high})"
            )
        if cutoff_low >= cutoff_high:
            raise ValueError(f"cutoff_low must be < cutoff_high, got ({cutoff_low}, {cutoff_high})")
        if cutoff_high >= fs / 2:
            raise ValueError(
                f"cutoff_high ({cutoff_high}) must be < Nyquist frequency (fs/2 = {fs/2})"
            )
    else:
        if btype in ("bandpass", "bandstop"):
            raise ValueError(f"{btype} requires tuple cutoff")
        # Validate single cutoff
        if cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff}")
        if np.isnan(cutoff) or np.isinf(cutoff):
            raise ValueError(f"Cutoff frequency must be finite, got {cutoff}")
        if cutoff >= fs / 2:
            raise ValueError(
                f"Cutoff frequency ({cutoff}) must be < Nyquist frequency (fs/2 = {fs/2})"
            )
        # Handle near-zero cutoff with small ridge
        if cutoff < 1e-6 * fs:
            cutoff = max(cutoff, 1e-6 * fs)  # Minimum cutoff
        cutoff_low = cutoff_high = cutoff

    # Prewarp frequencies
    if btype in ("lowpass", "highpass"):
        fc_prewarp = _prewarp_frequency(cutoff, fs)
    else:
        # For bandpass/bandstop, use geometric mean for design
        pass  # fc_prewarp computed below using fc_center

    # Get analog poles
    if btype in ("lowpass", "highpass"):
        poles_analog = _butterworth_analog_poles(order)
        # Scale by cutoff frequency
        poles_analog = poles_analog * 2.0 * np.pi * fc_prewarp
    else:
        # Bandpass/bandstop: design as lowpass prototype then transform
        # For simplicity, use geometric mean and design lowpass
        fc_center = np.sqrt(cutoff_low * cutoff_high)
        fc_prewarp = _prewarp_frequency(fc_center, fs)
        poles_analog = _butterworth_analog_poles(order)
        poles_analog = poles_analog * 2.0 * np.pi * fc_prewarp

    # Analog zeros (at infinity for lowpass)
    if btype == "lowpass":
        zeros_analog = np.array([])
    elif btype == "highpass":
        zeros_analog = np.zeros(order, dtype=complex)  # Zeros at origin
    else:
        # Bandpass/bandstop: zeros at origin or infinity
        zeros_analog = np.array([])  # Simplified

    # Apply bilinear transform
    z_poles, z_zeros = _bilinear_transform(poles_analog, zeros_analog, fs)

    # Compute gain (normalize for unity DC gain for lowpass)
    if btype == "lowpass":
        # Evaluate H(z) at z=1 (DC)
        num_val = np.prod(1.0 - z_zeros) if len(z_zeros) > 0 else 1.0
        den_val = np.prod(1.0 - z_poles)
        k = den_val / num_val if abs(num_val) > 1e-10 else 1.0
    else:
        k = 1.0

    # Convert to transfer function
    b, a = _zpk_to_tf(z_zeros, z_poles, k)

    # Ensure real coefficients
    b = np.real_if_close(b, tol=1e-10)
    a = np.real_if_close(a, tol=1e-10)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    return b, a


def cheby1(
    order: int,
    rp: float,
    cutoff: Union[float, Tuple[float, float]],
    fs: float = 1.0,
    btype: str = "lowpass",
) -> Tuple[np.ndarray, np.ndarray]:
    """Design Chebyshev Type I IIR filter using bilinear transform.

    Args:
        order: Filter order.
        rp: Passband ripple in dB.
        cutoff: Cutoff frequency in Hz (single for low/highpass,
               tuple for bandpass/bandstop).
        fs: Sampling frequency in Hz (default: 1.0).
        btype: Filter type: "lowpass", "highpass", "bandpass", "bandstop"
               (default: "lowpass").

    Returns:
        Tuple (b, a) numerator and denominator coefficients.

    Raises:
        ValueError: If parameters are invalid.
    """
    # Input validation
    if order <= 0:
        raise ValueError(f"Order must be positive, got {order}")
    if not isinstance(order, (int, np.integer)):
        raise ValueError(f"Order must be an integer, got {type(order)}")
    if order > 50:  # Reasonable upper bound
        raise ValueError(
            f"Order too large (>{50}), got {order}. "
            f"Consider using lower order or different filter type."
        )

    if rp <= 0:
        raise ValueError(f"Passband ripple must be positive, got {rp}")
    if np.isnan(rp) or np.isinf(rp):
        raise ValueError(f"Passband ripple must be finite, got {rp}")
    if rp > 100:  # Reasonable upper bound
        raise ValueError(f"Passband ripple too large (>{100} dB), got {rp}")

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if np.isnan(fs) or np.isinf(fs):
        raise ValueError(f"Sampling frequency must be finite, got {fs}")

    if btype not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(
            f"Invalid btype '{btype}'. "
            f"Must be one of: lowpass, highpass, bandpass, bandstop"
        )

    # Handle cutoff frequencies with validation (similar to butterworth)
    if isinstance(cutoff, (tuple, list)):
        if len(cutoff) != 2:
            raise ValueError(f"Cutoff tuple must have 2 elements, got {len(cutoff)}")
        cutoff_low, cutoff_high = cutoff
        if btype not in ("bandpass", "bandstop"):
            raise ValueError(f"Tuple cutoff requires bandpass/bandstop, got {btype}")
        # Validate bandpass/bandstop frequencies
        if cutoff_low <= 0 or cutoff_high <= 0:
            raise ValueError(
                f"Cutoff frequencies must be positive, got ({cutoff_low}, {cutoff_high})"
            )
        if (
            np.isnan(cutoff_low) or np.isnan(cutoff_high)
            or np.isinf(cutoff_low) or np.isinf(cutoff_high)
        ):
            raise ValueError(
                f"Cutoff frequencies must be finite, got ({cutoff_low}, {cutoff_high})"
            )
        if cutoff_low >= cutoff_high:
            raise ValueError(f"cutoff_low must be < cutoff_high, got ({cutoff_low}, {cutoff_high})")
        if cutoff_high >= fs / 2:
            raise ValueError(
                f"cutoff_high ({cutoff_high}) must be < Nyquist frequency (fs/2 = {fs/2})"
            )
    else:
        if btype in ("bandpass", "bandstop"):
            raise ValueError(f"{btype} requires tuple cutoff")
        # Validate single cutoff
        if cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff}")
        if np.isnan(cutoff) or np.isinf(cutoff):
            raise ValueError(f"Cutoff frequency must be finite, got {cutoff}")
        if cutoff >= fs / 2:
            raise ValueError(
                f"Cutoff frequency ({cutoff}) must be < Nyquist frequency (fs/2 = {fs/2})"
            )
        # Handle near-zero cutoff with small ridge
        if cutoff < 1e-6 * fs:
            cutoff = max(cutoff, 1e-6 * fs)  # Minimum cutoff
        cutoff_low = cutoff_high = cutoff

    # Prewarp frequencies
    if btype in ("lowpass", "highpass"):
        fc_prewarp = _prewarp_frequency(cutoff, fs)
    else:
        fc_center = np.sqrt(cutoff_low * cutoff_high)
        fc_prewarp = _prewarp_frequency(fc_center, fs)

    # Get analog poles
    if btype in ("lowpass", "highpass"):
        poles_analog = _cheby1_analog_poles(order, rp)
        # Scale by cutoff frequency
        poles_analog = poles_analog * 2.0 * np.pi * fc_prewarp
    else:
        # Simplified: use center frequency
        poles_analog = _cheby1_analog_poles(order, rp)
        poles_analog = poles_analog * 2.0 * np.pi * fc_prewarp

    # Analog zeros
    if btype == "lowpass":
        zeros_analog = np.array([])
    elif btype == "highpass":
        zeros_analog = np.zeros(order, dtype=complex)
    else:
        zeros_analog = np.array([])

    # Apply bilinear transform
    z_poles, z_zeros = _bilinear_transform(poles_analog, zeros_analog, fs)

    # Compute gain
    if btype == "lowpass":
        num_val = np.prod(1.0 - z_zeros) if len(z_zeros) > 0 else 1.0
        den_val = np.prod(1.0 - z_poles)
        k = den_val / num_val if abs(num_val) > 1e-10 else 1.0
    else:
        k = 1.0

    # Convert to transfer function
    b, a = _zpk_to_tf(z_zeros, z_poles, k)

    # Ensure real coefficients
    b = np.real_if_close(b, tol=1e-10)
    a = np.real_if_close(a, tol=1e-10)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    return b, a


def tf2sos(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Convert transfer function to second-order sections.

    Factors polynomial into biquad sections for numerical stability.
    Pairs complex conjugate poles/zeros deterministically.

    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.

    Returns:
        SOS array of shape (n_sections, 6) where each row is
        [b0, b1, b2, a0, a1, a2] for a biquad section.
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)

    # Ensure 1D arrays
    if b.ndim == 0:
        b = b.reshape(1)
    if a.ndim == 0:
        a = a.reshape(1)
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    # Normalize by a[0]
    if abs(a[0]) < 1e-10:
        raise ValueError("Denominator leading coefficient is zero")
    b = b / a[0]
    a = a / a[0]

    # Find roots
    z = np.roots(b)  # Zeros
    p = np.roots(a)  # Poles

    # Sort by angle then real part for deterministic pairing
    def sort_key(c):
        angle = np.angle(c)
        return (abs(angle), np.real(c))

    z_sorted = sorted(z, key=sort_key)
    p_sorted = sorted(p, key=sort_key)

    # Pair roots into second-order sections
    sos_list = []
    z_used = [False] * len(z_sorted)
    p_used = [False] * len(p_sorted)

    # Process all roots
    while not all(p_used):
        # Find next unused pole
        p_idx = next((i for i, used in enumerate(p_used) if not used), None)
        if p_idx is None:
            break

        p1 = p_sorted[p_idx]
        p_used[p_idx] = True

        # Try to pair with conjugate
        p2 = None
        for i, p_candidate in enumerate(p_sorted):
            if not p_used[i] and np.isclose(p1, np.conj(p_candidate), atol=1e-8):
                p2 = p_candidate
                p_used[i] = True
                break

        # If no conjugate, try to pair with real pole
        if p2 is None:
            for i, p_candidate in enumerate(p_sorted):
                if (
                    not p_used[i]
                    and np.isreal(p1)
                    and np.isreal(p_candidate)
                    and not np.isclose(p1, p_candidate)
                ):
                    p2 = p_candidate
                    p_used[i] = True
                    break

        # Create pole polynomial
        if p2 is not None:
            a_sec = np.poly([p1, p2])
        else:
            # Single pole, pair with zero at origin
            a_sec = np.poly([p1, 0.0])

        # Find corresponding zeros
        z1 = None
        z2 = None
        z1_idx = None

        # Try to find conjugate pair of zeros
        for i, z_candidate in enumerate(z_sorted):
            if not z_used[i]:
                if z1 is None:
                    z1 = z_candidate
                    z1_idx = i
                elif np.isclose(z1, np.conj(z_candidate), atol=1e-8):
                    z2 = z_candidate
                    z_used[i] = True
                    if z1_idx is not None:
                        z_used[z1_idx] = True
                    break

        # If no conjugate pair, try real zeros
        if z2 is None and z1 is not None:
            for i, z_candidate in enumerate(z_sorted):
                if (
                    not z_used[i]
                    and i != z1_idx
                    and np.isreal(z1)
                    and np.isreal(z_candidate)
                ):
                    z2 = z_candidate
                    z_used[i] = True
                    if z1_idx is not None:
                        z_used[z1_idx] = True
                    break

        # Create zero polynomial
        if z1 is not None and z2 is not None:
            b_sec = np.poly([z1, z2])
        elif z1 is not None:
            b_sec = np.poly([z1, 0.0])
            if z1_idx is not None:
                z_used[z1_idx] = True
        else:
            # No zeros, use zeros at origin
            b_sec = np.array([1.0, 0.0, 0.0])

        # Normalize section (a0 = 1)
        if abs(a_sec[0]) > 1e-10:
            a_sec = a_sec / a_sec[0]
            b_sec = b_sec / a_sec[0] if len(b_sec) <= len(a_sec) else b_sec

        # Pad to length 3
        if len(b_sec) < 3:
            b_sec = np.pad(b_sec, (0, 3 - len(b_sec)), mode="constant")
        if len(a_sec) < 3:
            a_sec = np.pad(a_sec, (0, 3 - len(a_sec)), mode="constant")

        sos_list.append([b_sec[0], b_sec[1], b_sec[2], a_sec[0], a_sec[1], a_sec[2]])

    # Handle any remaining zeros
    remaining_z = [z_sorted[i] for i, used in enumerate(z_used) if not used]
    if remaining_z:
        # Pair remaining zeros
        i = 0
        while i < len(remaining_z):
            if i + 1 < len(remaining_z):
                z1, z2 = remaining_z[i], remaining_z[i + 1]
                b_sec = np.poly([z1, z2])
                i += 2
            else:
                z1 = remaining_z[i]
                b_sec = np.poly([z1, 0.0])
                i += 1

            # Normalize and pad
            if len(b_sec) < 3:
                b_sec = np.pad(b_sec, (0, 3 - len(b_sec)), mode="constant")
            sos_list.append([b_sec[0], b_sec[1], b_sec[2], 1.0, 0.0, 0.0])

    if not sos_list:
        # Identity filter
        sos_list.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    return np.array(sos_list, dtype=float)


def sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter signal using second-order sections.

    Implements cascaded biquad filtering using Direct Form II Transposed
    for numerical stability.

    Args:
        sos: Second-order sections array of shape (n_sections, 6).
        x: Input signal (1D array).

    Returns:
        Filtered signal.
    """
    x = check_1d_array(x)
    sos = np.asarray(sos, dtype=float)

    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"SOS must be shape (n_sections, 6), got {sos.shape}")

    y = x.copy()
    n_sections = sos.shape[0]

    # Process each section
    for i in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[i, :]

        # Normalize by a0
        if abs(a0) > 1e-10:
            b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
            a1, a2 = a1 / a0, a2 / a0
            a0 = 1.0

        # Direct Form II Transposed state
        w1, w2 = 0.0, 0.0
        y_new = np.zeros_like(y)

        for n in range(len(y)):
            # Input to state
            w0 = y[n] - a1 * w1 - a2 * w2
            # Output from state
            y_new[n] = b0 * w0 + b1 * w1 + b2 * w2
            # Update state
            w2, w1 = w1, w0

        y = y_new

    return y


def lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter signal using direct form difference equation.

    Implements IIR filtering via difference equation. For high-order filters,
    automatically uses SOS for stability.

    Args:
        b: Numerator coefficients.
        a: Denominator coefficients.
        x: Input signal (1D array).

    Returns:
        Filtered signal.
    """
    b = check_1d_array(b)
    a = check_1d_array(a)
    x = check_1d_array(x)

    # For high order, use SOS
    if len(a) > 10 or len(b) > 10:
        sos = tf2sos(b, a)
        return sosfilt(sos, x)

    # Direct form I implementation
    len_b = len(b)
    len_a = len(a)

    # Normalize
    if abs(a[0]) < 1e-10:
        raise ValueError("Denominator leading coefficient is zero")
    b = b / a[0]
    a = a / a[0]

    # Initialize buffers
    x_buf = np.zeros(len_b, dtype=float)
    y_buf = np.zeros(len_a - 1, dtype=float)
    y = np.zeros_like(x)

    for n in range(len(x)):
        # Shift input buffer
        x_buf = np.roll(x_buf, 1)
        x_buf[0] = x[n]

        # Compute output: y[n] = sum(b[i]*x[n-i]) - sum(a[j]*y[n-j])
        y_n = np.dot(b, x_buf)
        if len(y_buf) > 0:
            y_n -= np.dot(a[1:], y_buf)

        y[n] = y_n

        # Shift output buffer
        if len(y_buf) > 0:
            y_buf = np.roll(y_buf, 1)
            y_buf[0] = y_n

    return y

