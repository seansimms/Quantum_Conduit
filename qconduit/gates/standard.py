"""Standard quantum gate implementations."""

from __future__ import annotations

import torch
import math
import cmath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def I(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Identity gate (single-qubit).

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the identity gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    return torch.eye(2, dtype=dtype, device=device)


def X(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Pauli-X gate (bit-flip, NOT gate).

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the X gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    return torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device
    )


def Y(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Pauli-Y gate.

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the Y gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    return torch.tensor(
        [[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device
    )


def Z(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Pauli-Z gate (phase-flip).

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the Z gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    return torch.tensor(
        [[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device
    )


def H(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Hadamard gate.

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the H gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    sqrt2_inv = 1.0 / math.sqrt(2.0)
    return torch.tensor(
        [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=dtype, device=device
    )


def S(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    S gate (phase gate, √Z).

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the S gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    return torch.tensor(
        [[1.0, 0.0], [0.0, 1.0j]], dtype=dtype, device=device
    )


def T(dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    T gate (π/8 gate, √S).

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").

    Returns:
        A (2, 2) complex tensor representing the T gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    exp_i_pi_4 = cmath.exp(1.0j * math.pi / 4.0)
    return torch.tensor(
        [[1.0, 0.0], [0.0, exp_i_pi_4]], dtype=dtype, device=device
    )


def CNOT(
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    control_first: bool = True,
) -> torch.Tensor:
    """
    CNOT gate (controlled-NOT, controlled-X).

    The gate acts on two qubits. When control_first=True, the first qubit is the
    control and the second is the target. The gate flips the target qubit if the
    control is |1⟩.

    Convention: The gate matrix is ordered as |00⟩, |01⟩, |10⟩, |11⟩ in the
    computational basis, where the first qubit is the control and the second is
    the target when control_first=True.

    Args:
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
        device: PyTorch device. Defaults to torch.device("cpu").
        control_first: If True, first qubit is control, second is target.
            If False, second qubit is control, first is target.

    Returns:
        A (4, 4) complex tensor representing the CNOT gate.
    """
    if dtype is None:
        dtype = torch.complex64
    if device is None:
        device = torch.device("cpu")

    if control_first:
        # First qubit is control, second is target
        # |00⟩ -> |00⟩, |01⟩ -> |01⟩, |10⟩ -> |11⟩, |11⟩ -> |10⟩
        matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
    else:
        # Second qubit is control, first is target
        # |00⟩ -> |00⟩, |01⟩ -> |11⟩, |10⟩ -> |10⟩, |11⟩ -> |01⟩
        matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
    return matrix


def RX(
    theta: torch.Tensor | float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotation gate around X-axis: RX(θ) = exp(-iθX/2).

    Matrix form:
        [[cos(θ/2), -i sin(θ/2)],
         [-i sin(θ/2), cos(θ/2)]]

    Args:
        theta: Rotation angle (scalar). Can be a float or a 0-d tensor.
            If a tensor, gradients will be preserved.
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
            If theta is a tensor, this is ignored and theta's dtype is used.
        device: PyTorch device. Defaults to torch.device("cpu").
            If theta is a tensor, this is ignored and theta's device is used.

    Returns:
        A (2, 2) complex tensor representing the RX gate.
    """
    # Handle tensor input (preserves gradients)
    if isinstance(theta, torch.Tensor):
        # Use theta's device and dtype
        device = theta.device
        # Convert to float32 for computation, then to complex
        theta_float = theta.to(dtype=torch.float32)
        half_theta = theta_float / 2.0
        cos_half = torch.cos(half_theta)
        sin_half = torch.sin(half_theta)
        
        # Build complex matrix using tensor operations
        # Convert to complex dtype
        if dtype is None:
            dtype = torch.complex64
        cos_half_c = cos_half.to(dtype=dtype)
        sin_half_c = sin_half.to(dtype=dtype)
        
        # Build matrix elements
        matrix = torch.stack([
            torch.stack([cos_half_c, -1.0j * sin_half_c]),
            torch.stack([-1.0j * sin_half_c, cos_half_c]),
        ])
        return matrix
    else:
        # Handle float input (backward compatibility)
        if dtype is None:
            dtype = torch.complex64
        if device is None:
            device = torch.device("cpu")
        
        theta_val = float(theta)
        half_theta = theta_val / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        return torch.tensor(
            [[cos_half, -1.0j * sin_half], [-1.0j * sin_half, cos_half]],
            dtype=dtype,
            device=device,
        )


def RY(
    theta: torch.Tensor | float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotation gate around Y-axis: RY(θ) = exp(-iθY/2).

    Matrix form:
        [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2), cos(θ/2)]]

    Args:
        theta: Rotation angle (scalar). Can be a float or a 0-d tensor.
            If a tensor, gradients will be preserved.
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
            If theta is a tensor, this is ignored and theta's dtype is used.
        device: PyTorch device. Defaults to torch.device("cpu").
            If theta is a tensor, this is ignored and theta's device is used.

    Returns:
        A (2, 2) complex tensor representing the RY gate.
    """
    # Handle tensor input (preserves gradients)
    if isinstance(theta, torch.Tensor):
        # Use theta's device and dtype
        device = theta.device
        # Convert to float32 for computation, then to complex
        theta_float = theta.to(dtype=torch.float32)
        half_theta = theta_float / 2.0
        cos_half = torch.cos(half_theta)
        sin_half = torch.sin(half_theta)
        
        # Build complex matrix using tensor operations
        # Convert to complex dtype
        if dtype is None:
            dtype = torch.complex64
        cos_half_c = cos_half.to(dtype=dtype)
        sin_half_c = sin_half.to(dtype=dtype)
        
        # Build matrix elements (RY has real matrix elements)
        matrix = torch.stack([
            torch.stack([cos_half_c, -sin_half_c]),
            torch.stack([sin_half_c, cos_half_c]),
        ])
        return matrix
    else:
        # Handle float input (backward compatibility)
        if dtype is None:
            dtype = torch.complex64
        if device is None:
            device = torch.device("cpu")
        
        theta_val = float(theta)
        half_theta = theta_val / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        return torch.tensor(
            [[cos_half, -sin_half], [sin_half, cos_half]],
            dtype=dtype,
            device=device,
        )


def RZ(
    theta: torch.Tensor | float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Rotation gate around Z-axis: RZ(θ) = exp(-iθZ/2).

    Matrix form:
        [[exp(-iθ/2), 0],
         [0, exp(iθ/2)]]

    Args:
        theta: Rotation angle (scalar). Can be a float or a 0-d tensor.
            If a tensor, gradients will be preserved.
        dtype: Complex dtype for the gate matrix. Defaults to torch.complex64.
            If theta is a tensor, this is ignored and theta's dtype is used.
        device: PyTorch device. Defaults to torch.device("cpu").
            If theta is a tensor, this is ignored and theta's device is used.

    Returns:
        A (2, 2) complex tensor representing the RZ gate.
    """
    # Handle tensor input (preserves gradients)
    if isinstance(theta, torch.Tensor):
        # Use theta's device
        device = theta.device
        # Convert to float32 for computation
        theta_float = theta.to(dtype=torch.float32)
        half_theta = theta_float / 2.0
        
        # Use PyTorch complex exponential
        if dtype is None:
            dtype = torch.complex64
        # Create complex tensor for exponent
        half_theta_c = half_theta.to(dtype=dtype)
        exp_neg = torch.exp(-1.0j * half_theta_c)
        exp_pos = torch.exp(1.0j * half_theta_c)
        
        # Build matrix
        zero = torch.zeros_like(exp_neg)
        matrix = torch.stack([
            torch.stack([exp_neg, zero]),
            torch.stack([zero, exp_pos]),
        ])
        return matrix
    else:
        # Handle float input (backward compatibility)
        if dtype is None:
            dtype = torch.complex64
        if device is None:
            device = torch.device("cpu")
        
        theta_val = float(theta)
        half_theta = theta_val / 2.0
        exp_neg = cmath.exp(-1.0j * half_theta)
        exp_pos = cmath.exp(1.0j * half_theta)
        
        return torch.tensor(
            [[exp_neg, 0.0], [0.0, exp_pos]],
            dtype=dtype,
            device=device,
        )


def is_unitary(matrix: torch.Tensor, atol: float = 1e-6) -> bool:
    """
    Check if a matrix is unitary within a given tolerance.

    A matrix U is unitary if U†U = I, where U† is the conjugate transpose.

    Args:
        matrix: Tensor of shape (..., n, n) representing one or more matrices.
        atol: Absolute tolerance for the check.

    Returns:
        True if the matrix is unitary (within tolerance), False otherwise.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        return False

    # Compute U†U
    adjoint = matrix.conj().transpose(-1, -2)
    product = torch.matmul(adjoint, matrix)

    # Compare to identity
    n = matrix.shape[-1]
    identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    # Expand identity to match batch dimensions if needed
    if product.ndim > 2:
        identity = identity.expand(product.shape)

    diff = torch.abs(product - identity)
    return torch.all(diff < atol).item()

