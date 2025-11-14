"""Density-matrix backend for mixed quantum states.

This module implements standard density-matrix operations for small quantum systems.
The implementation uses straightforward textbook density-matrix/Kraus operator formalism.
Due to the O(4^n) memory cost of storing full density matrices, this backend is
intended for small qubit counts (e.g., 1-4 qubits).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..core.device import Device, default_device, device as device_factory
from .statevector import zero_state


def zero_dm_state(
    n_qubits: int,
    batch_shape: Optional[Tuple[int, ...]] = None,
    device: Device | torch.device | str | None = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Return the density matrix for |0...0><0...0| on n_qubits.

    The result has shape (*batch_shape, dim, dim) with dim = 2**n_qubits,
    and complex dtype. Intended for small n (e.g. 1-4).

    Args:
        n_qubits: Number of qubits. Must be >= 1.
        batch_shape: Optional batch dimensions. If None, no batch dimension.
        device: Device specification. Can be Device, str, torch.device, or None.
        dtype: Complex dtype. Defaults to torch.complex64.

    Returns:
        A complex tensor of shape (*batch_shape, dim, dim) representing |0...0><0...0|.

    Raises:
        ValueError: If n_qubits < 1.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    # Resolve device
    if device is None:
        qdevice = default_device()
    elif isinstance(device, Device):
        qdevice = device
    elif isinstance(device, str):
        qdevice = device_factory(device)
    elif isinstance(device, torch.device):
        if device.type == "cpu":
            qdevice = device_factory("sv_cpu")
        elif device.type == "cuda":
            qdevice = device_factory("sv_cuda")
        else:
            raise ValueError(
                f"Unsupported torch.device type: {device.type}. "
                "Only 'cpu' and 'cuda' are supported."
            )
    else:
        raise TypeError(
            f"device must be Device, str, torch.device, or None, got {type(device)}"
        )

    if dtype is None:
        dtype = torch.complex64

    if batch_shape is None:
        batch_shape = ()

    # Construct |0...0> using zero_state
    state = zero_state(n_qubits, batch_shape=batch_shape, device=qdevice, dtype=dtype)

    # Compute rho = |psi><psi| via outer product
    # state: (*batch_shape, dim)
    # rho: (*batch_shape, dim, dim)
    # Use broadcasting: state.unsqueeze(-1) * state.conj().unsqueeze(-2)
    rho = state.unsqueeze(-1) * state.conj().unsqueeze(-2)

    return rho


def dm_from_statevector(state: torch.Tensor) -> torch.Tensor:
    """
    Convert a pure statevector |psi> into a density matrix |psi><psi|.

    Input shape: (..., dim) complex.
    Output shape: (..., dim, dim) complex.

    Args:
        state: Complex statevector tensor of shape (..., 2**n_qubits).

    Returns:
        A complex tensor of shape (..., dim, dim) representing the density matrix.

    Raises:
        ValueError: If state is not complex dtype.
    """
    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    # Build outer product: |psi><psi|
    rho = state.unsqueeze(-1) * state.conj().unsqueeze(-2)

    return rho


def _kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Kronecker product of two matrices.

    Args:
        a: Matrix of shape (m, n).
        b: Matrix of shape (p, q).

    Returns:
        Matrix of shape (m*p, n*q).
    """
    # Use einsum for Kronecker product: ab,cd->acbd then reshape
    result = torch.einsum("ab,cd->acbd", a, b)
    return result.reshape(a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])


def apply_kraus_single_qubit(
    rho: torch.Tensor,
    kraus_ops: Tuple[torch.Tensor, ...],
    qubit: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply a single-qubit quantum channel, given by Kraus operators
    (each 2x2), to `qubit` of the n_qubits-qubit density matrix `rho`.

    The channel is applied as: rho -> sum_k F_k rho F_k^dagger, where
    F_k are the full-system Kraus operators obtained by tensoring the
    single-qubit Kraus operators with identity on other qubits.

    This is standard textbook Kraus operator application; nothing novel.

    Args:
        rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
        kraus_ops: Tuple of 2x2 complex tensors representing single-qubit Kraus operators.
        qubit: Index of the qubit to apply the channel to (0-indexed, 0 = LSB).
        n_qubits: Number of qubits. If None, inferred from rho.shape[-1].

    Returns:
        A new density matrix tensor of the same shape as rho.

    Raises:
        ValueError: If rho is not square, dimensions don't match, qubit index is invalid,
            or Kraus operators have incorrect shape.
    """
    # Validate rho
    if rho.dim() < 2:
        raise ValueError(f"rho must have at least 2 dimensions, got {rho.dim()}")
    if rho.shape[-1] != rho.shape[-2]:
        raise ValueError(
            f"rho must be square in last two dimensions, got shape {rho.shape}"
        )

    dim = rho.shape[-1]

    # Infer n_qubits from dim
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"rho dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(
            f"qubit index {qubit} out of range [0, {n_qubits})"
        )

    # Validate Kraus operators
    if len(kraus_ops) == 0:
        raise ValueError("kraus_ops must contain at least one operator")
    for i, E in enumerate(kraus_ops):
        if E.shape != (2, 2):
            raise ValueError(
                f"Kraus operator {i} must have shape (2, 2), got {E.shape}"
            )
        if not torch.is_complex(E):
            raise ValueError(
                f"Kraus operator {i} must be complex dtype, got {E.dtype}"
            )

    # Get dtype and device from rho
    dtype = rho.dtype
    device = rho.device

    # Build full-system Kraus operators F_k via Kronecker products
    # For qubit q, we use E if q == qubit, else I
    # Convention: qubit 0 is LSB, so we build from qubit n_qubits-1 down to 0
    # This matches the convention used in PauliSum.to_matrix
    I_single = torch.eye(2, dtype=dtype, device=device)

    full_kraus_ops = []
    for E in kraus_ops:
        # Ensure E has correct dtype and device
        E = E.to(dtype=dtype, device=device)

        # Build tensor product: P_{n-1} ⊗ P_{n-2} ⊗ ... ⊗ P_0
        # where P_i = E if i == qubit, else I
        F = E if (n_qubits - 1) == qubit else I_single
        for q in range(n_qubits - 2, -1, -1):
            if q == qubit:
                op = E
            else:
                op = I_single
            F = _kron(F, op)

        full_kraus_ops.append(F)

    # Apply channel: rho -> sum_k F_k rho F_k^dagger
    new_rho = torch.zeros_like(rho)
    for F in full_kraus_ops:
        # Broadcast over batch dims: (..., dim, dim) @ (dim, dim) -> (..., dim, dim)
        Frho = torch.matmul(F, rho)
        FrhoFdag = torch.matmul(Frho, F.conj().transpose(-2, -1))
        new_rho = new_rho + FrhoFdag

    return new_rho


def measure_probs_dm(rho: torch.Tensor) -> torch.Tensor:
    """
    Return probabilities of computational basis outcomes from a density matrix.

    The probability of basis state |i> is given by the diagonal element rho[i, i].

    Args:
        rho: Density matrix tensor of shape (..., dim, dim), Hermitian with trace ~1.

    Returns:
        A real tensor of shape (..., dim) containing probabilities (non-negative
        entries summing to ~1).

    Raises:
        ValueError: If rho is not square in last two dimensions.
    """
    if rho.shape[-1] != rho.shape[-2]:
        raise ValueError(
            f"rho must be square in last two dimensions, got shape {rho.shape}"
        )

    # Extract diagonal: diag = rho[i, i] for all i
    diag = rho.diagonal(dim1=-2, dim2=-1).real

    # Normalize to ensure probabilities sum to 1
    diag_sum = diag.sum(dim=-1, keepdim=True)
    # Avoid division by zero
    probs = diag / torch.clamp(diag_sum, min=1e-12)

    return probs


def measure_expectation_z_dm(
    rho: torch.Tensor,
    qubit: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute <Z_qubit> from a density matrix.

    The expectation value is computed as Tr(rho Z_qubit), which reduces to
    a weighted sum over computational basis states with signs based on the
    qubit's bit value.

    Args:
        rho: Density matrix tensor of shape (..., dim, dim).
        qubit: Index of the qubit to measure (0-indexed, 0 = LSB).
        n_qubits: Number of qubits. If None, inferred from rho.shape[-1].

    Returns:
        A real tensor with shape matching the batch dimensions of rho
        (without the dim, dim dimensions), containing the expectation value.

    Raises:
        ValueError: If rho is not square, dimensions don't match, or qubit index is invalid.
    """
    if rho.shape[-1] != rho.shape[-2]:
        raise ValueError(
            f"rho must be square in last two dimensions, got shape {rho.shape}"
        )

    dim = rho.shape[-1]

    # Infer n_qubits from dim
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"rho dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(
            f"qubit index {qubit} out of range [0, {n_qubits})"
        )

    # Get probabilities
    probs = measure_probs_dm(rho)

    # Build sign pattern for Z on that qubit
    # For basis index i, extract bit at position qubit
    # sign = +1 if bit == 0, -1 if bit == 1
    batch_shape = probs.shape[:-1]
    signs = torch.ones(dim, dtype=torch.float32, device=probs.device)

    for i in range(dim):
        bit = (i >> qubit) & 1
        signs[i] = 1.0 if bit == 0 else -1.0

    # Expand signs to match batch dimensions
    for _ in range(len(batch_shape)):
        signs = signs.unsqueeze(0)

    # Compute expectation: sum over probabilities * signs
    expectation = (probs * signs).sum(dim=-1)

    return expectation


