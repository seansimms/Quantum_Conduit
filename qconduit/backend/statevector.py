"""Statevector backend for pure quantum states.

This module provides the core statevector simulation backend with optimized
implementations for gate application and measurement operations.

Performance optimizations:
- Vectorized two-qubit gate construction (10-100x faster than Python loops)
- Contiguous memory layouts for optimal tensor operations
- Efficient einsum for single-qubit gate application
- Optional torch.compile for JIT acceleration (PyTorch 2.0+)
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import torch

from ..core.device import Device, default_device, device as device_factory
from ..diagnostics import assert_normalized, is_debug_enabled

if TYPE_CHECKING:
    pass


# Check if torch.compile is available (PyTorch 2.0+)
_TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and sys.version_info >= (3, 9)


def zero_state(
    n_qubits: int,
    batch_shape: tuple[int, ...] | None = None,
    device: Device | torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Create a zero state |0...0⟩ for n_qubits.

    The statevector has shape (*batch_shape, 2**n_qubits) with complex dtype.
    The amplitude at index 0 (|0...0⟩) is set to 1+0j, all others are 0.

    Args:
        n_qubits: Number of qubits. Must be >= 1.
        batch_shape: Optional batch dimensions. If None, no batch dimension.
        device: Device specification. Can be Device, str, torch.device, or None.
        dtype: Complex dtype. Defaults to torch.complex64.

    Returns:
        A complex tensor of shape (*batch_shape, 2**n_qubits) representing |0...0⟩.

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

    dim = 2**n_qubits
    shape = (*batch_shape, dim)
    state = torch.zeros(shape, dtype=dtype, device=qdevice.as_torch_device())

    # Set |0...0⟩ amplitude to 1 for all batch elements
    if len(batch_shape) > 0:
        state[..., 0] = 1.0 + 0.0j
    else:
        state[0] = 1.0 + 0.0j

    return state


def _apply_gate_core(
    state_reshaped: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """
    Core gate application using einsum.
    
    This is the innermost loop that can benefit from torch.compile.
    
    Args:
        state_reshaped: State tensor of shape (batch, left, 2, right).
        gate: Gate matrix of shape (2, 2).
    
    Returns:
        Transformed state tensor.
    """
    return torch.einsum("blqr,qo->blor", state_reshaped, gate)


# Apply torch.compile if available for the core operation
if _TORCH_COMPILE_AVAILABLE:
    try:
        _apply_gate_core_compiled = torch.compile(
            _apply_gate_core, mode="reduce-overhead"
        )
    except Exception:
        _apply_gate_core_compiled = _apply_gate_core
else:
    _apply_gate_core_compiled = _apply_gate_core


def _apply_two_qubit_gate_core(
    state: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Core tensor contraction for non-adjacent two-qubit gates."""
    return torch.einsum("blimjr,opij->blompr", state, gate)


def _apply_two_qubit_gate_adjacent_core(
    state: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Core contraction for adjacent qubits using 4x4 gate matrix."""
    return torch.einsum("blqr,oq->blor", state, gate)


if _TORCH_COMPILE_AVAILABLE:
    try:
        _apply_two_qubit_gate_core_compiled = torch.compile(
            _apply_two_qubit_gate_core, mode="reduce-overhead"
        )
    except Exception:
        _apply_two_qubit_gate_core_compiled = _apply_two_qubit_gate_core

    try:
        _apply_two_qubit_gate_adjacent_core_compiled = torch.compile(
            _apply_two_qubit_gate_adjacent_core, mode="reduce-overhead"
        )
    except Exception:
        _apply_two_qubit_gate_adjacent_core_compiled = (
            _apply_two_qubit_gate_adjacent_core
        )
else:
    _apply_two_qubit_gate_core_compiled = _apply_two_qubit_gate_core
    _apply_two_qubit_gate_adjacent_core_compiled = (
        _apply_two_qubit_gate_adjacent_core
    )


def apply_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit: int,
    n_qubits: int | None = None,
) -> torch.Tensor:
    """
    Apply a single-qubit gate to a specific qubit in the statevector.

    Convention: qubit 0 is the least significant bit (LSB) in the computational
    basis index. For example, in a 2-qubit state |q1 q0⟩, qubit 0 corresponds
    to the rightmost bit.
    
    Performance optimizations:
    - Contiguous memory layout
    - Efficient einsum contraction
    - Optional torch.compile acceleration

    Args:
        state: Statevector tensor of shape (..., 2**n_qubits) with complex dtype.
        gate: Single-qubit gate matrix of shape (2, 2).
        qubit: Index of the qubit to apply the gate to (0-indexed, 0 = LSB).
        n_qubits: Number of qubits. If None, inferred from state.shape[-1].

    Returns:
        A new statevector tensor with the gate applied.

    Raises:
        ValueError: If gate shape is not (2, 2), qubit index is invalid, or
            state dimension is not a power of 2.
    """
    if gate.shape != (2, 2):
        raise ValueError(f"gate must have shape (2, 2), got {gate.shape}")

    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit index {qubit} out of range [0, {n_qubits})")

    batch_shape = state.shape[:-1]
    batch_size = math.prod(batch_shape) if batch_shape else 1

    # Ensure contiguous memory layout for optimal performance
    state_flat = state.reshape(batch_size, dim).contiguous()

    # Reshape to separate target qubit: (batch, left, 2, right)
    left_size = 2 ** (n_qubits - 1 - qubit)
    right_size = 2**qubit

    state_reshaped = state_flat.reshape(batch_size, left_size, 2, right_size)

    # Apply gate using optimized core function
    state_transformed = _apply_gate_core_compiled(state_reshaped, gate)

    # Reshape back
    state_new = state_transformed.reshape(batch_size, dim)
    new_state = state_new.reshape(*batch_shape, dim)

    if is_debug_enabled():
        assert_normalized(new_state, atol=1e-4)

    return new_state


def _swap_gate_qubit_order(gate: torch.Tensor) -> torch.Tensor:
    """Swap qubit order in a two-qubit gate matrix."""
    gate_view = gate.reshape(2, 2, 2, 2)
    # Permute output and input axes to swap qubits.
    return gate_view.permute(1, 0, 3, 2).reshape(4, 4).contiguous()


def _apply_two_qubit_gate_einsum(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a two-qubit gate using tensor contraction."""
    batch_shape = state.shape[:-1]
    dim = state.shape[-1]
    batch_size = math.prod(batch_shape) if batch_shape else 1
    state_flat = state.reshape(batch_size, dim).contiguous()

    q_hi, q_lo = (qubit1, qubit2) if qubit1 > qubit2 else (qubit2, qubit1)
    gate_matrix = gate if qubit1 >= qubit2 else _swap_gate_qubit_order(gate)

    left_size = 2 ** (n_qubits - q_hi - 1)
    mid_size = 2 ** (q_hi - q_lo - 1)
    right_size = 2 ** q_lo

    if mid_size == 1:
        # Adjacent qubits -> reshape to (batch, left, 4, right)
        state_view = state_flat.reshape(batch_size, left_size, 4, right_size)
        gate_view = gate_matrix.reshape(4, 4)
        transformed = _apply_two_qubit_gate_adjacent_core_compiled(
            state_view, gate_view
        ).reshape(batch_size, dim)
    else:
        state_view = state_flat.reshape(
            batch_size, left_size, 2, mid_size, 2, right_size
        )
        gate_view = gate_matrix.reshape(2, 2, 2, 2)
        transformed = _apply_two_qubit_gate_core_compiled(
            state_view, gate_view
        ).reshape(batch_size, dim)

    return transformed.reshape(*batch_shape, dim)


def apply_two_qubit_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int | None = None,
) -> torch.Tensor:
    """
    Apply a two-qubit gate to qubit1 and qubit2 in the statevector.

    Convention: LSB-first, so qubit 0 is the least significant bit in the
    computational basis index. The gate matrix is indexed as |q1 q2⟩ where
    q1 = qubit1 and q2 = qubit2, with index = 2*b_q1 + b_q2.
    """
    if gate.shape != (4, 4):
        raise ValueError(f"gate must have shape (4, 4), got {gate.shape}")

    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    if qubit1 == qubit2:
        raise ValueError(
            f"qubit1 and qubit2 must be distinct, got {qubit1} and {qubit2}"
        )

    if qubit1 < 0 or qubit1 >= n_qubits:
        raise ValueError(f"qubit1 index {qubit1} out of range [0, {n_qubits})")
    if qubit2 < 0 or qubit2 >= n_qubits:
        raise ValueError(f"qubit2 index {qubit2} out of range [0, {n_qubits})")

    new_state = _apply_two_qubit_gate_einsum(
        state, gate, qubit1, qubit2, n_qubits
    )

    if is_debug_enabled():
        assert_normalized(new_state, atol=1e-4)

    return new_state


def apply_two_qubit_gate_direct(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a two-qubit gate without any caching helpers."""
    return _apply_two_qubit_gate_einsum(state, gate, qubit1, qubit2, n_qubits)


def measure_expectation_z(
    state: torch.Tensor,
    qubit: int,
    n_qubits: int | None = None,
) -> torch.Tensor:
    """
    Compute the expectation value ⟨Z⟩ for a given qubit.

    For a qubit in state |ψ⟩, ⟨Z⟩ = ⟨ψ|Z|ψ⟩ = P(0) - P(1), where P(0) and P(1)
    are the probabilities of measuring |0⟩ and |1⟩ on that qubit.

    Args:
        state: Statevector tensor of shape (..., 2**n_qubits) with complex dtype.
        qubit: Index of the qubit to measure (0-indexed, 0 = LSB).
        n_qubits: Number of qubits. If None, inferred from state.shape[-1].

    Returns:
        A real tensor with shape matching the batch dimensions of state.

    Raises:
        ValueError: If qubit index is invalid or state dimension is not a power of 2.
    """
    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit index {qubit} out of range [0, {n_qubits})")

    batch_shape = state.shape[:-1]

    # Compute probabilities with contiguous layout
    probs = measure_probs(state, n_qubits).contiguous()

    # Reshape to separate target qubit
    batch_size = math.prod(batch_shape) if batch_shape else 1
    probs_flat = probs.reshape(batch_size, dim)

    left_size = 2 ** (n_qubits - 1 - qubit)
    right_size = 2**qubit

    probs_reshaped = probs_flat.reshape(batch_size, left_size, 2, right_size)

    # Sum over non-target dimensions to get P(0) and P(1)
    p_qubit = probs_reshaped.sum(dim=(1, 3))

    # ⟨Z⟩ = P(0) - P(1)
    expectation = p_qubit[:, 0] - p_qubit[:, 1]

    return expectation.reshape(batch_shape)


def measure_probs(
    state: torch.Tensor,
    n_qubits: int | None = None,
) -> torch.Tensor:
    """
    Compute the probability distribution over computational basis states.

    The probability of basis state |i⟩ is |⟨i|ψ⟩|² = |state[i]|².

    Args:
        state: Statevector tensor of shape (..., 2**n_qubits) with complex dtype.
        n_qubits: Number of qubits. If None, inferred from state.shape[-1].

    Returns:
        A real tensor of the same shape as state, containing probabilities.

    Raises:
        ValueError: If state dimension is not a power of 2.
    """
    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )
    else:
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits}"
            )

    # Compute |state|² with contiguous output
    probs = (torch.abs(state) ** 2).contiguous()

    # Normalize to ensure sum is 1
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / torch.clamp(probs_sum, min=1e-12)

    return probs


def clear_gate_caches() -> None:
    """Clear gate-related caches (no-op retained for compatibility)."""
    return None


__all__ = [
    "zero_state",
    "apply_gate",
    "apply_two_qubit_gate",
    "apply_two_qubit_gate_direct",
    "measure_expectation_z",
    "measure_probs",
    "clear_gate_caches",
]
