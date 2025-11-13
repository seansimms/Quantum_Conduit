"""Statevector backend for pure quantum states."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from ..core.device import Device, device as device_factory, default_device
from ..diagnostics import is_debug_enabled, assert_normalized

if TYPE_CHECKING:
    pass


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
    # Use ellipsis to set the last dimension index 0 for all batch dimensions
    if len(batch_shape) > 0:
        # For batched states, set state[..., 0] = 1
        state[..., 0] = 1.0 + 0.0j
    else:
        # For non-batched states, set state[0] = 1
        state[0] = 1.0 + 0.0j

    return state


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
        # Infer n_qubits from dimension
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
        raise ValueError(
            f"qubit index {qubit} out of range [0, {n_qubits})"
        )

    # Reshape state to separate the target qubit
    # For qubit i (0-indexed, 0 = LSB), we need to reshape as:
    # (*batch, 2**(n_qubits - 1 - i), 2, 2**i)
    # Then apply gate on the middle dimension of size 2
    batch_shape = state.shape[:-1]
    batch_size = math.prod(batch_shape) if batch_shape else 1

    # Reshape to (batch_size, 2**n_qubits)
    state_flat = state.reshape(batch_size, dim)

    # For qubit i, we want to group states where this qubit is 0 or 1
    # The index of a basis state |b_{n-1} ... b_0⟩ is sum(b_k * 2^k)
    # For qubit i, bit b_i determines whether we're in the first or second half
    # of a group of size 2**(i+1)

    # Reshape to bring qubit i to a separate dimension
    # Strategy: reshape to (batch, 2**(n-i-1), 2, 2**i)
    left_size = 2 ** (n_qubits - 1 - qubit)
    right_size = 2**qubit

    state_reshaped = state_flat.reshape(batch_size, left_size, 2, right_size)

    # Apply gate on the qubit dimension (dimension 2)
    # Use einsum: 'b l q r, q q2 -> b l q2 r'
    # Note: q and q2 are the gate input and output dimensions
    state_transformed = torch.einsum(
        "blqr,qo->blor", state_reshaped, gate
    )

    # Reshape back to (batch_size, 2**n_qubits)
    state_new = state_transformed.reshape(batch_size, dim)

    # Reshape back to original batch shape
    new_state = state_new.reshape(*batch_shape, dim)

    # In debug mode, verify that unitary gate preserves normalization
    if is_debug_enabled():
        assert_normalized(new_state, atol=1e-4)

    return new_state


def apply_two_qubit_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int | None = None,
) -> torch.Tensor:
    """
    Apply a two-qubit gate to qubit1 and qubit2 in the statevector.

    Convention: qubit 0 is the least significant bit (LSB). The gate matrix
    is ordered for |qubit1, qubit2⟩ basis states, where qubit1 and qubit2
    are the qubit indices passed to this function (not necessarily in sorted order).

    Args:
        state: Statevector tensor of shape (..., 2**n_qubits) with complex dtype.
        gate: Two-qubit gate matrix of shape (4, 4).
        qubit1: Index of the first qubit (0-indexed).
        qubit2: Index of the second qubit (0-indexed).
        n_qubits: Number of qubits. If None, inferred from state.shape[-1].

    Returns:
        A new statevector tensor with the gate applied.

    Raises:
        ValueError: If gate shape is not (4, 4), qubit indices are invalid or
            equal, or state dimension is not a power of 2.
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
        raise ValueError(f"qubit1 and qubit2 must be distinct, got {qubit1} and {qubit2}")

    if qubit1 < 0 or qubit1 >= n_qubits:
        raise ValueError(f"qubit1 index {qubit1} out of range [0, {n_qubits})")
    if qubit2 < 0 or qubit2 >= n_qubits:
        raise ValueError(f"qubit2 index {qubit2} out of range [0, {n_qubits})")

    # Order qubits so q_low < q_high for consistent reshaping
    q_low, q_high = min(qubit1, qubit2), max(qubit1, qubit2)
    swapped = qubit1 > qubit2

    batch_shape = state.shape[:-1]
    batch_size = math.prod(batch_shape) if batch_shape else 1

    state_flat = state.reshape(batch_size, dim)

    # Reshape to bring the two qubits together
    # We need to separate: left (before q_low), q_low, middle (between q_low and q_high),
    # q_high, right (after q_high)
    left_size = 2**q_low
    middle_size = 2 ** (q_high - q_low - 1)
    right_size = 2 ** (n_qubits - 1 - q_high)

    # Reshape to (batch, left, 2, middle, 2, right)
    # The statevector uses little-endian indexing: index = sum(bit_i * 2^i)
    # To correctly extract qubit0 (q_low) and qubit1 (q_high), we need to ensure
    # that when we combine them, the index represents |q_low, q_high⟩ correctly.
    # The initial reshape creates (batch, left, q_low, middle, q_high, right)
    # but this extracts q_low before q_high, making q_low the higher-order bit.
    # We need to swap them so q_high comes before q_low in the bit extraction order.
    state_reshaped = state_flat.reshape(
        batch_size, left_size, 2, middle_size, 2, right_size
    )

    # Combine the two qubit dimensions: (batch, left, middle, right, 2, 2)
    # Then reshape to (batch, left, middle, right, 4)
    # After permute, dimensions are: (batch, left, middle, right, q_high, q_low)
    # The combined index represents |q_high, q_low⟩ where index = q_high * 2 + q_low
    state_reshaped = state_reshaped.permute(0, 1, 3, 5, 4, 2).reshape(
        batch_size, left_size, middle_size, right_size, 4
    )
    
    # The reshape extracts bits in a way that doesn't match the expected |q_high, q_low⟩ ordering.
    # We need to correct the bit order for the swapped case, but not for the not-swapped case.
    # The correction swaps |01⟩ <-> |10⟩, which is the permutation [0, 2, 1, 3]
    # For swapped case: apply bit correction to get correct |q_high, q_low⟩ ordering
    # For not-swapped case: skip bit correction, reshape already gives correct |q_high, q_low⟩
    if swapped:
        bit_correction_perm = torch.tensor([0, 2, 1, 3], device=state_reshaped.device, dtype=torch.long)
        state_reshaped = state_reshaped[..., bit_correction_perm]
    
    # After bit correction (if applied), the combined index now correctly represents |q_high, q_low⟩
    # The gate matrix is defined for (qubit1, qubit2) ordering
    # If swapped (qubit1 > qubit2), the gate is for |qubit1, qubit2⟩ = |q_high, q_low⟩
    # which matches our reshape after correction, so no transformation needed
    # If not swapped (qubit1 < qubit2), the gate is for |qubit1, qubit2⟩ = |q_low, q_high⟩
    # so we need to swap the statevector indices from |q_high, q_low⟩ to |q_low, q_high⟩
    if not swapped:
        # Gate is defined for |qubit1, qubit2⟩ = |q_low, q_high⟩
        # But our reshape is for |q_high, q_low⟩
        # We need to swap the qubit ordering in the statevector indices
        swap_perm = torch.tensor([0, 2, 1, 3], device=state_reshaped.device, dtype=torch.long)
        state_reshaped = state_reshaped[..., swap_perm]

    # Apply gate: (batch, left, middle, right, 4) @ (4, 4) -> (batch, left, middle, right, 4)
    state_transformed = torch.einsum("blmrq,qo->blmro", state_reshaped, gate)
    
    # Swap back if we swapped the statevector
    if not swapped:
        swap_perm = torch.tensor([0, 2, 1, 3], device=state_transformed.device, dtype=torch.long)
        state_transformed = state_transformed[..., swap_perm]
    
    # Reverse the bit correction to restore the original bit order for reshaping back
    # Only apply if we applied it in the first place (swapped case)
    if swapped:
        bit_correction_perm = torch.tensor([0, 2, 1, 3], device=state_transformed.device, dtype=torch.long)
        state_transformed = state_transformed[..., bit_correction_perm]

    # Reshape back: (batch, left, middle, right, 4) -> (batch, left, middle, right, 2, 2)
    # The 4 states are in |q_high, q_low⟩ order (after reversing transformations)
    state_transformed = state_transformed.reshape(
        batch_size, left_size, middle_size, right_size, 2, 2
    )

    # Permute back: (batch, left, middle, right, q_high, q_low) -> (batch, left, q_low, middle, q_high, right)
    state_transformed = state_transformed.permute(0, 1, 5, 2, 4, 3)

    # Reshape back to (batch_size, 2**n_qubits)
    state_new = state_transformed.reshape(batch_size, dim)

    new_state = state_new.reshape(*batch_shape, dim)

    # In debug mode, verify that unitary gate preserves normalization
    if is_debug_enabled():
        assert_normalized(new_state, atol=1e-4)

    return new_state


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
        A real tensor with shape matching the batch dimensions of state (without
        the 2**n_qubits dimension), containing the expectation value ⟨Z⟩.

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
        raise ValueError(
            f"qubit index {qubit} out of range [0, {n_qubits})"
        )

    batch_shape = state.shape[:-1]

    # Compute probabilities for |0⟩ and |1⟩ on the target qubit
    # For qubit i, we sum probabilities over all basis states where bit i is 0 or 1
    probs = measure_probs(state, n_qubits)

    # Reshape probabilities to separate the target qubit
    # Similar to apply_gate, we reshape to (batch, 2**(n-i-1), 2, 2**i)
    batch_size = math.prod(batch_shape) if batch_shape else 1
    probs_flat = probs.reshape(batch_size, dim)

    left_size = 2 ** (n_qubits - 1 - qubit)
    right_size = 2**qubit

    probs_reshaped = probs_flat.reshape(batch_size, left_size, 2, right_size)

    # Sum over left and right dimensions to get P(0) and P(1) for the target qubit
    # Shape: (batch, 2)
    p_qubit = probs_reshaped.sum(dim=(1, 3))  # Sum over left_size and right_size

    # ⟨Z⟩ = P(0) - P(1)
    expectation = p_qubit[:, 0] - p_qubit[:, 1]

    # Reshape to match batch_shape
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
        Probabilities are normalized to sum to 1 (within numerical tolerance).

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

    # Compute |state|²
    probs = torch.abs(state) ** 2

    # Normalize to ensure sum is 1 (within numerical tolerance)
    # Sum over the last dimension
    probs_sum = probs.sum(dim=-1, keepdim=True)
    # Avoid division by zero
    probs = probs / torch.clamp(probs_sum, min=1e-12)

    return probs

