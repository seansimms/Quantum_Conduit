"""Expectation evaluation for Pauli-sum Hamiltonians."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from ..backend.statevector import apply_gate, measure_expectation_z
from ..gates.standard import H, I, S
from .pauli import PauliSum, PauliTerm


def basis_change_gate(
    label: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Get the basis-change gate for measuring a Pauli operator via Z measurements.

    This implements the standard textbook technique for measuring arbitrary
    Pauli operators by rotating to the Z basis and then measuring Z.

    The mapping is:
    - Z: U = I (no change, already in Z basis)
    - X: U = H (Hadamard rotates X to Z: H Z H = X)
    - Y: U = S† H (rotates Y to Z: S† H Z H S = Y, up to global phase)
    - I: U = I (no change, identity term)

    This is a completely standard textbook mapping; there is nothing novel here.

    Args:
        label: Pauli label "I", "X", "Y", or "Z".
        dtype: Complex dtype for the gate matrix.
        device: PyTorch device.

    Returns:
        A (2, 2) complex tensor representing the basis-change gate.

    Raises:
        ValueError: If label is not in {"I", "X", "Y", "Z"}.
    """
    if label == "Z" or label == "I":
        return I(dtype=dtype, device=device)
    elif label == "X":
        return H(dtype=dtype, device=device)
    elif label == "Y":
        # S† H: conjugate transpose of S, then H
        s_gate = S(dtype=dtype, device=device)
        h_gate = H(dtype=dtype, device=device)
        # S† = S.conj().transpose(0, 1)
        s_dagger = s_gate.conj().transpose(0, 1)
        # S† @ H
        return s_dagger @ h_gate
    else:
        raise ValueError(
            f"Invalid Pauli label '{label}'. Must be one of {{'I', 'X', 'Y', 'Z'}}"
        )


def expectation_pauli_term(state: torch.Tensor, term: PauliTerm) -> torch.Tensor:
    """
    Compute the expectation value ⟨ψ|P|ψ⟩ for a single PauliTerm P.

    This uses the standard technique of rotating to the Z basis and then
    computing the expectation via bitstring sign patterns. This is textbook
    quantum computing; nothing proprietary or novel.

    Args:
        state: Complex statevector tensor of shape (..., 2**n_qubits).
        term: PauliTerm with n_qubits matching the state dimension.

    Returns:
        Real tensor with shape matching the batch dimensions of state
        (without the 2**n_qubits dimension), containing the expectation value.

    Raises:
        ValueError: If state dimension does not match term.n_qubits(), or if
            term is invalid.
    """
    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    n_qubits = term.n_qubits()

    # Validate dimension
    if 2**n_qubits != dim:
        raise ValueError(
            f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
            f"for term with {n_qubits} qubits"
        )

    # Handle identity term: ⟨ψ|I|ψ⟩ = 1, so result is just the coefficient
    if term.is_identity():
        # Return coeff with appropriate batch shape
        batch_shape = state.shape[:-1]
        if len(batch_shape) == 0:
            return torch.tensor(
                term.coeff, dtype=state.real.dtype, device=state.device
            )
        else:
            # Create tensor with batch shape filled with coeff
            result = torch.full(
                batch_shape,
                term.coeff,
                dtype=state.real.dtype,
                device=state.device,
            )
            return result

    # Make a working copy of the state (do not modify input)
    rot_state = state.clone()

    # Apply basis-change gates for each qubit
    for qubit_idx, pauli_label in enumerate(term.paulis):
        if pauli_label == "I":
            # Skip identity (no rotation needed)
            continue

        # Get basis-change gate
        gate = basis_change_gate(
            pauli_label, dtype=rot_state.dtype, device=rot_state.device
        )

        # Apply gate to the qubit
        rot_state = apply_gate(
            rot_state, gate, qubit=qubit_idx, n_qubits=n_qubits
        )

    # After rotation, we need to compute ⟨Z ⊗ Z ⊗ ... ⊗ Z⟩ for the qubits
    # where the term is non-I. The expectation of a product of Z operators
    # is: ⟨∏_{q in Q} Z_q⟩ = ∑_{bitstrings} (-1)^{# of 1s on those qubits} |ψ(bitstring)|²

    # Compute probabilities
    probs = torch.abs(rot_state) ** 2  # shape: (..., 2**n_qubits)

    # Create sign pattern: for each basis state |i⟩, determine the sign
    # based on the parity of 1s on the qubits where the term is non-I
    batch_shape = probs.shape[:-1]
    dim = probs.shape[-1]

    # Find qubits where term is non-I
    non_identity_qubits = [
        q for q, p in enumerate(term.paulis) if p != "I"
    ]

    if len(non_identity_qubits) == 0:
        # All I's (should have been caught above, but handle gracefully)
        batch_shape = state.shape[:-1]
        if len(batch_shape) == 0:
            return torch.tensor(
                term.coeff, dtype=state.real.dtype, device=state.device
            )
        else:
            return torch.full(
                batch_shape,
                term.coeff,
                dtype=state.real.dtype,
                device=state.device,
            )

    # Build sign pattern: for each basis index i, compute the parity
    # of 1s on the non-identity qubits
    # sign[i] = (-1)^(sum of bits at non_identity_qubits positions)
    signs = torch.ones(dim, dtype=torch.float32, device=probs.device)

    for basis_idx in range(dim):
        # Extract bits at non-identity qubit positions
        parity = 0
        for q in non_identity_qubits:
            # Extract bit q from basis_idx
            bit = (basis_idx >> q) & 1
            parity += bit
        # Sign is (-1)^parity
        signs[basis_idx] = 1.0 if (parity % 2 == 0) else -1.0

    # Expand signs to match batch dimensions
    # signs: (dim,)
    # probs: (..., dim)
    # We want to broadcast signs to (..., dim)
    for _ in range(len(batch_shape)):
        signs = signs.unsqueeze(0)

    # Compute expectation: sum over probabilities * signs
    expectation = (probs * signs).sum(dim=-1)  # shape: (...)

    # Multiply by coefficient
    result = term.coeff * expectation

    # Ensure result is real (should already be, but explicit for clarity)
    return result.real


def expectation_pauli_sum(state: torch.Tensor, hamiltonian: PauliSum) -> torch.Tensor:
    """
    Compute the expectation value ⟨ψ|H|ψ⟩ for a PauliSum Hamiltonian H.

    This evaluates ⟨ψ|H|ψ⟩ = ∑ᵢ cᵢ ⟨ψ|Pᵢ|ψ⟩ by summing expectations of
    individual PauliTerm objects. This is standard linear algebra; nothing novel.

    Args:
        state: Complex statevector tensor of shape (..., 2**n_qubits).
        hamiltonian: PauliSum with n_qubits matching the state dimension.

    Returns:
        Real tensor with shape matching the batch dimensions of state
        (without the 2**n_qubits dimension), containing the expectation value.

    Raises:
        ValueError: If state dimension does not match hamiltonian.n_qubits().
    """
    if not torch.is_complex(state):
        raise ValueError(f"state must be complex dtype, got {state.dtype}")

    dim = state.shape[-1]
    n_qubits = hamiltonian.n_qubits()

    # Validate dimension
    if n_qubits == 0:
        # Empty Hamiltonian: expectation is 0
        batch_shape = state.shape[:-1]
        if len(batch_shape) == 0:
            return torch.tensor(0.0, dtype=state.real.dtype, device=state.device)
        else:
            return torch.zeros(
                batch_shape, dtype=state.real.dtype, device=state.device
            )

    if 2**n_qubits != dim:
        raise ValueError(
            f"state dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
            f"for Hamiltonian with {n_qubits} qubits"
        )

    # Sum expectations over all terms
    total = None
    for term in hamiltonian.terms:
        exp_term = expectation_pauli_term(state, term)
        if total is None:
            total = exp_term
        else:
            total = total + exp_term

    # Handle empty Hamiltonian
    if total is None:
        batch_shape = state.shape[:-1]
        if len(batch_shape) == 0:
            return torch.tensor(0.0, dtype=state.real.dtype, device=state.device)
        else:
            return torch.zeros(
                batch_shape, dtype=state.real.dtype, device=state.device
            )

    return total


def expectation_pauli_sum_dm(
    rho: torch.Tensor,
    hamiltonian: PauliSum,
) -> torch.Tensor:
    """
    Compute <H> = Tr(rho H) for a PauliSum Hamiltonian and density matrix rho.

    This uses hamiltonian.to_matrix() internally and is intended for small n_qubits.
    This is standard linear algebra: Tr(ρH) = sum_{i,j} rho_{ij} H_{ji}.

    Args:
        rho: Density matrix tensor of shape (..., dim, dim) with complex dtype.
        hamiltonian: PauliSum with n_qubits matching the density matrix dimension.

    Returns:
        Real tensor with shape matching the batch dimensions of rho
        (without the dim, dim dimensions), containing the expectation value.

    Raises:
        ValueError: If rho is not square, dimensions don't match, or n_qubits is too large.
    """
    if rho.shape[-1] != rho.shape[-2]:
        raise ValueError(
            f"rho must be square in last two dimensions, got shape {rho.shape}"
        )

    dim = rho.shape[-1]
    n_qubits = hamiltonian.n_qubits()

    if n_qubits == 0:
        # Empty Hamiltonian: expectation is 0
        batch_shape = rho.shape[:-2]
        if len(batch_shape) == 0:
            return torch.tensor(0.0, dtype=rho.real.dtype, device=rho.device)
        else:
            return torch.zeros(
                batch_shape, dtype=rho.real.dtype, device=rho.device
            )

    if 2**n_qubits != dim:
        raise ValueError(
            f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
            f"for Hamiltonian with {n_qubits} qubits"
        )

    # Compute full Hamiltonian matrix
    H = hamiltonian.to_matrix(dtype=rho.dtype, device=rho.device)

    # Compute Tr(rho H) = sum_{i,j} rho_{ij} H_{ji}
    # For batched rho: (..., dim, dim) @ (dim, dim) -> (..., dim, dim)
    M = torch.matmul(rho, H)

    # Extract trace: sum of diagonal elements
    tr = M.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real

    return tr

