"""Exact time evolution for small systems using dense matrix exponentiation."""

from __future__ import annotations

import math
from typing import Optional

import torch

from qconduit.core.device import default_device
from qconduit.exact.diagonalize import paulisum_to_dense
from qconduit.operators import PauliSum


def exact_time_evolution_statevector(
    state: torch.Tensor,
    hamiltonian: PauliSum,
    time: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩ exactly via dense matrix exponentiation.

    This helper is intended for small systems in tests and examples. It
    constructs a dense matrix representation of H and uses standard linear
    algebra to apply exp(-i H t).

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ(0)⟩.
    hamiltonian:
        PauliSum representing the Hamiltonian H.
    time:
        Real evolution time t.
    device:
        Optional device for computation. Defaults to the state's device or
        `default_device()` if the state's device is not set.

    Returns
    -------
    torch.Tensor
        1D complex tensor of shape (2**n,) representing |ψ(t)⟩.

    Raises
    ------
    ValueError:
        If state is not 1D, has zero length, or has length that is not a power of 2.
    """
    # Validate state shape
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")
    if state.numel() == 0:
        raise ValueError("state must have nonzero length")

    dim = state.shape[0]

    # Check if dimension is a power of 2
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"Statevector length {dim} must be a power of 2.")

    # Determine device
    if device is None:
        if state.device.type == "meta":
            device = default_device().as_torch_device()
        else:
            device = state.device
    else:
        device = device

    # Move state to device and cast to complex128 for accuracy
    state = state.to(dtype=torch.complex128, device=device)

    # Get dense Hamiltonian matrix
    h_matrix = paulisum_to_dense(hamiltonian, num_qubits=n_qubits, device=device, dtype=torch.complex128)

    # Compute exp(-i H t) via eigendecomposition
    # H is Hermitian, so we use eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(h_matrix)

    # U = V diag(exp(-i * e_j * t)) V^†
    phases = torch.exp(-1.0j * eigenvalues * time)
    u_matrix = (eigenvectors * phases.unsqueeze(0)) @ eigenvectors.conj().T

    # Apply unitary to state
    state_t = u_matrix @ state.unsqueeze(1)
    state_t = state_t.squeeze(1)

    return state_t


__all__ = [
    "exact_time_evolution_statevector",
]

