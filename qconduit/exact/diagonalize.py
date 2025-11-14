"""Exact diagonalization utilities for PauliSum Hamiltonians."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from qconduit.core.device import default_device
from qconduit.gates.standard import I, X, Y, Z
from qconduit.operators import PauliSum


def _single_site_matrix(
    label: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Get the 2×2 matrix representation of a single-qubit Pauli operator.

    Parameters
    ----------
    label:
        Pauli label: "I", "X", "Y", or "Z".
    dtype:
        Complex dtype for the matrix.
    device:
        PyTorch device.

    Returns
    -------
    torch.Tensor
        2×2 complex matrix representing the Pauli operator.

    Raises
    ------
    ValueError:
        If label is not in {"I", "X", "Y", "Z"}.
    """
    if label == "I":
        return I(dtype=dtype, device=device)
    elif label == "X":
        return X(dtype=dtype, device=device)
    elif label == "Y":
        return Y(dtype=dtype, device=device)
    elif label == "Z":
        return Z(dtype=dtype, device=device)
    else:
        raise ValueError(f"Invalid Pauli label: {label}. Must be one of {{I, X, Y, Z}}")


def paulisum_to_dense(
    hamiltonian: PauliSum,
    num_qubits: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Convert a PauliSum acting on `num_qubits` qubits into a dense Hermitian
    matrix of shape (2**num_qubits, 2**num_qubits).

    Parameters
    ----------
    hamiltonian:
        PauliSum describing the Hamiltonian.
    num_qubits:
        Number of qubits (must be >= 1).
    device:
        Optional torch device. Defaults to `default_device().as_torch_device()`.
    dtype:
        Complex dtype for the dense matrix (default: complex128).

    Returns
    -------
    torch.Tensor
        Dense Hermitian matrix representing the Hamiltonian.

    Raises
    ------
    ValueError:
        If num_qubits < 1 or if hamiltonian.n_qubits() != num_qubits.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

    if hamiltonian.n_qubits() != 0 and hamiltonian.n_qubits() != num_qubits:
        raise ValueError(
            f"hamiltonian.n_qubits() = {hamiltonian.n_qubits()} "
            f"does not match num_qubits = {num_qubits}"
        )

    if device is None:
        device = default_device().as_torch_device()

    dim = 1 << num_qubits  # 2**num_qubits
    H = torch.zeros((dim, dim), dtype=dtype, device=device)

    # Build dense matrix for each term
    for term in hamiltonian.terms:
        # Get coefficient (may be complex, but typically real)
        coeff = term.coeff

        # Build Kronecker product of Pauli matrices
        # For little-endian indexing (qubit 0 is LSB), we need to reverse
        # the order: P_{n-1} ⊗ P_{n-2} ⊗ ... ⊗ P_0
        # This matches the convention in PauliSum.to_matrix()
        term_matrix = _single_site_matrix(term.paulis[num_qubits - 1], dtype, device)
        for i in range(num_qubits - 2, -1, -1):
            pauli_i = _single_site_matrix(term.paulis[i], dtype, device)
            term_matrix = torch.kron(term_matrix, pauli_i)

        # Add to total matrix with coefficient
        H = H + coeff * term_matrix

    # Enforce Hermiticity numerically (harmless for analytic PauliSum input,
    # but helps with numerical roundoff)
    H = (H + H.conj().T) / 2.0

    return H


def exact_eigensystem(
    hamiltonian: PauliSum,
    num_qubits: int,
    k: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the full eigensystem of a PauliSum Hamiltonian via dense
    diagonalization.

    Parameters
    ----------
    hamiltonian:
        PauliSum Hamiltonian.
    num_qubits:
        Number of qubits.
    k:
        Reserved for future use (subset of eigenpairs). Currently must be
        None; full spectrum is always returned.
    device:
        Optional device.
    dtype:
        Complex dtype for dense matrix and eigenvectors.

    Returns
    -------
    eigenvalues, eigenvectors:
        - eigenvalues: real-valued tensor of shape (dim,)
        - eigenvectors: complex tensor of shape (dim, dim) whose columns are
          normalized eigenvectors corresponding to the eigenvalues.

    Raises
    ------
    ValueError:
        If k is not None (partial spectrum not yet supported).
    """
    if k is not None:
        raise ValueError("Partial spectrum (k != None) is not supported yet.")

    H = paulisum_to_dense(hamiltonian, num_qubits, device=device, dtype=dtype)

    # Compute eigensystem (eigenvalues are real for Hermitian matrices)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)

    # Ensure eigenvalues are real (they should be, but take .real for safety)
    eigenvalues = eigenvalues.real

    return eigenvalues, eigenvectors


def exact_ground_state(
    hamiltonian: PauliSum,
    num_qubits: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ground-state energy and eigenvector for a PauliSum Hamiltonian.

    Parameters
    ----------
    hamiltonian:
        PauliSum Hamiltonian.
    num_qubits:
        Number of qubits.
    device:
        Optional device.
    dtype:
        Complex dtype.

    Returns
    -------
    ground_energy, ground_state:
        - ground_energy: scalar tensor (float64) with the lowest eigenvalue.
        - ground_state: complex tensor of shape (dim,) normalized to 1.

    Raises
    ------
    RuntimeError:
        If the ground state eigenvector has zero norm.
    """
    eigenvalues, eigenvectors = exact_eigensystem(
        hamiltonian, num_qubits, device=device, dtype=dtype
    )

    # Ground state is the first eigenvector (lowest eigenvalue)
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    # Normalize explicitly
    norm = torch.linalg.norm(ground_state)
    if norm == 0.0:
        raise RuntimeError("Zero-norm eigenvector encountered.")
    ground_state = ground_state / norm

    return ground_energy, ground_state


