"""State tomography utilities using Pauli expectation values."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from qconduit.core.device import default_device

# Standard 2x2 Pauli matrices
_I = torch.eye(2, dtype=torch.complex128)
_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
_Y = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex128)
_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)

_PAULI_DICT = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
}


def pauli_matrix_from_label(
    label: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct the n-qubit Pauli operator corresponding to a label string.

    Examples:
        "X"   -> X (1 qubit)
        "Z"   -> Z
        "IX"  -> I ⊗ X  (2 qubits, qubit 0 = I, qubit 1 = X)
        "ZZ"  -> Z ⊗ Z
        "XYZ" -> X ⊗ Y ⊗ Z

    Parameters
    ----------
    label:
        String of characters in {"I", "X", "Y", "Z"} with length n >= 1.
    device:
        Optional device for the resulting tensor.

    Returns
    -------
    torch.Tensor
        2D complex tensor of shape (2**n, 2**n) with the corresponding Pauli operator.

    Raises
    ------
    ValueError
        If the label is empty or contains invalid characters.
    """
    if len(label) < 1:
        raise ValueError("label must be non-empty")
    
    # Validate all characters
    for c in label:
        if c not in _PAULI_DICT:
            raise ValueError(f"Invalid Pauli label character '{c}'. Must be one of {{'I', 'X', 'Y', 'Z'}}")
    
    # Build tensor product
    op = _PAULI_DICT[label[0]]
    for c in label[1:]:
        op = torch.kron(op, _PAULI_DICT[c])
    
    # Move to device if provided
    if device is not None:
        op = op.to(device=device)
    
    return op


def pauli_expectation_from_statevector(
    state: torch.Tensor,
    label: str,
) -> complex:
    """
    Compute ⟨ψ|P|ψ⟩ for a small system, where P is an n-qubit Pauli operator
    specified by `label`.

    The statevector must have length 2**n, where n == len(label).

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ⟩.
    label:
        String of length n in {"I", "X", "Y", "Z"}.

    Returns
    -------
    complex
        The complex expectation value ⟨ψ|P|ψ⟩.

    Raises
    ------
    ValueError
        If state is invalid, label is invalid, or dimensions don't match.
    """
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")
    
    dim = state.shape[0]
    if dim == 0:
        raise ValueError("state must have non-zero length")
    
    n_qubits = len(label)
    if n_qubits < 1:
        raise ValueError("label must be non-empty")
    
    expected_dim = 1 << n_qubits
    if dim != expected_dim:
        raise ValueError(
            f"State dimension {dim} does not match label length {n_qubits} "
            f"(expected 2**{n_qubits} = {expected_dim})."
        )
    
    # Cast state to complex128
    state_c = state.to(dtype=torch.complex128)
    
    # Build Pauli operator
    P = pauli_matrix_from_label(label, device=state_c.device)
    
    # Compute ⟨ψ|P|ψ⟩
    bra = state_c.conj().unsqueeze(0)  # (1, dim)
    ket = state_c.unsqueeze(1)         # (dim, 1)
    value = (bra @ (P @ ket)).item()
    
    return complex(value)


def single_qubit_pauli_expectations_from_statevector(
    state: torch.Tensor,
) -> tuple[float, float, float]:
    """
    Compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for a single-qubit pure state.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2,) representing |ψ⟩.

    Returns
    -------
    (ex_x, ex_y, ex_z):
        Tuple of real-valued expectations.

    Raises
    ------
    ValueError
        If state is not shape (2,) or has zero norm.
    """
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")
    if state.shape[0] != 2:
        raise ValueError(f"state must have length 2 for single-qubit, got {state.shape[0]}")
    
    # Normalize state
    s = state.to(dtype=torch.complex128)
    norm = s.abs().pow(2).sum().sqrt()
    if norm == 0:
        raise ValueError("Statevector has zero norm.")
    s = s / norm
    
    # Compute expectations
    ex_x = pauli_expectation_from_statevector(s, "X")
    ex_y = pauli_expectation_from_statevector(s, "Y")
    ex_z = pauli_expectation_from_statevector(s, "Z")
    
    # Return real parts as floats
    return (float(ex_x.real), float(ex_y.real), float(ex_z.real))


def reconstruct_single_qubit_density_from_pauli(
    ex_x: float,
    ex_y: float,
    ex_z: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Reconstruct a single-qubit density matrix from Pauli expectation values:

        ρ = 1/2 (I + ⟨X⟩ X + ⟨Y⟩ Y + ⟨Z⟩ Z).

    Parameters
    ----------
    ex_x, ex_y, ex_z:
        Real-valued expectation estimates of X, Y, Z.
    device:
        Optional device for the resulting tensor.

    Returns
    -------
    torch.Tensor
        2x2 complex density matrix ρ, Hermitian with trace ~ 1.
    """
    ex_x_t = float(ex_x)
    ex_y_t = float(ex_y)
    ex_z_t = float(ex_z)
    
    # Get Pauli matrices on the correct device
    I_mat = _I.to(device=device) if device is not None else _I
    X_mat = _X.to(device=device) if device is not None else _X
    Y_mat = _Y.to(device=device) if device is not None else _Y
    Z_mat = _Z.to(device=device) if device is not None else _Z
    
    rho = 0.5 * (
        I_mat
        + ex_x_t * X_mat
        + ex_y_t * Y_mat
        + ex_z_t * Z_mat
    )
    
    return rho


def two_qubit_pauli_expectations_from_statevector(
    state: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute ⟨σ_a ⊗ σ_b⟩ for all a,b ∈ {I,X,Y,Z} for a two-qubit pure state.

    Parameters
    ----------
    state:
        1D complex tensor of shape (4,) representing |ψ⟩.

    Returns
    -------
    Dict[str, float]
        Mapping from Pauli labels ("II", "IX", ..., "ZZ") to real expectation values.

    Raises
    ------
    ValueError
        If state is not shape (4,) or has zero norm.
    """
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")
    if state.shape[0] != 4:
        raise ValueError(f"state must have length 4 for two-qubit, got {state.shape[0]}")
    
    # Normalize state
    s = state.to(dtype=torch.complex128)
    norm = s.abs().pow(2).sum().sqrt()
    if norm == 0:
        raise ValueError("Statevector has zero norm.")
    s = s / norm
    
    # Required labels
    required_labels = [
        "II", "IX", "IY", "IZ",
        "XI", "XX", "XY", "XZ",
        "YI", "YX", "YY", "YZ",
        "ZI", "ZX", "ZY", "ZZ",
    ]
    
    # Compute expectations
    expectations = {}
    for label in required_labels:
        ex = pauli_expectation_from_statevector(s, label)
        expectations[label] = float(ex.real)
    
    return expectations


def reconstruct_two_qubit_density_from_pauli(
    expectations: Mapping[str, float],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Reconstruct a two-qubit density matrix from Pauli tensor expectations.

    The density matrix is expanded as:

        ρ = 1/4 ∑_{a,b ∈ {I,X,Y,Z}} c_{ab} (σ_a ⊗ σ_b),

    where c_{ab} = ⟨σ_a ⊗ σ_b⟩ are real-valued expectation estimates.

    The `expectations` mapping must provide entries for all 16 labels:

        "II", "IX", "IY", "IZ",
        "XI", "XX", "XY", "XZ",
        "YI", "YX", "YY", "YZ",
        "ZI", "ZX", "ZY", "ZZ".

    Parameters
    ----------
    expectations:
        Mapping from 2-character Pauli labels to real-valued expectations.
    device:
        Optional device for the resulting tensor.

    Returns
    -------
    torch.Tensor
        4x4 complex density matrix ρ.

    Raises
    ------
    ValueError
        If any required label is missing from expectations.
    """
    required_labels = [
        "II", "IX", "IY", "IZ",
        "XI", "XX", "XY", "XZ",
        "YI", "YX", "YY", "YZ",
        "ZI", "ZX", "ZY", "ZZ",
    ]
    
    # Validate all labels are present
    missing = [lbl for lbl in required_labels if lbl not in expectations]
    if missing:
        raise ValueError(
            f"Missing required expectation values for labels: {missing}"
        )
    
    # Initialize density matrix
    if device is None:
        device = default_device().as_torch_device()
    
    rho = torch.zeros((4, 4), dtype=torch.complex128, device=device)
    
    # Accumulate terms
    for lbl in required_labels:
        c = float(expectations[lbl])
        P = pauli_matrix_from_label(lbl, device=device)
        rho = rho + c * P
    
    # Scale by 1/4
    rho = 0.25 * rho
    
    return rho


__all__ = [
    "pauli_matrix_from_label",
    "pauli_expectation_from_statevector",
    "single_qubit_pauli_expectations_from_statevector",
    "reconstruct_single_qubit_density_from_pauli",
    "two_qubit_pauli_expectations_from_statevector",
    "reconstruct_two_qubit_density_from_pauli",
]


