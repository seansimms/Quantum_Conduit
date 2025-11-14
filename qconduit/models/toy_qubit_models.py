"""Generic toy qubit Hamiltonian builders for small systems."""

from __future__ import annotations

from typing import List, Sequence

from qconduit.operators import PauliSum, PauliTerm


def two_qubit_generic_chemistry_like(
    c_i: float,
    c_z0: float,
    c_z1: float,
    c_z0z1: float,
    c_xx: float,
    c_yy: float,
) -> PauliSum:
    """
    Construct a generic 2-qubit Hamiltonian of the common chemistry-inspired form:

        H = c_i     * I⊗I
          + c_z0    * Z⊗I
          + c_z1    * I⊗Z
          + c_z0z1  * Z⊗Z
          + c_xx    * X⊗X
          + c_yy    * Y⊗Y

    This covers the standard structure obtained from Jordan–Wigner
    mappings of small two-orbital fermionic problems (e.g., H2 in a
    minimal basis), but the coefficients are fully user-specified.

    Parameters
    ----------
    c_i, c_z0, c_z1, c_z0z1, c_xx, c_yy:
        Real coefficients defining the Hamiltonian.

    Returns
    -------
    PauliSum
        The 2-qubit Hamiltonian as a PauliSum acting on qubits (0, 1).
    """
    terms: List[PauliTerm] = []

    # I⊗I term (identity)
    if abs(c_i) > 1e-15:  # Skip zero terms for efficiency
        terms.append(PauliTerm(coeff=float(c_i), paulis=("I", "I")))

    # Z⊗I term
    if abs(c_z0) > 1e-15:
        terms.append(PauliTerm(coeff=float(c_z0), paulis=("Z", "I")))

    # I⊗Z term
    if abs(c_z1) > 1e-15:
        terms.append(PauliTerm(coeff=float(c_z1), paulis=("I", "Z")))

    # Z⊗Z term
    if abs(c_z0z1) > 1e-15:
        terms.append(PauliTerm(coeff=float(c_z0z1), paulis=("Z", "Z")))

    # X⊗X term
    if abs(c_xx) > 1e-15:
        terms.append(PauliTerm(coeff=float(c_xx), paulis=("X", "X")))

    # Y⊗Y term
    if abs(c_yy) > 1e-15:
        terms.append(PauliTerm(coeff=float(c_yy), paulis=("Y", "Y")))

    return PauliSum(terms=terms)


def diagonal_z_field(
    num_qubits: int,
    local_fields: Sequence[float],
) -> PauliSum:
    """
    Construct a purely diagonal Hamiltonian of the form:

        H = sum_i h_i Z_i

    where `local_fields[i] = h_i`.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    local_fields:
        Sequence of length `num_qubits` with on-site field strengths.

    Returns
    -------
    PauliSum
        Diagonal Hamiltonian in the computational basis.

    Raises
    ------
    ValueError:
        If len(local_fields) != num_qubits or num_qubits <= 0.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if len(local_fields) != num_qubits:
        raise ValueError(
            f"local_fields must have length {num_qubits}, got {len(local_fields)}"
        )

    terms: List[PauliTerm] = []

    for i in range(num_qubits):
        if abs(local_fields[i]) > 1e-15:  # Skip zero terms
            paulis = ["I"] * num_qubits
            paulis[i] = "Z"
            terms.append(PauliTerm(coeff=float(local_fields[i]), paulis=tuple(paulis)))

    return PauliSum(terms=terms)


