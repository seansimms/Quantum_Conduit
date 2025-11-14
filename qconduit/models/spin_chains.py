"""Standard 1D spin-chain Hamiltonian builders."""

from __future__ import annotations

from typing import List, Tuple

from qconduit.operators import PauliSum, PauliTerm


def _validate_chain_length(num_sites: int) -> None:
    """Validate that num_sites is positive."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")


def _nearest_neighbor_pairs(num_sites: int, periodic: bool) -> List[Tuple[int, int]]:
    """
    Generate nearest-neighbor pairs for a 1D chain.

    Parameters
    ----------
    num_sites:
        Number of sites in the chain.
    periodic:
        If True, include periodic boundary condition (last site connects to first).

    Returns
    -------
    List of (i, j) tuples representing nearest-neighbor pairs.
    """
    pairs: List[Tuple[int, int]] = []
    # Open chain: pairs (i, i+1) for i = 0 .. num_sites-2
    for i in range(num_sites - 1):
        pairs.append((i, i + 1))
    # Periodic boundary: add (num_sites-1, 0)
    if periodic and num_sites > 1:
        pairs.append((num_sites - 1, 0))
    return pairs


def transverse_field_ising_chain(
    num_sites: int,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
    periodic: bool = False,
) -> PauliSum:
    """
    Construct the 1D transverse-field Ising Hamiltonian on a spin-1/2 chain:

        H = - J * sum_{<i,j>} Z_i Z_j - h * sum_i X_i

    where <i,j> denotes nearest neighbors (with optional periodic boundary
    conditions).

    Parameters
    ----------
    num_sites:
        Number of spins (qubits) in the chain (must be positive).
    j_coupling:
        Coupling strength J for the ZZ interactions.
    h_field:
        Transverse field strength h multiplying the X terms.
    periodic:
        If True, use periodic boundary conditions (add term Z_{N-1} Z_0).
        If False, open chain.

    Returns
    -------
    PauliSum
        The Hamiltonian H as a sum of PauliTerms acting on `num_sites` qubits.
    """
    _validate_chain_length(num_sites)

    terms: List[PauliTerm] = []

    # Add ZZ interaction terms for nearest neighbors
    pairs = _nearest_neighbor_pairs(num_sites, periodic)
    for i, j in pairs:
        # Build Pauli string: Z on sites i and j, I elsewhere
        paulis = ["I"] * num_sites
        paulis[i] = "Z"
        paulis[j] = "Z"
        # Coefficient: -J
        terms.append(PauliTerm(coeff=-float(j_coupling), paulis=tuple(paulis)))

    # Add X field terms for each site
    for i in range(num_sites):
        paulis = ["I"] * num_sites
        paulis[i] = "X"
        # Coefficient: -h
        terms.append(PauliTerm(coeff=-float(h_field), paulis=tuple(paulis)))

    return PauliSum(terms=terms)


def heisenberg_xxz_chain(
    num_sites: int,
    j_coupling: float = 1.0,
    delta: float = 1.0,
    periodic: bool = False,
) -> PauliSum:
    """
    Construct the 1D spin-1/2 Heisenberg XXZ Hamiltonian:

        H = J * sum_{<i,j>} (X_i X_j + Y_i Y_j + Δ Z_i Z_j)

    Parameters
    ----------
    num_sites:
        Number of spins in the chain.
    j_coupling:
        Overall coupling strength J.
    delta:
        Anisotropy parameter Δ for the ZZ term.
    periodic:
        If True, periodic boundary conditions; else open chain.

    Returns
    -------
    PauliSum
        The XXZ Hamiltonian as a PauliSum.
    """
    _validate_chain_length(num_sites)

    terms: List[PauliTerm] = []

    # Add interaction terms for nearest neighbors
    pairs = _nearest_neighbor_pairs(num_sites, periodic)
    for i, j in pairs:
        # X_i X_j term
        paulis_xx = ["I"] * num_sites
        paulis_xx[i] = "X"
        paulis_xx[j] = "X"
        terms.append(PauliTerm(coeff=float(j_coupling), paulis=tuple(paulis_xx)))

        # Y_i Y_j term
        paulis_yy = ["I"] * num_sites
        paulis_yy[i] = "Y"
        paulis_yy[j] = "Y"
        terms.append(PauliTerm(coeff=float(j_coupling), paulis=tuple(paulis_yy)))

        # Z_i Z_j term with anisotropy
        paulis_zz = ["I"] * num_sites
        paulis_zz[i] = "Z"
        paulis_zz[j] = "Z"
        terms.append(PauliTerm(coeff=float(j_coupling * delta), paulis=tuple(paulis_zz)))

    return PauliSum(terms=terms)


def ising_zz_chain(
    num_sites: int,
    j_coupling: float = 1.0,
    periodic: bool = False,
) -> PauliSum:
    """
    Construct a simple 1D Ising ZZ Hamiltonian:

        H = - J * sum_{<i,j>} Z_i Z_j

    with optional periodic boundary conditions.

    Parameters
    ----------
    num_sites:
        Number of spins in the chain.
    j_coupling:
        Coupling strength J.
    periodic:
        If True, periodic boundary conditions; else open chain.

    Returns
    -------
    PauliSum
        The Ising ZZ Hamiltonian as a PauliSum.
    """
    _validate_chain_length(num_sites)

    terms: List[PauliTerm] = []

    # Add ZZ interaction terms for nearest neighbors
    pairs = _nearest_neighbor_pairs(num_sites, periodic)
    for i, j in pairs:
        paulis = ["I"] * num_sites
        paulis[i] = "Z"
        paulis[j] = "Z"
        # Coefficient: -J
        terms.append(PauliTerm(coeff=-float(j_coupling), paulis=tuple(paulis)))

    return PauliSum(terms=terms)


