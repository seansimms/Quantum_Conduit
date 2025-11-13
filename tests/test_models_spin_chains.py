"""Tests for spin-chain Hamiltonian builders."""

from __future__ import annotations

import torch

from qconduit.exact import exact_eigensystem, paulisum_to_dense
from qconduit.models import (
    heisenberg_xxz_chain,
    ising_zz_chain,
    transverse_field_ising_chain,
)


def test_tfim_term_counts():
    """Test that TFIM generates correct number of terms."""
    # Open chain: 3 sites -> 2 ZZ pairs + 3 X terms = 5 terms
    H = transverse_field_ising_chain(num_sites=3, periodic=False)
    assert len(H.terms) == 5

    # Periodic chain: 3 sites -> 3 ZZ pairs + 3 X terms = 6 terms
    H_periodic = transverse_field_ising_chain(num_sites=3, periodic=True)
    assert len(H_periodic.terms) == 6


def test_tfim_sign_conventions():
    """Test TFIM sign conventions against analytic eigenvalues."""
    # For num_sites=2, j_coupling=1.0, h_field=0.0, open chain:
    # H = -Z0 Z1
    H = transverse_field_ising_chain(num_sites=2, j_coupling=1.0, h_field=0.0, periodic=False)

    # Compute eigenvalues
    evals, _ = exact_eigensystem(H, num_qubits=2)

    # Analytic eigenvalues for -Z0 Z1:
    # In computational basis: |00> -> +1, |01> -> -1, |10> -> -1, |11> -> +1
    # So eigenvalues are [-1, -1, 1, 1] (sorted: [-1, -1, 1, 1])
    expected_evals = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=evals.dtype)
    evals_sorted = torch.sort(evals)[0]
    expected_sorted = torch.sort(expected_evals)[0]

    assert torch.allclose(evals_sorted, expected_sorted, atol=1e-10)


def test_heisenberg_xxz_structure():
    """Test that Heisenberg XXZ generates correct term types."""
    H = heisenberg_xxz_chain(num_sites=2, j_coupling=1.0, delta=1.0, periodic=False)

    # Should have exactly 3 terms: X0X1, Y0Y1, Z0Z1
    assert len(H.terms) == 3

    # Check term types
    pauli_strings = [term.paulis for term in H.terms]

    # Should have one X/X, one Y/Y, and one Z/Z
    has_xx = any(paulis == ("X", "X") for paulis in pauli_strings)
    has_yy = any(paulis == ("Y", "Y") for paulis in pauli_strings)
    has_zz = any(paulis == ("Z", "Z") for paulis in pauli_strings)

    assert has_xx
    assert has_yy
    assert has_zz


def test_ising_zz_chain_parity():
    """Test Ising ZZ chain coefficients and structure."""
    H = ising_zz_chain(num_sites=3, j_coupling=2.0, periodic=True)

    # Should have 3 ZZ terms for periodic chain
    assert len(H.terms) == 3

    # Each term should have coefficient -2.0
    for term in H.terms:
        assert abs(term.coeff - (-2.0)) < 1e-10
        # Should be a ZZ term (exactly 2 Z's, rest I's)
        z_count = sum(1 for p in term.paulis if p == "Z")
        assert z_count == 2

    # Check that pairs are (0,1), (1,2), (2,0)
    pairs = []
    for term in H.terms:
        z_indices = [i for i, p in enumerate(term.paulis) if p == "Z"]
        pairs.append(tuple(sorted(z_indices)))

    expected_pairs = [(0, 1), (1, 2), (0, 2)]  # (2,0) becomes (0,2) when sorted
    assert set(pairs) == set(expected_pairs)


def test_tfim_validation():
    """Test that TFIM validates inputs correctly."""
    # Should raise ValueError for num_sites <= 0
    try:
        transverse_field_ising_chain(num_sites=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        transverse_field_ising_chain(num_sites=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_heisenberg_xxz_validation():
    """Test that Heisenberg XXZ validates inputs correctly."""
    try:
        heisenberg_xxz_chain(num_sites=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_ising_zz_validation():
    """Test that Ising ZZ validates inputs correctly."""
    try:
        ising_zz_chain(num_sites=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_tfim_single_site():
    """Test TFIM with single site (only X term, no ZZ)."""
    H = transverse_field_ising_chain(num_sites=1, j_coupling=1.0, h_field=0.5, periodic=False)

    # Should have 1 term: X0
    assert len(H.terms) == 1
    assert H.terms[0].paulis == ("X",)
    assert abs(H.terms[0].coeff - (-0.5)) < 1e-10


def test_heisenberg_xxz_single_site():
    """Test Heisenberg XXZ with single site (no interactions)."""
    H = heisenberg_xxz_chain(num_sites=1, j_coupling=1.0, delta=1.0, periodic=False)

    # Should have 0 terms (no neighbors)
    assert len(H.terms) == 0


def test_ising_zz_single_site():
    """Test Ising ZZ with single site (no interactions)."""
    H = ising_zz_chain(num_sites=1, j_coupling=1.0, periodic=False)

    # Should have 0 terms (no neighbors)
    assert len(H.terms) == 0

