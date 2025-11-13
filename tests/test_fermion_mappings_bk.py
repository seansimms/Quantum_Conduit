"""Tests for Bravyi-Kitaev fermion-to-qubit mapping."""

from __future__ import annotations

import pytest
import torch

from qconduit.exact import exact_eigensystem, paulisum_to_dense
from qconduit.fermion import (
    FermionOperator,
    FermionTerm,
    bravyi_kitaev,
    jordan_wigner,
)


class TestBravyiKitaev:
    """Tests for Bravyi-Kitaev mapping."""

    def test_bk_number_operator_one_mode(self):
        """Test BK mapping of number operator n_0 = a_0^\\dagger a_0 for 1 mode."""
        # Fermionic operator: n_0 = a_0^\dagger a_0
        term = FermionTerm(coeff=1.0, operators=((0, "+"), (0, "-")))
        fop = FermionOperator(terms=(term,))

        # Map with both JW and BK
        H_jw = jordan_wigner(fop, n_spin_orbitals=1)
        H_bk = bravyi_kitaev(fop, n_spin_orbitals=1)

        # Compute eigenvalues
        evals_jw, _ = exact_eigensystem(H_jw, num_qubits=1)
        evals_bk, _ = exact_eigensystem(H_bk, num_qubits=1)

        # Eigenvalues should match (sorted)
        evals_jw_sorted = torch.sort(evals_jw)[0]
        evals_bk_sorted = torch.sort(evals_bk)[0]
        assert torch.allclose(evals_jw_sorted, evals_bk_sorted, atol=1e-7)

    def test_bk_hopping_term_two_modes(self):
        """Test BK mapping of hopping term a_0^\\dagger a_1 + h.c. for 2 modes."""
        # Fermionic operator: a_0^\dagger a_1 + a_1^\dagger a_0
        t1 = FermionTerm(1.0, ((0, "+"), (1, "-")))
        t2 = FermionTerm(1.0, ((1, "+"), (0, "-")))
        fop = FermionOperator(terms=(t1, t2))

        # Map with both JW and BK
        H_jw = jordan_wigner(fop, n_spin_orbitals=2)
        H_bk = bravyi_kitaev(fop, n_spin_orbitals=2)

        # Compute eigenvalues
        evals_jw, _ = exact_eigensystem(H_jw, num_qubits=2)
        evals_bk, _ = exact_eigensystem(H_bk, num_qubits=2)

        # Eigenvalues should match (sorted)
        evals_jw_sorted = torch.sort(evals_jw)[0]
        evals_bk_sorted = torch.sort(evals_bk)[0]
        assert torch.allclose(evals_jw_sorted, evals_bk_sorted, atol=1e-7)

    def test_bk_hermiticity_number_operator(self):
        """Test that BK-mapped number operator is Hermitian."""
        # Number operator: n_0 + n_1
        t1 = FermionTerm(1.0, ((0, "+"), (0, "-")))
        t2 = FermionTerm(1.0, ((1, "+"), (1, "-")))
        fop = FermionOperator(terms=(t1, t2))

        H_bk = bravyi_kitaev(fop, n_spin_orbitals=2)
        dense_bk = paulisum_to_dense(H_bk, num_qubits=2)

        # Check Hermiticity: H = H^\dagger
        assert torch.allclose(dense_bk, dense_bk.conj().T, atol=1e-10)

    def test_bk_hermiticity_hopping(self):
        """Test that BK-mapped Hermitian fermionic operator is Hermitian."""
        # Hopping term: a_0^\dagger a_1 + a_1^\dagger a_0 (already Hermitian)
        t1 = FermionTerm(1.0, ((0, "+"), (1, "-")))
        t2 = FermionTerm(1.0, ((1, "+"), (0, "-")))
        fop = FermionOperator(terms=(t1, t2))

        H_bk = bravyi_kitaev(fop, n_spin_orbitals=2)
        dense_bk = paulisum_to_dense(H_bk, num_qubits=2)

        # Check Hermiticity
        assert torch.allclose(dense_bk, dense_bk.conj().T, atol=1e-10)

    def test_bk_zero_operator(self):
        """Test BK mapping of zero operator."""
        fop = FermionOperator(terms=())
        H_bk = bravyi_kitaev(fop, n_spin_orbitals=2)
        assert H_bk.n_qubits() == 0 or len(H_bk.terms) == 0

        # Dense representation should be all zeros
        if H_bk.n_qubits() > 0:
            dense_bk = paulisum_to_dense(H_bk, num_qubits=2)
            assert torch.allclose(dense_bk, torch.zeros((4, 4), dtype=torch.complex128), atol=1e-12)

    def test_bk_invalid_n_spin_orbitals(self):
        """Test that invalid n_spin_orbitals raises ValueError."""
        term = FermionTerm(coeff=1.0, operators=((0, "+"),))
        fop = FermionOperator(terms=(term,))

        with pytest.raises(ValueError, match="n_spin_orbitals must be >= 1"):
            bravyi_kitaev(fop, n_spin_orbitals=0)

        with pytest.raises(ValueError, match="n_spin_orbitals must be >= 1"):
            bravyi_kitaev(fop, n_spin_orbitals=-1)

    def test_bk_invalid_mode_index(self):
        """Test that mode index out of range raises ValueError."""
        term = FermionTerm(coeff=1.0, operators=((2, "+"),))  # mode 2 for n=2
        fop = FermionOperator(terms=(term,))

        with pytest.raises(ValueError, match="Mode index.*out of range"):
            bravyi_kitaev(fop, n_spin_orbitals=2)

    def test_bk_two_body_term(self):
        """Test BK mapping of two-body term and spectral equivalence with JW."""
        # Number operator product: n_0 n_1
        term = FermionTerm(
            coeff=1.0, operators=((0, "+"), (1, "+"), (1, "-"), (0, "-"))
        )
        fop = FermionOperator(terms=(term,))

        # Map with both JW and BK
        H_jw = jordan_wigner(fop, n_spin_orbitals=2)
        H_bk = bravyi_kitaev(fop, n_spin_orbitals=2)

        # Compute eigenvalues
        evals_jw, _ = exact_eigensystem(H_jw, num_qubits=2)
        evals_bk, _ = exact_eigensystem(H_bk, num_qubits=2)

        # Eigenvalues should match (sorted)
        evals_jw_sorted = torch.sort(evals_jw)[0]
        evals_bk_sorted = torch.sort(evals_bk)[0]
        assert torch.allclose(evals_jw_sorted, evals_bk_sorted, atol=1e-7)

    def test_bk_creation_annihilation_spectral_equivalence(self):
        """Test that BK and JW give same spectrum for creation/annihilation operators."""
        # Test a_0^\dagger
        term1 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        fop1 = FermionOperator(terms=(term1,))

        H_jw1 = jordan_wigner(fop1, n_spin_orbitals=1)
        H_bk1 = bravyi_kitaev(fop1, n_spin_orbitals=1)

        evals_jw1, _ = exact_eigensystem(H_jw1, num_qubits=1)
        evals_bk1, _ = exact_eigensystem(H_bk1, num_qubits=1)

        evals_jw1_sorted = torch.sort(evals_jw1)[0]
        evals_bk1_sorted = torch.sort(evals_bk1)[0]
        assert torch.allclose(evals_jw1_sorted, evals_bk1_sorted, atol=1e-7)

        # Test a_0
        term2 = FermionTerm(coeff=1.0, operators=((0, "-"),))
        fop2 = FermionOperator(terms=(term2,))

        H_jw2 = jordan_wigner(fop2, n_spin_orbitals=1)
        H_bk2 = bravyi_kitaev(fop2, n_spin_orbitals=1)

        evals_jw2, _ = exact_eigensystem(H_jw2, num_qubits=1)
        evals_bk2, _ = exact_eigensystem(H_bk2, num_qubits=1)

        evals_jw2_sorted = torch.sort(evals_jw2)[0]
        evals_bk2_sorted = torch.sort(evals_bk2)[0]
        assert torch.allclose(evals_jw2_sorted, evals_bk2_sorted, atol=1e-7)

