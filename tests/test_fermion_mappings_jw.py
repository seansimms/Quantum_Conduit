"""Tests for Jordan-Wigner fermion-to-qubit mapping."""

from __future__ import annotations

import pytest
import torch

from qconduit.exact import paulisum_to_dense
from qconduit.fermion import (
    FermionOperator,
    FermionTerm,
    jordan_wigner,
)
from qconduit.operators import PauliSum, PauliTerm


def fermion_operator_to_dense(
    fermion_op: FermionOperator,
    n_spin_orbitals: int,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Naive dense representation of a FermionOperator on the occupation basis
    for n_spin_orbitals <= 2 (2^n dimension).

    For tests only. This implements the standard fermionic operator action
    on occupation number basis states.

    Parameters
    ----------
    fermion_op:
        Fermionic operator to convert.
    n_spin_orbitals:
        Number of spin-orbitals (must be <= 2 for this implementation).
    dtype:
        Complex dtype for the matrix.

    Returns
    -------
    torch.Tensor
        Dense matrix representation of the fermionic operator.
    """
    if n_spin_orbitals > 2:
        raise ValueError("fermion_operator_to_dense only supports n_spin_orbitals <= 2")

    dim = 2**n_spin_orbitals
    matrix = torch.zeros((dim, dim), dtype=dtype, device=torch.device("cpu"))

    # Build basis states as occupation vectors
    # State |n0 n1 ...> where ni ∈ {0, 1}
    # Index i corresponds to binary representation: i = sum(nj * 2^j)

    for i in range(dim):
        # Decompose index i into occupation numbers
        occupations = [(i >> j) & 1 for j in range(n_spin_orbitals)]

        # Apply each term in the fermionic operator
        for term in fermion_op.terms:
            # Apply the operator string to this basis state
            result_states = [(1.0, occupations.copy())]  # (coeff, state)

            # Apply each ladder operator in order (right to left for second quantization)
            # In second quantization, a^\dagger a means: apply a first, then a^\dagger
            for mode, op_type in reversed(term.operators):
                new_results = []
                for coeff, state in result_states:
                    if op_type == "-":  # Annihilation a_p
                        if state[mode] == 0:
                            # Annihilation on empty mode gives zero
                            continue
                        else:
                            # a_p |1> = (-1)^{sum_{j<p} n_j} |0>
                            parity = sum(state[j] for j in range(mode))
                            sign = (-1) ** parity
                            new_state = state.copy()
                            new_state[mode] = 0
                            new_results.append((coeff * sign, new_state))
                    else:  # Creation a_p^\dagger
                        if state[mode] == 1:
                            # Creation on occupied mode gives zero
                            continue
                        else:
                            # a_p^\dagger |0> = (-1)^{sum_{j<p} n_j} |1>
                            parity = sum(state[j] for j in range(mode))
                            sign = (-1) ** parity
                            new_state = state.copy()
                            new_state[mode] = 1
                            new_results.append((coeff * sign, new_state))
                result_states = new_results
                if not result_states:
                    break

            # Add contributions to matrix
            for coeff, final_state in result_states:
                # Convert final state to index
                final_idx = sum(final_state[j] * (2**j) for j in range(n_spin_orbitals))
                matrix[final_idx, i] += term.coeff * coeff

    return matrix


class TestJordanWigner:
    """Tests for Jordan-Wigner mapping."""

    def test_jw_number_operator_one_mode(self):
        """Test JW mapping of number operator n_0 = a_0^\\dagger a_0 for 1 mode."""
        # Fermionic operator: n_0 = a_0^\dagger a_0
        term = FermionTerm(coeff=1.0, operators=((0, "+"), (0, "-")))
        fop = FermionOperator(terms=(term,))

        # JW map
        H_jw = jordan_wigner(fop, n_spin_orbitals=1)
        dense_jw = paulisum_to_dense(H_jw, num_qubits=1)

        # True fermionic operator matrix
        dense_f = fermion_operator_to_dense(fop, n_spin_orbitals=1)

        # Compare
        assert torch.allclose(dense_jw, dense_f, atol=1e-7)

        # Also check analytic form: n = (I - Z)/2 for 1 qubit
        I_term = PauliTerm(coeff=0.5, paulis=("I",))
        Z_term = PauliTerm(coeff=-0.5, paulis=("Z",))
        H_analytic = PauliSum.from_terms([I_term, Z_term])
        dense_analytic = paulisum_to_dense(H_analytic, num_qubits=1)
        assert torch.allclose(dense_jw, dense_analytic, atol=1e-7)

    def test_jw_hopping_term_two_modes(self):
        """Test JW mapping of hopping term a_0^\\dagger a_1 + h.c. for 2 modes."""
        # Fermionic operator: a_0^\dagger a_1 + a_1^\dagger a_0
        t1 = FermionTerm(1.0, ((0, "+"), (1, "-")))
        t2 = FermionTerm(1.0, ((1, "+"), (0, "-")))
        fop = FermionOperator(terms=(t1, t2))

        # Compute dense fermion matrix
        dense_f = fermion_operator_to_dense(fop, n_spin_orbitals=2)

        # JW map
        H_jw = jordan_wigner(fop, n_spin_orbitals=2)
        dense_jw = paulisum_to_dense(H_jw, num_qubits=2)

        # Compare
        assert torch.allclose(dense_jw, dense_f, atol=1e-7)

    def test_jw_zero_operator(self):
        """Test JW mapping of zero operator."""
        fop = FermionOperator(terms=())
        H_jw = jordan_wigner(fop, n_spin_orbitals=2)
        assert H_jw.n_qubits() == 0 or len(H_jw.terms) == 0

        # Dense representation should be all zeros
        if H_jw.n_qubits() > 0:
            dense_jw = paulisum_to_dense(H_jw, num_qubits=2)
            assert torch.allclose(dense_jw, torch.zeros((4, 4), dtype=torch.complex128), atol=1e-12)

    def test_jw_invalid_n_spin_orbitals(self):
        """Test that invalid n_spin_orbitals raises ValueError."""
        term = FermionTerm(coeff=1.0, operators=((0, "+"),))
        fop = FermionOperator(terms=(term,))

        with pytest.raises(ValueError, match="n_spin_orbitals must be >= 1"):
            jordan_wigner(fop, n_spin_orbitals=0)

        with pytest.raises(ValueError, match="n_spin_orbitals must be >= 1"):
            jordan_wigner(fop, n_spin_orbitals=-1)

    def test_jw_invalid_mode_index(self):
        """Test that mode index out of range raises ValueError."""
        term = FermionTerm(coeff=1.0, operators=((2, "+"),))  # mode 2 for n=2
        fop = FermionOperator(terms=(term,))

        with pytest.raises(ValueError, match="Mode index.*out of range"):
            jordan_wigner(fop, n_spin_orbitals=2)

    def test_jw_creation_operator(self):
        """Test JW mapping of single creation operator (non-Hermitian, so only check structure)."""
        term = FermionTerm(coeff=1.0, operators=((0, "+"),))
        fop = FermionOperator(terms=(term,))

        H_jw = jordan_wigner(fop, n_spin_orbitals=1)
        # Creation operator is non-Hermitian, so it can't be exactly represented
        # as a real PauliSum. We just check that it produces a valid PauliSum.
        assert H_jw.n_qubits() == 1
        assert len(H_jw.terms) > 0

    def test_jw_annihilation_operator(self):
        """Test JW mapping of single annihilation operator (non-Hermitian)."""
        term = FermionTerm(coeff=1.0, operators=((0, "-"),))
        fop = FermionOperator(terms=(term,))

        H_jw = jordan_wigner(fop, n_spin_orbitals=1)
        # Annihilation operator is non-Hermitian, so it can't be exactly represented
        # as a real PauliSum. We just check that it produces a valid PauliSum.
        assert H_jw.n_qubits() == 1
        assert len(H_jw.terms) > 0

    def test_jw_two_body_term(self):
        """Test JW mapping of two-body term a_0^\\dagger a_1^\\dagger a_1 a_0."""
        # Number operator product: n_0 n_1
        term = FermionTerm(
            coeff=1.0, operators=((0, "+"), (1, "+"), (1, "-"), (0, "-"))
        )
        fop = FermionOperator(terms=(term,))

        H_jw = jordan_wigner(fop, n_spin_orbitals=2)
        dense_jw = paulisum_to_dense(H_jw, num_qubits=2)
        dense_f = fermion_operator_to_dense(fop, n_spin_orbitals=2)

        assert torch.allclose(dense_jw, dense_f, atol=1e-7)

