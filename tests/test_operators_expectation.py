"""Tests for Pauli expectation evaluation."""

import math

import pytest
import torch

from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.gates.standard import H, X
from qconduit.operators.expectation import expectation_pauli_sum, expectation_pauli_term
from qconduit.operators.pauli import PauliSum, PauliTerm


class TestExpectationPauliTerm:
    """Tests for expectation_pauli_term function."""

    def test_expectation_z_0_state(self):
        """Test ⟨Z⟩ for |0⟩ state."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("Z",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(1.0), atol=1e-6)

    def test_expectation_z_1_state(self):
        """Test ⟨Z⟩ for |1⟩ state."""
        state = zero_state(n_qubits=1)
        gate_x = X(dtype=state.dtype, device=state.device)
        state = apply_gate(state, gate_x, qubit=0, n_qubits=1)
        term = PauliTerm(1.0, ("Z",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(-1.0), atol=1e-6)

    def test_expectation_x_plus_state(self):
        """Test ⟨X⟩ for |+⟩ = H|0⟩ state."""
        state = zero_state(n_qubits=1)
        gate_h = H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, gate_h, qubit=0, n_qubits=1)
        term = PauliTerm(1.0, ("X",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(1.0), atol=1e-6)

    def test_expectation_x_0_state(self):
        """Test ⟨X⟩ for |0⟩ state (should be 0)."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("X",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(0.0), atol=1e-6)

    def test_expectation_y_0_state(self):
        """Test ⟨Y⟩ for |0⟩ state (should be 0)."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("Y",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(0.0), atol=1e-6)

    def test_expectation_identity_term(self):
        """Test expectation of identity term."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(2.5, ("I",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(2.5), atol=1e-6)

    def test_expectation_2_qubit_zz(self):
        """Test ⟨Z⊗Z⟩ for 2-qubit states."""
        # |00⟩: ⟨Z⊗Z⟩ = +1
        state = zero_state(n_qubits=2)
        term = PauliTerm(1.0, ("Z", "Z"))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(1.0), atol=1e-6)

        # |01⟩: ⟨Z⊗Z⟩ = -1
        state = zero_state(n_qubits=2)
        gate_x = X(dtype=state.dtype, device=state.device)
        state = apply_gate(state, gate_x, qubit=0, n_qubits=2)
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(-1.0), atol=1e-6)

    def test_expectation_with_coefficient(self):
        """Test expectation with non-unit coefficient."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(0.5, ("Z",))
        exp = expectation_pauli_term(state, term)
        assert torch.allclose(exp, torch.tensor(0.5), atol=1e-6)

    def test_expectation_wrong_dimension(self):
        """Test that wrong state dimension raises ValueError."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("Z", "Z"))  # 2-qubit term
        with pytest.raises(ValueError, match="does not match"):
            expectation_pauli_term(state, term)

    def test_expectation_batched(self):
        """Test expectation with batched states."""
        # Batch: first is |0⟩, second is |1⟩
        state1 = zero_state(n_qubits=1)
        state2 = zero_state(n_qubits=1)
        gate_x = X(dtype=state1.dtype, device=state1.device)
        state2 = apply_gate(state2, gate_x, qubit=0, n_qubits=1)
        state_batch = torch.stack([state1, state2], dim=0)

        term = PauliTerm(1.0, ("Z",))
        exp = expectation_pauli_term(state_batch, term)

        assert exp.shape == (2,)
        assert torch.allclose(exp[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(exp[1], torch.tensor(-1.0), atol=1e-6)


class TestExpectationPauliSum:
    """Tests for expectation_pauli_sum function."""

    def test_expectation_single_term(self):
        """Test expectation with single term (should match expectation_pauli_term)."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])

        exp_sum = expectation_pauli_sum(state, hamiltonian)
        exp_term = expectation_pauli_term(state, term)

        assert torch.allclose(exp_sum, exp_term, atol=1e-6)

    def test_expectation_multiple_terms(self):
        """Test expectation with multiple terms."""
        state = zero_state(n_qubits=1)
        term_z = PauliTerm(1.0, ("Z",))
        term_x = PauliTerm(0.5, ("X",))
        hamiltonian = PauliSum.from_terms([term_z, term_x])

        exp = expectation_pauli_sum(state, hamiltonian)

        # For |0⟩: ⟨Z⟩ = 1, ⟨X⟩ = 0, so total = 1.0
        assert torch.allclose(exp, torch.tensor(1.0), atol=1e-6)

    def test_expectation_empty_hamiltonian(self):
        """Test expectation with empty Hamiltonian (should be 0)."""
        state = zero_state(n_qubits=1)
        hamiltonian = PauliSum()
        exp = expectation_pauli_sum(state, hamiltonian)
        assert torch.allclose(exp, torch.tensor(0.0), atol=1e-6)

    def test_expectation_wrong_dimension(self):
        """Test that wrong state dimension raises ValueError."""
        state = zero_state(n_qubits=1)
        term = PauliTerm(1.0, ("Z", "Z"))
        hamiltonian = PauliSum.from_terms([term])
        with pytest.raises(ValueError, match="does not match"):
            expectation_pauli_sum(state, hamiltonian)

    def test_expectation_batched(self):
        """Test expectation with batched states."""
        state1 = zero_state(n_qubits=1)
        state2 = zero_state(n_qubits=1)
        gate_x = X(dtype=state1.dtype, device=state1.device)
        state2 = apply_gate(state2, gate_x, qubit=0, n_qubits=1)
        state_batch = torch.stack([state1, state2], dim=0)

        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        exp = expectation_pauli_sum(state_batch, hamiltonian)

        assert exp.shape == (2,)
        assert torch.allclose(exp[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(exp[1], torch.tensor(-1.0), atol=1e-6)


class TestExpectationConsistencyWithMatrix:
    """Tests for consistency between expectation and matrix methods."""

    def test_expectation_vs_matrix_1_qubit(self):
        """Test that expectation matches matrix computation for 1-qubit."""
        # H = Z
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])

        # Test with |0⟩
        state = zero_state(n_qubits=1)
        exp_expectation = expectation_pauli_sum(state, hamiltonian)

        # Matrix computation: ⟨ψ|H|ψ⟩
        h_matrix = hamiltonian.to_matrix(dtype=state.dtype, device=state.device)
        state_conj = state.conj()
        exp_matrix = (state_conj @ (h_matrix @ state)).real

        assert torch.allclose(exp_expectation, exp_matrix, atol=1e-6)

        # Test with |+⟩ = H|0⟩
        gate_h = H(dtype=state.dtype, device=state.device)
        state_plus = apply_gate(state, gate_h, qubit=0, n_qubits=1)
        exp_expectation = expectation_pauli_sum(state_plus, hamiltonian)
        state_plus_conj = state_plus.conj()
        exp_matrix = (state_plus_conj @ (h_matrix @ state_plus)).real
        assert torch.allclose(exp_expectation, exp_matrix, atol=1e-6)

    def test_expectation_vs_matrix_1_qubit_combination(self):
        """Test expectation vs matrix for linear combination."""
        # H = 0.5 * Z + 0.5 * X
        term_z = PauliTerm(0.5, ("Z",))
        term_x = PauliTerm(0.5, ("X",))
        hamiltonian = PauliSum.from_terms([term_z, term_x])

        state = zero_state(n_qubits=1)
        exp_expectation = expectation_pauli_sum(state, hamiltonian)

        h_matrix = hamiltonian.to_matrix(dtype=state.dtype, device=state.device)
        state_conj = state.conj()
        exp_matrix = (state_conj @ (h_matrix @ state)).real

        assert torch.allclose(exp_expectation, exp_matrix, atol=1e-6)

    def test_expectation_vs_matrix_2_qubit(self):
        """Test that expectation matches matrix computation for 2-qubit."""
        # H = Z ⊗ I
        term = PauliTerm(1.0, ("Z", "I"))
        hamiltonian = PauliSum.from_terms([term])

        state = zero_state(n_qubits=2)
        exp_expectation = expectation_pauli_sum(state, hamiltonian)

        h_matrix = hamiltonian.to_matrix(dtype=state.dtype, device=state.device)
        state_conj = state.conj()
        exp_matrix = (state_conj @ (h_matrix @ state)).real

        assert torch.allclose(exp_expectation, exp_matrix, atol=1e-6)

        # Test with |01⟩
        gate_x = X(dtype=state.dtype, device=state.device)
        state_01 = apply_gate(state, gate_x, qubit=0, n_qubits=2)
        exp_expectation = expectation_pauli_sum(state_01, hamiltonian)
        state_01_conj = state_01.conj()
        exp_matrix = (state_01_conj @ (h_matrix @ state_01)).real
        assert torch.allclose(exp_expectation, exp_matrix, atol=1e-6)


