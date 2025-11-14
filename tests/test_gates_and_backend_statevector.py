"""Comprehensive tests for gates and statevector backend operations.

This test module validates:
1. Gate unitarity and matrix correctness
2. Statevector application semantics
3. Measurement and probabilities
4. Expectation values
5. Normalization diagnostics
"""

import pytest
import torch
import math
import qconduit as qc
from qconduit.gates.standard import is_unitary


class TestGateUnitarityAndMatrixCorrectness:
    """Test gate unitarity and matrix correctness for all gates."""

    def test_single_qubit_gates_unitarity(self):
        """Test all single-qubit gates are unitary."""
        gates = {
            "I": qc.I(),
            "X": qc.X(),
            "Y": qc.Y(),
            "Z": qc.Z(),
            "H": qc.H(),
            "S": qc.S(),
            "T": qc.T(),
        }
        for name, gate in gates.items():
            assert is_unitary(gate, atol=1e-6), f"{name} gate is not unitary"
            # Check U @ U.conj().T ≈ I
            adjoint = gate.conj().transpose(-1, -2)
            product = torch.matmul(adjoint, gate)
            identity = torch.eye(2, dtype=gate.dtype, device=gate.device)
            assert torch.allclose(product, identity, atol=1e-6), f"{name} fails U†U = I"

    def test_rotation_gates_unitarity(self):
        """Test rotation gates are unitary for various angles."""
        angles = [0.0, 0.1, 0.5, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]
        for angle in angles:
            for gate_func in [qc.RX, qc.RY, qc.RZ]:
                gate = gate_func(angle)
                assert is_unitary(gate, atol=1e-6), f"{gate_func.__name__}({angle}) is not unitary"

    def test_cnot_unitarity(self):
        """Test CNOT gate is unitary."""
        cnot = qc.CNOT()
        assert is_unitary(cnot, atol=1e-6)
        # Check U @ U.conj().T ≈ I
        adjoint = cnot.conj().transpose(-1, -2)
        product = torch.matmul(adjoint, cnot)
        identity = torch.eye(4, dtype=cnot.dtype, device=cnot.device)
        assert torch.allclose(product, identity, atol=1e-6)

    def test_reference_matrices(self):
        """Test gates match known reference matrices."""
        # X gate
        x_gate = qc.X()
        expected_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
        assert torch.allclose(x_gate, expected_x)

        # Z gate
        z_gate = qc.Z()
        expected_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
        assert torch.allclose(z_gate, expected_z)

        # H gate
        h_gate = qc.H()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected_h = torch.tensor(
            [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=torch.complex64
        )
        assert torch.allclose(h_gate, expected_h)

    def test_rotation_gate_properties(self):
        """Test rotation gates have correct properties."""
        # RX(0) = I
        rx_zero = qc.RX(0.0)
        identity = qc.I()
        assert torch.allclose(rx_zero, identity, atol=1e-6)

        # RX(π) ≈ -iX (up to global phase)
        rx_pi = qc.RX(math.pi)
        x_gate = qc.X()
        # Check that |RX(π)| matches |X| (ignoring global phase)
        assert torch.allclose(torch.abs(rx_pi), torch.abs(x_gate), atol=1e-6)

        # RY(0) = I
        ry_zero = qc.RY(0.0)
        assert torch.allclose(ry_zero, identity, atol=1e-6)

        # RZ(0) = I
        rz_zero = qc.RZ(0.0)
        assert torch.allclose(rz_zero, identity, atol=1e-6)

    def test_cnot_truth_table(self):
        """Test CNOT truth table on all basis states."""
        cnot = qc.CNOT(control_first=True)
        # Test on |00⟩, |01⟩, |10⟩, |11⟩
        basis_states = [
            (0, 0),  # |00⟩ -> |00⟩
            (1, 0),  # |01⟩ -> |01⟩
            (2, 3),  # |10⟩ -> |11⟩
            (3, 2),  # |11⟩ -> |10⟩
        ]
        for input_idx, expected_idx in basis_states:
            state = torch.zeros(4, dtype=torch.complex64)
            state[input_idx] = 1.0
            result = cnot @ state
            expected = torch.zeros(4, dtype=torch.complex64)
            expected[expected_idx] = 1.0
            assert torch.allclose(result, expected, atol=1e-6)


class TestStatevectorApplicationSemantics:
    """Test statevector application semantics for simple sequences."""

    def test_single_qubit_h_on_zero(self):
        """Test H|0⟩ = (|0⟩ + |1⟩)/√2."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        result = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex64)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_single_qubit_rx_pi_half(self):
        """Test RX(π/2)|0⟩."""
        state = qc.zero_state(n_qubits=1)
        rx_gate = qc.RX(math.pi / 2.0)
        result = qc.apply_gate(state, rx_gate, qubit=0, n_qubits=1)

        # RX(π/2)|0⟩ = (|0⟩ - i|1⟩)/√2
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.tensor([sqrt2_inv, -1.0j * sqrt2_inv], dtype=torch.complex64)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_bell_state_creation(self):
        """Test Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        state = qc.zero_state(n_qubits=2)  # |00⟩
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)  # (|00⟩ + |10⟩)/√2
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[0] = sqrt2_inv  # |00⟩
        expected[3] = sqrt2_inv  # |11⟩
        assert torch.allclose(state, expected, atol=1e-6)

    def test_ghz_state_creation(self):
        """Test 3-qubit GHZ state (|000⟩ + |111⟩)/√2."""
        state = qc.zero_state(n_qubits=3)  # |000⟩
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=3)
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=3)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=2, n_qubits=3)

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.zeros(8, dtype=torch.complex64)
        expected[0] = sqrt2_inv  # |000⟩
        expected[7] = sqrt2_inv  # |111⟩
        assert torch.allclose(state, expected, atol=1e-6)

    def test_global_phase_invariance(self):
        """Test that states are equivalent up to global phase."""
        state1 = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state1 = qc.apply_gate(state1, h_gate, qubit=0, n_qubits=1)

        # Apply a global phase
        state2 = state1 * torch.exp(1.0j * math.pi / 4.0)

        # Probabilities should be the same
        probs1 = qc.measure_probs(state1, n_qubits=1)
        probs2 = qc.measure_probs(state2, n_qubits=1)
        assert torch.allclose(probs1, probs2, atol=1e-6)


class TestMeasurementAndProbabilities:
    """Test measurement and probability computation."""

    def test_measure_probs_zero_state(self):
        """Test probabilities for |0⟩."""
        state = qc.zero_state(n_qubits=1)
        probs = qc.measure_probs(state, n_qubits=1)
        assert torch.allclose(probs[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(0.0), atol=1e-6)

    def test_measure_probs_plus_state(self):
        """Test probabilities for |+⟩ = (|0⟩ + |1⟩)/√2."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)
        probs = qc.measure_probs(state, n_qubits=1)
        assert torch.allclose(probs[0], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(0.5), atol=1e-6)

    def test_measure_probs_bell_state(self):
        """Test probabilities for Bell state |Φ+⟩."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)
        probs = qc.measure_probs(state, n_qubits=2)

        # Bell state has 0.5 probability for |00⟩ and |11⟩
        assert torch.allclose(probs[0], torch.tensor(0.5), atol=1e-6)  # |00⟩
        assert torch.allclose(probs[3], torch.tensor(0.5), atol=1e-6)  # |11⟩
        assert torch.allclose(probs[1], torch.tensor(0.0), atol=1e-6)  # |01⟩
        assert torch.allclose(probs[2], torch.tensor(0.0), atol=1e-6)  # |10⟩

    def test_manual_probability_computation(self):
        """Test probabilities match manual computation from amplitudes."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)
        probs = qc.measure_probs(state, n_qubits=1)

        # Manual computation
        manual_probs = torch.abs(state) ** 2
        assert torch.allclose(probs, manual_probs, atol=1e-6)


class TestExpectationValues:
    """Test expectation value computation."""

    def test_expectation_z_zero_state(self):
        """Test ⟨Z⟩ for |0⟩ is +1."""
        state = qc.zero_state(n_qubits=1)
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(1.0), atol=1e-6)

    def test_expectation_z_one_state(self):
        """Test ⟨Z⟩ for |1⟩ is -1."""
        state = qc.zero_state(n_qubits=1)
        x_gate = qc.X()
        state = qc.apply_gate(state, x_gate, qubit=0, n_qubits=1)  # |1⟩
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(-1.0), atol=1e-6)

    def test_expectation_z_plus_state(self):
        """Test ⟨Z⟩ for |+⟩ is 0."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)  # |+⟩
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(0.0), atol=1e-6)

    def test_expectation_z_minus_state(self):
        """Test ⟨Z⟩ for |-⟩ is 0."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        z_gate = qc.Z()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)  # |+⟩
        state = qc.apply_gate(state, z_gate, qubit=0, n_qubits=1)  # |-⟩
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(0.0), atol=1e-6)

    def test_expectation_z_bell_state(self):
        """Test ⟨Z⟩ for each qubit in Bell state."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)

        # In Bell state, each qubit individually has ⟨Z⟩ = 0
        z_exp_0 = qc.measure_expectation_z(state, qubit=0, n_qubits=2)
        z_exp_1 = qc.measure_expectation_z(state, qubit=1, n_qubits=2)
        assert torch.allclose(z_exp_0, torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(z_exp_1, torch.tensor(0.0), atol=1e-6)

    def test_expectation_z_via_matrix(self):
        """Test expectation value matches direct matrix computation."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)  # |+⟩

        # Direct computation: ⟨ψ|Z|ψ⟩
        z_gate = qc.Z()
        z_exp_direct = torch.real(torch.vdot(state, z_gate @ state))

        # Via measure_expectation_z
        z_exp_measured = qc.measure_expectation_z(state, qubit=0, n_qubits=1)

        assert torch.allclose(z_exp_direct, z_exp_measured, atol=1e-6)

    def test_expectation_zz_bell_state(self):
        """Test ⟨Z⊗Z⟩ for Bell state using PauliSum."""
        from qconduit.operators import PauliTerm, PauliSum
        from qconduit.operators.expectation import expectation_pauli_sum

        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)

        # ⟨Z⊗Z⟩ for Bell state should be -1
        zz_term = PauliTerm(coeff=1.0, paulis=("Z", "Z"))
        hamiltonian = PauliSum.from_terms([zz_term])
        zz_exp = expectation_pauli_sum(state, hamiltonian, n_qubits=2)

        assert torch.allclose(zz_exp, torch.tensor(-1.0), atol=1e-6)


class TestNormalizationDiagnostics:
    """Test normalization preservation and diagnostics."""

    def test_state_stays_normalized_simple_circuit(self):
        """Test state stays normalized through simple circuit."""
        state = qc.zero_state(n_qubits=1)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-6)

        # Apply random gates
        gates = [qc.H(), qc.X(), qc.Y(), qc.Z(), qc.RX(0.5), qc.RY(0.3), qc.RZ(0.7)]
        for gate in gates:
            state = qc.apply_gate(state, gate, qubit=0, n_qubits=1)
            assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)

    def test_state_stays_normalized_two_qubit_circuit(self):
        """Test state stays normalized through 2-qubit circuit."""
        state = qc.zero_state(n_qubits=2)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-6)

        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)

        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)

    def test_state_stays_normalized_three_qubit_circuit(self):
        """Test state stays normalized through 3-qubit circuit."""
        state = qc.zero_state(n_qubits=3)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-6)

        # Apply random gates
        h_gate = qc.H()
        for q in range(3):
            state = qc.apply_gate(state, h_gate, qubit=q, n_qubits=3)
            assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)

        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=3)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=2, n_qubits=3)
        assert torch.allclose(torch.norm(state), torch.tensor(1.0), atol=1e-5)

    def test_assert_normalized_helper(self):
        """Test assert_normalized helper function."""
        from qconduit.diagnostics import assert_normalized

        state = qc.zero_state(n_qubits=1)
        # Should not raise
        assert_normalized(state, atol=1e-6)

        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)
        assert_normalized(state, atol=1e-6)

    def test_probs_sum_to_one(self):
        """Test that probabilities always sum to 1."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        probs = qc.measure_probs(state, n_qubits=2)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

        # Apply more gates
        cnot = qc.CNOT(control_first=True)
        state = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)
        probs = qc.measure_probs(state, n_qubits=2)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


