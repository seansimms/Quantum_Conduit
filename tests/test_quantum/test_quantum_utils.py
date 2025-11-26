"""Regression tests for canonical quantum gate application utilities."""

import numpy as np
import pytest

from qconduit.quantum.gates import CNOT, H, X
from qconduit.quantum.utils import apply_single_qubit_gate, apply_two_qubit_gate


def initial_state(n: int) -> np.ndarray:
    """Return |0...0⟩ for n qubits."""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    return state


class TestSingleQubitGate:
    """Tests for apply_single_qubit_gate."""

    def test_x_on_zero(self):
        """X|0⟩ = |1⟩."""
        state = initial_state(1)
        result = apply_single_qubit_gate(state, X(), 0, 1)
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_allclose(result, expected)

    def test_h_on_zero(self):
        """H|0⟩ = (|0⟩ + |1⟩)/√2."""
        state = initial_state(1)
        result = apply_single_qubit_gate(state, H(), 0, 1)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(result, expected)

    def test_x_on_qubit0_2qubits(self):
        """X on qubit 0 of |00⟩ yields |01⟩ (index 1)."""
        state = initial_state(2)
        result = apply_single_qubit_gate(state, X(), 0, 2)
        expected = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_allclose(result, expected)

    def test_x_on_qubit1_2qubits(self):
        """X on qubit 1 of |00⟩ yields |10⟩ (index 2)."""
        state = initial_state(2)
        result = apply_single_qubit_gate(state, X(), 1, 2)
        expected = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_allclose(result, expected)

    def test_invalid_qubit_raises(self):
        """Invalid qubit index raises ValueError."""
        state = initial_state(2)
        with pytest.raises(ValueError):
            apply_single_qubit_gate(state, X(), 2, 2)


class TestTwoQubitGate:
    """Tests for apply_two_qubit_gate."""

    def test_cnot_control0_target1_from_01(self):
        """|01⟩ with CNOT(0,1) yields |11⟩."""
        state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        result = apply_two_qubit_gate(state, CNOT(), [0, 1], 2)
        expected = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        np.testing.assert_allclose(result, expected)

    def test_cnot_control0_target1_from_00(self):
        """|00⟩ with CNOT(0,1) stays |00⟩."""
        state = initial_state(2)
        result = apply_two_qubit_gate(state, CNOT(), [0, 1], 2)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_allclose(result, expected)

    def test_cnot_control1_target0_from_10(self):
        """|10⟩ with CNOT(1,0) yields |11⟩."""
        state = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        result = apply_two_qubit_gate(state, CNOT(), [1, 0], 2)
        expected = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        np.testing.assert_allclose(result, expected)

    def test_invalid_same_qubit_raises(self):
        """Same qubit for both raises ValueError."""
        state = initial_state(2)
        with pytest.raises(ValueError):
            apply_two_qubit_gate(state, CNOT(), [0, 0], 2)


class TestBellAndGHZ:
    """Regression tests for Bell and GHZ state preparation."""

    def test_bell_state(self):
        """H(q0) then CNOT(0,1) on |00⟩ yields (|00⟩ + |11⟩)/√2."""
        n_qubits = 2
        state = initial_state(n_qubits)
        state = apply_single_qubit_gate(state, H(), 0, n_qubits)
        state = apply_two_qubit_gate(state, CNOT(), [0, 1], n_qubits)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(state, expected)

    def test_ghz_state(self):
        """H(q0) then CNOT(0,1) then CNOT(0,2) on |000⟩ yields (|000⟩ + |111⟩)/√2."""
        n_qubits = 3
        state = initial_state(n_qubits)
        state = apply_single_qubit_gate(state, H(), 0, n_qubits)
        state = apply_two_qubit_gate(state, CNOT(), [0, 1], n_qubits)
        state = apply_two_qubit_gate(state, CNOT(), [0, 2], n_qubits)
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        np.testing.assert_allclose(state, expected)

    def test_ghz_chained(self):
        """H(q0) then CNOT(0,1) then CNOT(1,2) on |000⟩ yields (|000⟩ + |111⟩)/√2."""
        n_qubits = 3
        state = initial_state(n_qubits)
        state = apply_single_qubit_gate(state, H(), 0, n_qubits)
        state = apply_two_qubit_gate(state, CNOT(), [0, 1], n_qubits)
        state = apply_two_qubit_gate(state, CNOT(), [1, 2], n_qubits)
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        np.testing.assert_allclose(state, expected)


class TestDeterminism:
    """Ensure repeated calls produce identical results."""

    def test_bell_deterministic(self):
        """Bell state preparation is deterministic across runs."""
        n_qubits = 2
        results = []
        for _ in range(3):
            state = initial_state(n_qubits)
            state = apply_single_qubit_gate(state, H(), 0, n_qubits)
            state = apply_two_qubit_gate(state, CNOT(), [0, 1], n_qubits)
            results.append(state.copy())
        np.testing.assert_allclose(results[0], results[1])
        np.testing.assert_allclose(results[1], results[2])

