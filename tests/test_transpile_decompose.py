"""Tests for gate decomposition functions."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.transpile import (
    decompose_h_to_rz_rx_rz,
    decompose_x_to_rx,
    decompose_y_to_ry,
    decompose_z_to_rz,
    decompose_rz_to_clifford_t,
    decompose_gate_to_basis,
)
from qconduit.circuit import QuantumCircuit
from qconduit.backend.statevector import zero_state, apply_gate
from qconduit.gates import H, X, Y, Z, RX, RY, RZ, S, T


def _compare_states_up_to_global_phase(
    state1: torch.Tensor,
    state2: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    """
    Compare two quantum states up to a global phase.

    Parameters
    ----------
    state1:
        First statevector.
    state2:
        Second statevector.
    atol:
        Absolute tolerance for comparison.
    """
    # Compute inner product
    inner = (state1.conj() * state2).sum()
    # Extract global phase
    if inner.abs() < 1e-10:
        # States are orthogonal or one is zero - compare directly
        assert torch.allclose(state1, state2, atol=atol)
    else:
        global_phase = inner / inner.abs()
        phased = state2 * global_phase.conj()
        assert torch.allclose(state1, phased, atol=atol)


class TestHDecomposition:
    """Tests for Hadamard gate decomposition."""

    def test_h_decomposition_unitary_equivalence(self):
        """Test that H decomposition is unitary-equivalent to H gate."""
        # Original circuit with H
        c1 = QuantumCircuit(n_qubits=1)
        c1.add_gate("H", [0])

        # Decomposed circuit
        c2 = QuantumCircuit(n_qubits=1)
        decompose_h_to_rz_rx_rz(c2, 0)

        # Simulate both from |0⟩
        state1 = c1.simulate_state()
        state2 = c2.simulate_state()

        # Compare up to global phase
        _compare_states_up_to_global_phase(state1, state2)

    def test_h_decomposition_manual_verification(self):
        """Test H decomposition by manually applying gates."""
        # Start from |0⟩
        state = zero_state(n_qubits=1)

        # Apply H directly
        h_gate = H()
        state_h = apply_gate(state.clone(), h_gate, qubit=0, n_qubits=1)

        # Apply decomposition: Rz(π/2) Rx(π/2) Rz(π/2)
        rz_gate = RZ(math.pi / 2.0)
        rx_gate = RX(math.pi / 2.0)
        state_decomp = apply_gate(state.clone(), rz_gate, qubit=0, n_qubits=1)
        state_decomp = apply_gate(state_decomp, rx_gate, qubit=0, n_qubits=1)
        state_decomp = apply_gate(state_decomp, rz_gate, qubit=0, n_qubits=1)

        # Compare up to global phase
        _compare_states_up_to_global_phase(state_h, state_decomp)


class TestXYZDecomposition:
    """Tests for X, Y, Z gate decompositions."""

    def test_x_decomposition(self):
        """Test X decomposition to Rx(π)."""
        c1 = QuantumCircuit(n_qubits=1)
        c1.add_gate("X", [0])

        c2 = QuantumCircuit(n_qubits=1)
        decompose_x_to_rx(c2, 0)

        state1 = c1.simulate_state()
        state2 = c2.simulate_state()

        _compare_states_up_to_global_phase(state1, state2)

    def test_y_decomposition(self):
        """Test Y decomposition to Ry(π)."""
        c1 = QuantumCircuit(n_qubits=1)
        c1.add_gate("Y", [0])

        c2 = QuantumCircuit(n_qubits=1)
        decompose_y_to_ry(c2, 0)

        state1 = c1.simulate_state()
        state2 = c2.simulate_state()

        _compare_states_up_to_global_phase(state1, state2)

    def test_z_decomposition(self):
        """Test Z decomposition to Rz(π)."""
        c1 = QuantumCircuit(n_qubits=1)
        c1.add_gate("Z", [0])

        c2 = QuantumCircuit(n_qubits=1)
        decompose_z_to_rz(c2, 0)

        state1 = c1.simulate_state()
        state2 = c2.simulate_state()

        _compare_states_up_to_global_phase(state1, state2)


class TestRzToCliffordT:
    """Tests for Rz to Clifford+T decomposition."""

    def test_rz_pi_over_4_decomposition(self):
        """Test Rz(π/4) decomposes to T gate."""
        theta = math.pi / 4.0

        c_orig = QuantumCircuit(n_qubits=1)
        c_orig.add_gate("RZ", [0], [theta])

        c_decomp = QuantumCircuit(n_qubits=1)
        result = decompose_rz_to_clifford_t(c_decomp, 0, theta)
        assert result is True

        state_orig = c_orig.simulate_state()
        state_decomp = c_decomp.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_rz_pi_over_2_decomposition(self):
        """Test Rz(π/2) decomposes to S gate."""
        theta = math.pi / 2.0

        c_orig = QuantumCircuit(n_qubits=1)
        c_orig.add_gate("RZ", [0], [theta])

        c_decomp = QuantumCircuit(n_qubits=1)
        result = decompose_rz_to_clifford_t(c_decomp, 0, theta)
        assert result is True

        state_orig = c_orig.simulate_state()
        state_decomp = c_decomp.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_rz_zero_decomposition(self):
        """Test Rz(0) decomposes to identity (no gates)."""
        theta = 0.0

        c_orig = QuantumCircuit(n_qubits=1)
        c_orig.add_gate("RZ", [0], [theta])

        c_decomp = QuantumCircuit(n_qubits=1)
        result = decompose_rz_to_clifford_t(c_decomp, 0, theta)
        assert result is True

        # Decomposition should have no gates (identity)
        assert len(c_decomp.ops) == 0

        state_orig = c_orig.simulate_state()
        state_decomp = c_decomp.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_rz_multiple_pi_over_4(self):
        """Test Rz for various multiples of π/4."""
        for k in [-2, -1, 0, 1, 2, 3, 4, 5]:
            theta = k * (math.pi / 4.0)

            c_orig = QuantumCircuit(n_qubits=1)
            c_orig.add_gate("RZ", [0], [theta])

            c_decomp = QuantumCircuit(n_qubits=1)
            result = decompose_rz_to_clifford_t(c_decomp, 0, theta)
            assert result is True

            state_orig = c_orig.simulate_state()
            state_decomp = c_decomp.simulate_state()

            _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_rz_non_multiple_returns_false(self):
        """Test that Rz with non-multiple of π/4 returns False."""
        theta = math.pi / 3.0  # Not a multiple of π/4

        c_decomp = QuantumCircuit(n_qubits=1)
        result = decompose_rz_to_clifford_t(c_decomp, 0, theta)
        assert result is False

        # Circuit should be unchanged
        assert len(c_decomp.ops) == 0


class TestDecomposeGateToBasis:
    """Tests for decompose_gate_to_basis function."""

    def test_decompose_h_to_rx_rz_basis(self):
        """Test decomposing H gate to {RX, RZ} basis."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RX", "RZ"])

        # Should only contain RX and RZ gates
        gate_names = {op.name for op in decomp.ops}
        assert gate_names.issubset({"RX", "RZ"})
        assert len(decomp.ops) == 3  # Rz, Rx, Rz

        # Verify unitary equivalence
        state_orig = circuit.simulate_state()
        state_decomp = decomp.simulate_state()
        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_decompose_x_to_rx_basis(self):
        """Test decomposing X gate to {RX} basis."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("X", [0])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RX"])

        assert len(decomp.ops) == 1
        assert decomp.ops[0].name == "RX"
        assert decomp.ops[0].params[0] == pytest.approx(math.pi)

        state_orig = circuit.simulate_state()
        state_decomp = decomp.simulate_state()
        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_decompose_gate_already_in_basis(self):
        """Test that gates already in basis are copied unchanged."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("RX", [0], [math.pi / 3.0])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RX", "RZ", "CNOT"])

        assert len(decomp.ops) == 1
        assert decomp.ops[0].name == "RX"
        assert decomp.ops[0].params[0] == pytest.approx(math.pi / 3.0)

    def test_decompose_rz_to_clifford_t_basis(self):
        """Test decomposing Rz(π/4) to Clifford+T basis."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("RZ", [0], [math.pi / 4.0])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["S", "T", "H", "CNOT"])

        # Should only contain S and T gates
        gate_names = {op.name for op in decomp.ops}
        assert gate_names.issubset({"S", "T"})

        state_orig = circuit.simulate_state()
        state_decomp = decomp.simulate_state()
        _compare_states_up_to_global_phase(state_orig, state_decomp)

    def test_decompose_unsupported_gate_raises(self):
        """Test that unsupported gate decomposition raises ValueError."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        # Try to decompose H into a basis that doesn't support it
        with pytest.raises(ValueError, match="Cannot decompose"):
            decompose_gate_to_basis(circuit, 0, target_basis=["CNOT"])

    def test_decompose_invalid_index_raises(self):
        """Test that invalid gate index raises IndexError."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        with pytest.raises(IndexError, match="out of range"):
            decompose_gate_to_basis(circuit, 1, target_basis=["RX", "RZ"])

    def test_decompose_cnot_in_basis(self):
        """Test that CNOT in basis is copied unchanged."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("CNOT", [0, 1])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RX", "RZ", "CNOT"])

        assert len(decomp.ops) == 1
        assert decomp.ops[0].name == "CNOT"
        assert decomp.ops[0].qubits == (0, 1)

    def test_decompose_y_to_ry_basis(self):
        """Test decomposing Y gate to {RY} basis."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("Y", [0])

        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RY"])

        assert len(decomp.ops) == 1
        assert decomp.ops[0].name == "RY"
        assert decomp.ops[0].params[0] == pytest.approx(math.pi)

        state_orig = circuit.simulate_state()
        state_decomp = decomp.simulate_state()
        _compare_states_up_to_global_phase(state_orig, state_decomp)



    def test_decompose_rz_in_basis_fallback(self):
        """Test that RZ in basis is used as fallback when decomposition fails."""
        circuit = QuantumCircuit(n_qubits=1)
        # Use a non-multiple of π/4
        circuit.add_gate("RZ", [0], [math.pi / 3.0])

        # Basis includes RZ, so it should be copied
        decomp = decompose_gate_to_basis(circuit, 0, target_basis=["RZ", "S", "T"])

        assert len(decomp.ops) == 1
        assert decomp.ops[0].name == "RZ"
        assert decomp.ops[0].params[0] == pytest.approx(math.pi / 3.0)

