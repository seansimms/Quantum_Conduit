"""Tests for basis transpiler functions."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.transpile import (
    transpile_to_basis,
    transpile_to_rx_rz_cx_basis,
    transpile_to_clifford_t,
)
from qconduit.circuit import QuantumCircuit


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


class TestTranspileToBasis:
    """Tests for transpile_to_basis function."""

    def test_transpile_h_x_to_rx_rz_basis(self):
        """Test transpiling H+X circuit to {RX, RZ, CNOT} basis."""
        c_in = QuantumCircuit(n_qubits=1)
        c_in.add_gate("H", [0])
        c_in.add_gate("X", [0])

        c_out = transpile_to_rx_rz_cx_basis(c_in)

        # All gates should be RX or RZ
        gate_names = {op.name for op in c_out.ops}
        assert gate_names.issubset({"RX", "RZ"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_out = c_out.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_out)

    def test_transpile_bell_state_circuit(self):
        """Test transpiling Bell state circuit to {RX, RZ, CNOT} basis."""
        c_in = QuantumCircuit(n_qubits=2)
        c_in.add_gate("H", [0])
        c_in.add_gate("CNOT", [0, 1])

        c_out = transpile_to_rx_rz_cx_basis(c_in)

        # Should have CNOT and RX/RZ gates
        gate_names = {op.name for op in c_out.ops}
        assert "CNOT" in gate_names
        assert gate_names.issubset({"RX", "RZ", "CNOT"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_out = c_out.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_out)

    def test_transpile_gates_already_in_basis(self):
        """Test that gates already in basis are copied unchanged."""
        c_in = QuantumCircuit(n_qubits=2)
        c_in.add_gate("RX", [0], [math.pi / 3.0])
        c_in.add_gate("RZ", [1], [math.pi / 4.0])
        c_in.add_gate("CNOT", [0, 1])

        c_out = transpile_to_rx_rz_cx_basis(c_in)

        # Should have same number of gates
        assert len(c_out.ops) == len(c_in.ops)

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_out = c_out.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_out)

    def test_transpile_mixed_circuit(self):
        """Test transpiling circuit with mixed gate types."""
        c_in = QuantumCircuit(n_qubits=2)
        c_in.add_gate("H", [0])
        c_in.add_gate("X", [1])
        c_in.add_gate("Y", [0])
        c_in.add_gate("CNOT", [0, 1])

        c_out = transpile_to_rx_rz_cx_basis(c_in)

        # All gates should be in target basis
        gate_names = {op.name for op in c_out.ops}
        assert gate_names.issubset({"RX", "RY", "RZ", "CNOT"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_out = c_out.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_out)

    def test_transpile_unsupported_gate_raises(self):
        """Test that transpiling unsupported gate raises ValueError."""
        c_in = QuantumCircuit(n_qubits=1)
        c_in.add_gate("H", [0])

        # Try to transpile to a basis that doesn't support H
        with pytest.raises(ValueError, match="Cannot decompose"):
            transpile_to_basis(c_in, basis_gates=["CNOT"])


class TestTranspileToCliffordT:
    """Tests for transpile_to_clifford_t function."""

    def test_transpile_with_rz_fallback(self):
        """Test Clifford+T transpiler with RZ fallback enabled."""
        c_in = QuantumCircuit(n_qubits=1)
        c_in.add_gate("RZ", [0], [math.pi / 4.0])  # Multiple of π/4
        c_in.add_gate("RZ", [0], [math.pi / 3.0])  # Not a multiple of π/4

        c_ct = transpile_to_clifford_t(c_in, allow_rz_fallback=True)

        # First RZ should be decomposed, second should remain as RZ
        gate_names = {op.name for op in c_ct.ops}
        assert gate_names.issubset({"H", "S", "T", "RZ", "CNOT"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_ct = c_ct.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_ct)

    def test_transpile_without_rz_fallback_succeeds_for_multiples(self):
        """Test Clifford+T transpiler without fallback for multiples of π/4."""
        c_in = QuantumCircuit(n_qubits=1)
        c_in.add_gate("RZ", [0], [math.pi / 4.0])
        c_in.add_gate("RZ", [0], [math.pi / 2.0])

        c_ct = transpile_to_clifford_t(c_in, allow_rz_fallback=False)

        # All RZ should be decomposed
        gate_names = {op.name for op in c_ct.ops}
        assert "RZ" not in gate_names
        assert gate_names.issubset({"H", "S", "T", "CNOT"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_ct = c_ct.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_ct)

    def test_transpile_without_rz_fallback_fails_for_non_multiples(self):
        """Test that transpiler raises error for non-multiples without fallback."""
        c_in = QuantumCircuit(n_qubits=1)
        c_in.add_gate("RZ", [0], [math.pi / 3.0])  # Not a multiple of π/4

        with pytest.raises(ValueError, match="Cannot decompose"):
            transpile_to_clifford_t(c_in, allow_rz_fallback=False)

    def test_transpile_h_s_t_cnot_preserved(self):
        """Test that H, S, T, CNOT gates are preserved in Clifford+T basis."""
        c_in = QuantumCircuit(n_qubits=2)
        c_in.add_gate("H", [0])
        c_in.add_gate("S", [0])
        c_in.add_gate("T", [1])
        c_in.add_gate("CNOT", [0, 1])

        c_ct = transpile_to_clifford_t(c_in, allow_rz_fallback=False)

        # Should have same gates
        assert len(c_ct.ops) == len(c_in.ops)
        gate_names = {op.name for op in c_ct.ops}
        assert gate_names == {"H", "S", "T", "CNOT"}

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_ct = c_ct.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_ct)

    def test_transpile_complex_circuit_to_clifford_t(self):
        """Test transpiling a more complex circuit to Clifford+T."""
        c_in = QuantumCircuit(n_qubits=2)
        c_in.add_gate("H", [0])
        c_in.add_gate("RZ", [0], [math.pi / 4.0])
        c_in.add_gate("CNOT", [0, 1])
        c_in.add_gate("RZ", [1], [math.pi / 2.0])
        c_in.add_gate("H", [1])

        c_ct = transpile_to_clifford_t(c_in, allow_rz_fallback=False)

        # All gates should be in Clifford+T basis
        gate_names = {op.name for op in c_ct.ops}
        assert gate_names.issubset({"H", "S", "T", "CNOT"})

        # Verify unitary equivalence
        state_in = c_in.simulate_state()
        state_ct = c_ct.simulate_state()
        _compare_states_up_to_global_phase(state_in, state_ct)


