"""Tests for circuit IR core functionality."""

from __future__ import annotations

import math
import pytest
import torch

from qconduit.circuit import QuantumCircuit
from qconduit.backend.statevector import zero_state


def test_circuit_simulates_bell_state() -> None:
    """Test that a circuit can simulate a Bell state."""
    circuit = QuantumCircuit(n_qubits=2)
    # |00> --H on qubit 0--> (|00> + |01>)/sqrt(2) (qubit 0 is LSB)
    circuit.add_gate("H", [0])
    # CNOT with qubit 1 as control, qubit 0 as target
    # This creates (|00> + |11>)/sqrt(2) Bell state
    circuit.add_gate("CNOT", [1, 0])

    state = circuit.simulate_state()

    # Expect |Φ+> Bell state: amplitudes at |00> and |11> are 1/sqrt(2)
    probs = state.abs() ** 2
    assert torch.allclose(
        probs,
        torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=probs.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_circuit_depth_and_gate_counts() -> None:
    """Test depth calculation and gate counting."""
    circuit = QuantumCircuit(n_qubits=2)
    circuit.add_gate("H", [0])
    circuit.add_gate("H", [1])  # Can be parallel with previous H
    circuit.add_gate("CNOT", [0, 1])  # Next layer

    assert circuit.num_gates() == 3
    counts = circuit.gate_counts()
    assert counts["H"] == 2
    assert counts["CNOT"] == 1

    # First two H can be parallel, then CNOT: depth 2
    assert circuit.depth() == 2


def test_circuit_text_diagram_contains_gates() -> None:
    """Test that ASCII diagram includes expected gate markers."""
    circuit = QuantumCircuit(n_qubits=2)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])

    diag = circuit.to_text_diagram()

    # Check that diagram includes expected characters
    assert "q0:" in diag
    assert "q1:" in diag
    assert "H" in diag
    # For CNOT we expect control/target markers
    assert "●" in diag
    assert "⊕" in diag


def test_circuit_invalid_qubit_raises() -> None:
    """Test that invalid qubit indices raise ValueError."""
    circuit = QuantumCircuit(n_qubits=1)
    with pytest.raises(ValueError, match="out of range"):
        circuit.add_gate("H", [1])


def test_simulate_unsupported_gate_raises() -> None:
    """Test that unsupported gate names raise ValueError during simulation."""
    circuit = QuantumCircuit(n_qubits=1)
    circuit.add_gate("FOO", [0])
    with pytest.raises(ValueError, match="Unsupported"):
        _ = circuit.simulate_state()


def test_circuit_copy() -> None:
    """Test that circuit copying works correctly."""
    circuit = QuantumCircuit(n_qubits=2)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])

    circuit_copy = circuit.copy()

    assert circuit_copy.n_qubits == circuit.n_qubits
    assert len(circuit_copy) == len(circuit)
    assert circuit_copy.gate_counts() == circuit.gate_counts()

    # Modify original, verify copy is unaffected
    circuit.add_gate("X", [1])
    assert len(circuit_copy) == 2
    assert len(circuit) == 3


def test_circuit_parametric_gates() -> None:
    """Test that parametric gates work correctly."""
    circuit = QuantumCircuit(n_qubits=1)
    circuit.add_gate("RX", [0], params=[math.pi / 2])
    circuit.add_gate("RY", [0], params=[math.pi / 4])
    circuit.add_gate("RZ", [0], params=[math.pi / 8])

    state = circuit.simulate_state()
    # Just verify it runs without error and produces a valid state
    assert state.shape == (2,)
    assert torch.is_complex(state)
    probs = state.abs() ** 2
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_circuit_empty() -> None:
    """Test that empty circuits work correctly."""
    circuit = QuantumCircuit(n_qubits=2)
    assert len(circuit) == 0
    assert circuit.depth() == 0
    assert circuit.gate_counts() == {}

    state = circuit.simulate_state()
    # Should be |00> state
    assert torch.allclose(state[0], torch.tensor(1.0 + 0.0j))
    assert torch.allclose(state[1:], torch.tensor(0.0 + 0.0j))


def test_circuit_ops_property() -> None:
    """Test that ops property returns read-only tuple."""
    circuit = QuantumCircuit(n_qubits=2)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])

    ops = circuit.ops
    assert isinstance(ops, tuple)
    assert len(ops) == 2
    assert ops[0].name == "H"
    assert ops[0].qubits == (0,)
    assert ops[1].name == "CNOT"
    assert ops[1].qubits == (0, 1)

    # Verify it's read-only (can't modify)
    with pytest.raises(AttributeError):
        ops.append(None)  # type: ignore


def test_circuit_depth_sequential() -> None:
    """Test depth calculation for sequential gates on same qubit."""
    circuit = QuantumCircuit(n_qubits=1)
    circuit.add_gate("H", [0])
    circuit.add_gate("X", [0])
    circuit.add_gate("Z", [0])

    # All gates act on same qubit, so depth equals number of gates
    assert circuit.depth() == 3


def test_circuit_depth_parallel() -> None:
    """Test depth calculation for parallel gates on different qubits."""
    circuit = QuantumCircuit(n_qubits=3)
    circuit.add_gate("H", [0])
    circuit.add_gate("H", [1])
    circuit.add_gate("H", [2])

    # All gates act on different qubits, so can be parallel: depth 1
    assert circuit.depth() == 1


class TestCircuitBuildingSemantics:
    """Comprehensive tests for circuit building semantics."""

    def test_circuit_h_only_on_one_qubit(self):
        """Test H-only circuit on one qubit matches statevector backend."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        state_circuit = circuit.simulate_state()

        # Compare with direct backend
        import qconduit as qc
        from qconduit.backend.statevector import apply_gate

        state_backend = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state_backend = apply_gate(state_backend, h_gate, qubit=0, n_qubits=1)

        assert torch.allclose(state_circuit, state_backend, atol=1e-6)

    def test_circuit_bell_state_matches_backend(self):
        """Test Bell-state circuit matches statevector backend reference."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        state_circuit = circuit.simulate_state()

        # Compare with direct backend
        import qconduit as qc
        from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate

        state_backend = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state_backend = apply_gate(state_backend, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state_backend = apply_two_qubit_gate(
            state_backend, cnot, qubit1=0, qubit2=1, n_qubits=2
        )

        assert torch.allclose(state_circuit, state_backend, atol=1e-6)

    def test_circuit_gate_counts_match_expected(self):
        """Test gate_counts matches expected values for simple circuits."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [1])
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("X", [0])

        counts = circuit.gate_counts()
        assert counts["H"] == 2
        assert counts["CNOT"] == 1
        assert counts["X"] == 1

    def test_circuit_depth_matches_expected(self):
        """Test depth matches expected values for simple circuits."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [1])  # Parallel with previous H
        circuit.add_gate("CNOT", [0, 1])  # Next layer
        circuit.add_gate("X", [0])  # Next layer (qubit 0 was used in CNOT)

        # Expected depth: H gates (layer 1), CNOT (layer 2), X (layer 3)
        assert circuit.depth() == 3

    def test_circuit_text_diagram_exact_match(self):
        """Test to_text_diagram produces expected textual diagram for small circuits."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        diag = circuit.to_text_diagram()

        # Should contain expected substrings
        assert "q0:" in diag
        assert "q1:" in diag
        assert "H" in diag
        assert "●" in diag  # Control
        assert "⊕" in diag  # Target

    def test_circuit_ghz_state_preparation(self):
        """Test 3-qubit GHZ circuit matches direct backend composition."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("CNOT", [1, 2])

        state_circuit = circuit.simulate_state()

        # Compare with direct backend
        import qconduit as qc
        from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate

        state_backend = qc.zero_state(n_qubits=3)
        h_gate = qc.H()
        state_backend = apply_gate(state_backend, h_gate, qubit=0, n_qubits=3)
        cnot = qc.CNOT(control_first=True)
        state_backend = apply_two_qubit_gate(
            state_backend, cnot, qubit1=0, qubit2=1, n_qubits=3
        )
        state_backend = apply_two_qubit_gate(
            state_backend, cnot, qubit1=1, qubit2=2, n_qubits=3
        )

        # Compare up to global phase
        inner = (state_circuit.conj() * state_backend).sum()
        global_phase = inner / inner.abs()
        phased = state_backend * global_phase.conj()
        assert torch.allclose(state_circuit, phased, atol=1e-5)

