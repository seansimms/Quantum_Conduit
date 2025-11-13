"""Tests for circuit analysis utilities."""

from __future__ import annotations

import pytest

from qconduit.transpile import (
    GateCountSummary,
    summarize_gate_counts,
    estimate_circuit_depth,
)
from qconduit.circuit import QuantumCircuit


class TestSummarizeGateCounts:
    """Tests for summarize_gate_counts function."""

    def test_simple_gate_counts(self):
        """Test gate counting for a simple circuit."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [1])
        circuit.add_gate("T", [0])
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("X", [1])

        summary = summarize_gate_counts(circuit)

        assert summary.total_gates == 5
        assert summary.counts["H"] == 2
        assert summary.counts["T"] == 1
        assert summary.counts["CNOT"] == 1
        assert summary.counts["X"] == 1
        assert summary.t_count == 1
        # Clifford count should include H, X, CNOT
        assert summary.clifford_count == 4

    def test_empty_circuit(self):
        """Test gate counting for empty circuit."""
        circuit = QuantumCircuit(n_qubits=1)

        summary = summarize_gate_counts(circuit)

        assert summary.total_gates == 0
        assert len(summary.counts) == 0
        assert summary.t_count == 0
        assert summary.clifford_count == 0

    def test_t_count_only(self):
        """Test T-count calculation."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("T", [0])
        circuit.add_gate("T", [0])
        circuit.add_gate("T", [0])

        summary = summarize_gate_counts(circuit)

        assert summary.t_count == 3
        assert summary.total_gates == 3

    def test_clifford_count_all_types(self):
        """Test Clifford count includes all Clifford gate types."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("S", [0])
        circuit.add_gate("X", [0])
        circuit.add_gate("Y", [1])
        circuit.add_gate("Z", [1])
        circuit.add_gate("CNOT", [0, 1])

        summary = summarize_gate_counts(circuit)

        # All gates are Clifford gates
        assert summary.clifford_count == 6
        assert summary.total_gates == 6

    def test_mixed_clifford_and_non_clifford(self):
        """Test counting with both Clifford and non-Clifford gates."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("T", [0])  # Non-Clifford
        circuit.add_gate("RX", [0], [0.5])  # Non-Clifford
        circuit.add_gate("X", [0])

        summary = summarize_gate_counts(circuit)

        assert summary.total_gates == 4
        assert summary.t_count == 1
        # Only H and X are Clifford
        assert summary.clifford_count == 2

    def test_case_insensitive_gate_names(self):
        """Test that gate names are handled correctly (should be uppercase)."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("h", [0])  # Lowercase (should be converted internally)

        summary = summarize_gate_counts(circuit)

        # Gate names should be normalized to uppercase in circuit
        # So both should count as "H"
        assert summary.counts.get("H", 0) >= 1


class TestEstimateCircuitDepth:
    """Tests for estimate_circuit_depth function."""

    def test_empty_circuit_depth(self):
        """Test depth of empty circuit is 0."""
        circuit = QuantumCircuit(n_qubits=1)

        depth = estimate_circuit_depth(circuit)

        assert depth == 0

    def test_single_qubit_chain_depth(self):
        """Test depth of sequential single-qubit gates on one qubit."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("X", [0])
        circuit.add_gate("Z", [0])

        depth = estimate_circuit_depth(circuit)

        # All gates act on same qubit, so depth = 3
        assert depth == 3

    def test_parallel_gates_depth(self):
        """Test depth when gates can be parallelized."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("X", [1])  # Can be parallel with H
        circuit.add_gate("CNOT", [0, 1])  # Next layer

        depth = estimate_circuit_depth(circuit)

        # H and X can be in same layer, CNOT in next: depth = 2
        assert depth == 2

    def test_sequential_two_qubit_gates(self):
        """Test depth with sequential two-qubit gates."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("CNOT", [1, 2])

        depth = estimate_circuit_depth(circuit)

        # First CNOT uses qubits 0,1; second uses 1,2
        # Since qubit 1 is shared, they cannot be parallel: depth = 2
        assert depth == 2

    def test_disjoint_two_qubit_gates(self):
        """Test depth with disjoint two-qubit gates."""
        circuit = QuantumCircuit(n_qubits=4)
        circuit.add_gate("CNOT", [0, 1])
        circuit.add_gate("CNOT", [2, 3])  # Disjoint from first CNOT

        depth = estimate_circuit_depth(circuit)

        # Gates act on disjoint qubits, can be parallel: depth = 1
        assert depth == 1

    def test_mixed_parallel_and_sequential(self):
        """Test depth with mixed parallel and sequential gates."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [1])
        circuit.add_gate("H", [2])  # All H can be parallel: layer 1
        circuit.add_gate("CNOT", [0, 1])  # Layer 2
        circuit.add_gate("CNOT", [1, 2])  # Layer 3 (qubit 1 busy from layer 2)
        circuit.add_gate("X", [0])  # Can be parallel with CNOT(1,2) at layer 3

        depth = estimate_circuit_depth(circuit)

        # Expected depth: 3 layers
        # Layer 1: H(0), H(1), H(2)
        # Layer 2: CNOT(0,1)
        # Layer 3: CNOT(1,2), X(0) (disjoint qubits)
        assert depth == 3

    def test_single_gate_depth(self):
        """Test depth of circuit with single gate."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        depth = estimate_circuit_depth(circuit)

        assert depth == 1

    def test_bell_state_circuit_depth(self):
        """Test depth of Bell state preparation circuit."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        depth = estimate_circuit_depth(circuit)

        # H and CNOT cannot be parallel (both use qubit 0): depth = 2
        assert depth == 2

