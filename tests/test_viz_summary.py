"""Tests for circuit summary and comparison utilities."""

import io
import math

from qconduit.circuit import QuantumCircuit
from qconduit.viz.summary import (
    circuit_summary,
    compare_circuits,
    print_circuit_summary,
)


def test_circuit_summary_empty():
    """Test summary of empty circuit."""
    circ = QuantumCircuit(3)
    summary = circuit_summary(circ)

    assert summary["n_qubits"] == 3
    assert summary["n_gates"] == 0
    assert summary["estimated_depth"] == 0
    assert summary["t_count"] == 0
    assert summary["clifford_count"] == 0
    assert summary["params_count"] == 0
    assert summary["uses_param_gates"] is False
    assert summary["gate_counts"] == {}


def test_circuit_summary_basic():
    """Test summary of a basic circuit."""
    circ = QuantumCircuit(2)
    circ.add_gate("H", [0])
    circ.add_gate("X", [1])
    circ.add_gate("CNOT", [0, 1])

    summary = circuit_summary(circ)

    assert summary["n_qubits"] == 2
    assert summary["n_gates"] == 3
    assert summary["estimated_depth"] > 0
    assert "H" in summary["gate_counts"]
    assert "X" in summary["gate_counts"]
    assert "CNOT" in summary["gate_counts"]
    assert summary["gate_counts"]["H"] == 1
    assert summary["gate_counts"]["X"] == 1
    assert summary["gate_counts"]["CNOT"] == 1


def test_circuit_summary_parametric():
    """Test summary with parametric gates."""
    circ = QuantumCircuit(1)
    circ.add_gate("RX", [0], [math.pi / 2])
    circ.add_gate("RZ", [0], [math.pi / 4])

    summary = circuit_summary(circ)

    assert summary["params_count"] == 2
    assert summary["uses_param_gates"] is True
    assert "RX" in summary["gate_counts"]
    assert "RZ" in summary["gate_counts"]


def test_circuit_summary_t_count():
    """Test T-count computation."""
    circ = QuantumCircuit(2)
    circ.add_gate("T", [0])
    circ.add_gate("T", [1])
    circ.add_gate("H", [0])

    summary = circuit_summary(circ)

    assert summary["t_count"] == 2
    assert summary["clifford_count"] >= 1  # H is Clifford


def test_print_circuit_summary():
    """Test pretty-printing of circuit summary."""
    circ = QuantumCircuit(2)
    circ.add_gate("H", [0])
    circ.add_gate("X", [1])

    f = io.StringIO()
    print_circuit_summary(circ, file=f)
    output = f.getvalue()

    assert "Circuit Summary" in output
    assert "Qubits:" in output
    assert "Total Gates:" in output
    assert "Gate Counts:" in output


def test_compare_circuits_identical():
    """Test comparison of identical circuits."""
    circ1 = QuantumCircuit(2)
    circ1.add_gate("H", [0])
    circ1.add_gate("X", [1])

    circ2 = QuantumCircuit(2)
    circ2.add_gate("H", [0])
    circ2.add_gate("X", [1])

    result = compare_circuits(circ1, circ2)

    assert result["same_unitary"] is True
    assert all(v == 0 for v in result["delta_gate_counts"].values())
    assert result["depth_diff"] == 0
    assert result["param_diff"] == 0


def test_compare_circuits_different():
    """Test comparison of different circuits."""
    circ1 = QuantumCircuit(2)
    circ1.add_gate("H", [0])

    circ2 = QuantumCircuit(2)
    circ2.add_gate("X", [0])
    circ2.add_gate("H", [1])

    result = compare_circuits(circ1, circ2)

    assert result["same_unitary"] is False
    # delta_gate_counts only includes non-zero differences
    # H appears once in each, so delta is 0 and won't be in dict
    assert result["delta_gate_counts"].get("H", 0) == 0
    assert result["delta_gate_counts"]["X"] == 1  # One X in circ2


def test_compare_circuits_large():
    """Test comparison of large circuits (unitary not computed)."""
    circ1 = QuantumCircuit(5)  # 2^5 = 32 > 16, so unitary not computed
    circ1.add_gate("H", [0])

    circ2 = QuantumCircuit(5)
    circ2.add_gate("X", [0])

    result = compare_circuits(circ1, circ2, max_dim_for_unitary=16)

    assert result["same_unitary"] is None  # Not computed for large circuits
    assert result["depth_diff"] is not None
    assert result["param_diff"] == 0


def test_compare_circuits_parametric():
    """Test comparison of circuits with parameters."""
    circ1 = QuantumCircuit(1)
    circ1.add_gate("RX", [0], [math.pi / 2])

    circ2 = QuantumCircuit(1)
    circ2.add_gate("RX", [0], [math.pi / 2])
    circ2.add_gate("RY", [0], [math.pi / 4])

    result = compare_circuits(circ1, circ2)

    assert result["param_diff"] == 1  # circ2 has one more parameter


def test_compare_circuits_different_qubits():
    """Test comparison of circuits with different qubit counts."""
    circ1 = QuantumCircuit(2)
    circ1.add_gate("H", [0])

    circ2 = QuantumCircuit(3)
    circ2.add_gate("H", [0])

    result = compare_circuits(circ1, circ2)

    # Unitary comparison should be None (different dimensions)
    assert result["same_unitary"] is None


def test_circuit_summary_depth():
    """Test that depth is computed correctly."""
    circ = QuantumCircuit(3)
    # Add gates that can be parallelized
    circ.add_gate("H", [0])
    circ.add_gate("H", [1])  # Can be parallel with H on 0
    circ.add_gate("CNOT", [0, 1])  # Requires both qubits

    summary = circuit_summary(circ)

    assert summary["estimated_depth"] >= 2
    assert summary["estimated_depth"] <= 3  # At most 3 layers


def test_circuit_summary_clifford_count():
    """Test Clifford count computation."""
    circ = QuantumCircuit(2)
    circ.add_gate("H", [0])
    circ.add_gate("S", [1])
    circ.add_gate("X", [0])
    circ.add_gate("CNOT", [0, 1])
    circ.add_gate("T", [0])  # Not Clifford

    summary = circuit_summary(circ)

    assert summary["clifford_count"] == 4  # H, S, X, CNOT
    assert summary["t_count"] == 1

