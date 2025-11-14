"""Tests for variational ansatz implementations."""

from __future__ import annotations

import pytest
import torch

from qconduit.operators import PauliSum, PauliTerm
from qconduit.variational import (
    HardwareEfficientAnsatz,
    LayeredEntanglerAnsatz,
    QAOAAnsatz,
)


class TestHardwareEfficientAnsatz:
    """Tests for HardwareEfficientAnsatz."""

    def test_parameter_count(self) -> None:
        """Test that parameter count is correct."""
        ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=2)
        assert ansatz.num_parameters == 3 * 2 * 2  # 12 parameters

    def test_validation_num_qubits(self) -> None:
        """Test that invalid num_qubits raises ValueError."""
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            HardwareEfficientAnsatz(num_qubits=0, num_layers=1)

    def test_validation_num_layers(self) -> None:
        """Test that invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            HardwareEfficientAnsatz(num_qubits=2, num_layers=0)

    def test_build_circuit_correct_length(self) -> None:
        """Test that build_circuit works with correct parameter length."""
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) > 0

    def test_build_circuit_wrong_length(self) -> None:
        """Test that wrong parameter length raises ValueError."""
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
        params = torch.zeros(5)  # Wrong length (should be 4)
        with pytest.raises(ValueError, match="params length"):
            ansatz.build_circuit(params)

    def test_build_circuit_wrong_dimension(self) -> None:
        """Test that wrong parameter dimension raises ValueError."""
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
        params = torch.zeros((2, 2))  # 2D instead of 1D
        with pytest.raises(ValueError, match="params must be 1D"):
            ansatz.build_circuit(params)

    def test_circuit_structure(self) -> None:
        """Test that circuit has expected structure."""
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        # Should have 2 qubits * 2 gates (RX, RZ) + 1 CNOT = 5 gates
        assert len(circuit.ops) == 5


class TestLayeredEntanglerAnsatz:
    """Tests for LayeredEntanglerAnsatz."""

    def test_parameter_count(self) -> None:
        """Test that parameter count is correct."""
        ansatz = LayeredEntanglerAnsatz(num_qubits=3, num_layers=2)
        assert ansatz.num_parameters == 3 * 2 * 2  # 12 parameters

    def test_validation_num_qubits(self) -> None:
        """Test that invalid num_qubits raises ValueError."""
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            LayeredEntanglerAnsatz(num_qubits=0, num_layers=1)

    def test_validation_num_layers(self) -> None:
        """Test that invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            LayeredEntanglerAnsatz(num_qubits=2, num_layers=0)

    def test_build_circuit_linear(self) -> None:
        """Test that build_circuit works with linear entanglement."""
        ansatz = LayeredEntanglerAnsatz(num_qubits=2, num_layers=1, ring_entanglement=False)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        assert circuit.n_qubits == 2

    def test_build_circuit_ring(self) -> None:
        """Test that build_circuit works with ring entanglement."""
        ansatz = LayeredEntanglerAnsatz(num_qubits=3, num_layers=1, ring_entanglement=True)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        assert circuit.n_qubits == 3
        # Should have more gates due to ring entanglement
        assert len(circuit.ops) > 0

    def test_build_circuit_wrong_length(self) -> None:
        """Test that wrong parameter length raises ValueError."""
        ansatz = LayeredEntanglerAnsatz(num_qubits=2, num_layers=1)
        params = torch.zeros(5)  # Wrong length
        with pytest.raises(ValueError, match="params length"):
            ansatz.build_circuit(params)


class TestQAOAAnsatz:
    """Tests for QAOAAnsatz."""

    def test_parameter_count(self) -> None:
        """Test that parameter count is correct."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = QAOAAnsatz(num_qubits=1, depth=3, cost_hamiltonian=hamiltonian)
        assert ansatz.num_parameters == 2 * 3  # 6 parameters

    def test_validation_num_qubits(self) -> None:
        """Test that invalid num_qubits raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            QAOAAnsatz(num_qubits=0, depth=1, cost_hamiltonian=hamiltonian)

    def test_validation_depth(self) -> None:
        """Test that invalid depth raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        with pytest.raises(ValueError, match="depth must be >= 1"):
            QAOAAnsatz(num_qubits=1, depth=0, cost_hamiltonian=hamiltonian)

    def test_validation_hamiltonian_qubits(self) -> None:
        """Test that mismatched Hamiltonian qubits raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z", "Z"))])
        with pytest.raises(ValueError, match="does not match num_qubits"):
            QAOAAnsatz(num_qubits=1, depth=1, cost_hamiltonian=hamiltonian)

    def test_build_circuit_correct_length(self) -> None:
        """Test that build_circuit works with correct parameter length."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = QAOAAnsatz(num_qubits=1, depth=1, cost_hamiltonian=hamiltonian)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        assert circuit.n_qubits == 1
        assert len(circuit.ops) > 0  # Should have H + cost + mixer gates

    def test_build_circuit_wrong_length(self) -> None:
        """Test that wrong parameter length raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = QAOAAnsatz(num_qubits=1, depth=1, cost_hamiltonian=hamiltonian)
        params = torch.zeros(3)  # Wrong length (should be 2)
        with pytest.raises(ValueError, match="params length"):
            ansatz.build_circuit(params)

    def test_build_circuit_structure(self) -> None:
        """Test that QAOA circuit has expected structure."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = QAOAAnsatz(num_qubits=1, depth=1, cost_hamiltonian=hamiltonian)
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)
        # Should start with H, then have cost and mixer gates
        assert circuit.ops[0].name == "H"


