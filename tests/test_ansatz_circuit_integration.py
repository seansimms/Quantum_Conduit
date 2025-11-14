"""Tests for ansatz-circuit integration."""

from __future__ import annotations

import pytest
import torch

from qconduit.layers.ansatzes import HardwareEfficientAnsatz


def test_hardware_efficient_ansatz_build_circuit_matches_forward() -> None:
    """Test that build_circuit produces a circuit that matches forward() output."""
    ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
    params = torch.randn(ansatz.num_parameters, dtype=torch.float32)

    state_direct = ansatz(params)
    circuit = ansatz.build_circuit(params=params)
    state_circuit = circuit.simulate_state()

    assert state_direct.shape == state_circuit.shape
    assert torch.allclose(state_direct, state_circuit, atol=1e-5, rtol=1e-5)


def test_build_circuit_default_params_zero_angles() -> None:
    """Test that build_circuit with None params defaults to zero angles."""
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    circuit = ansatz.build_circuit(params=None)

    # Simulate circuit and ansatz with zero params; they should match.
    params = torch.zeros(ansatz.num_parameters, dtype=torch.float32)
    state_direct = ansatz(params)
    state_circuit = circuit.simulate_state()

    assert torch.allclose(state_direct, state_circuit, atol=1e-5, rtol=1e-5)


def test_build_circuit_invalid_params_shape_raises() -> None:
    """Test that invalid params shape raises ValueError."""
    ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
    params = torch.zeros(3, dtype=torch.float32)  # wrong length

    with pytest.raises(ValueError, match="length equal to num_parameters"):
        _ = ansatz.build_circuit(params=params)


def test_build_circuit_multiple_qubits_and_layers() -> None:
    """Test build_circuit with larger circuits."""
    ansatz = HardwareEfficientAnsatz(n_qubits=3, depth=2)
    params = torch.randn(ansatz.num_parameters, dtype=torch.float32)

    state_direct = ansatz(params)
    circuit = ansatz.build_circuit(params=params)
    state_circuit = circuit.simulate_state()

    assert torch.allclose(state_direct, state_circuit, atol=1e-5, rtol=1e-5)

    # Verify circuit structure
    assert circuit.n_qubits == 3
    counts = circuit.gate_counts()
    assert counts["RX"] == 6  # 3 qubits * 2 layers
    assert counts["CNOT"] == 4  # 2 CNOTs per layer * 2 layers


def test_build_circuit_circuit_properties() -> None:
    """Test that built circuit has expected properties."""
    ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
    params = torch.randn(ansatz.num_parameters, dtype=torch.float32)
    circuit = ansatz.build_circuit(params=params)

    # Check gate counts
    counts = circuit.gate_counts()
    assert counts["RX"] == 4  # 2 qubits * 2 layers
    assert counts["CNOT"] == 2  # 1 CNOT per layer * 2 layers

    # Check depth (should be less than total gates due to parallelism)
    assert circuit.depth() > 0
    assert circuit.depth() <= len(circuit)

    # Check that all ops have correct structure
    for op in circuit.ops:
        assert op.name in ("RX", "CNOT")
        if op.name == "RX":
            assert len(op.qubits) == 1
            assert op.params is not None
            assert len(op.params) == 1
        elif op.name == "CNOT":
            assert len(op.qubits) == 2
            assert op.params is None


