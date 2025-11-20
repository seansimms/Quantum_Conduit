"""Tests for batched circuit application utilities."""

import pytest
import torch

from qconduit.backend.statevector import zero_state
from qconduit.batched import (
    BatchedState,
    apply_ansatz_batch_to_state,
    apply_circuit_to_batched_states,
    apply_circuits_batch_to_states,
    batched_build_circuits_from_params,
)
from qconduit.circuit import QuantumCircuit
from qconduit.variational import HardwareEfficientAnsatz
from qconduit.variational.vqe import _apply_circuit_to_statevector


def test_apply_circuit_to_batched_states_h_gate():
    """Test applying H gate to batch of states."""
    # Create H gate circuit
    circuit = QuantumCircuit(n_qubits=1)
    circuit.add_gate("H", [0])

    # Create batch of states: |0>, |1>, |+>, |->
    B = 4
    states = torch.zeros(B, 2, dtype=torch.complex128)
    states[0, 0] = 1.0  # |0>
    states[1, 1] = 1.0  # |1>
    states[2, 0] = 1.0 / torch.sqrt(torch.tensor(2.0))
    states[2, 1] = 1.0 / torch.sqrt(torch.tensor(2.0))  # |+>
    states[3, 0] = 1.0 / torch.sqrt(torch.tensor(2.0))
    states[3, 1] = -1.0 / torch.sqrt(torch.tensor(2.0))  # |->
    batched = BatchedState(states, n_qubits=1)

    # Apply circuit
    result = apply_circuit_to_batched_states(circuit, batched)

    # Verify each row matches single-run result
    for i in range(B):
        psi_single = _apply_circuit_to_statevector(circuit, states[i], device=None)
        assert torch.allclose(result.states[i], psi_single, atol=1e-10)


def test_apply_circuit_to_batched_states_mismatched_n_qubits():
    """Test that mismatched n_qubits raises ValueError."""
    circuit = QuantumCircuit(n_qubits=2)
    states = torch.randn(3, 2, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=1)

    with pytest.raises(ValueError, match="does not match"):
        apply_circuit_to_batched_states(circuit, batched)


def test_apply_ansatz_batch_to_state():
    """Test applying ansatz batch to initial state."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    B = 5
    params_batch = torch.randn(B, ansatz.num_parameters)

    # Initial state |0>
    psi0 = zero_state(n_qubits=1, batch_shape=None)

    # Apply batch
    result = apply_ansatz_batch_to_state(
        lambda p: ansatz.build_circuit(p), params_batch, psi0
    )

    assert result.states.shape == (B, 2)
    assert result.n_qubits == 1

    # Verify each row matches single-run result
    for i in range(B):
        circuit = ansatz.build_circuit(params_batch[i])
        psi_single = _apply_circuit_to_statevector(circuit, psi0, device=None)
        assert torch.allclose(result.states[i], psi_single, atol=1e-8)


def test_apply_ansatz_batch_to_state_wrong_params_shape():
    """Test that wrong params shape raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    params_batch = torch.randn(5)  # 1D instead of 2D
    psi0 = zero_state(n_qubits=1, batch_shape=None)

    with pytest.raises(ValueError, match="must be 2D"):
        apply_ansatz_batch_to_state(
            lambda p: ansatz.build_circuit(p), params_batch, psi0
        )


def test_batched_build_circuits_from_params():
    """Test building circuits from parameter batch."""
    ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
    B = 3
    params_batch = torch.randn(B, ansatz.num_parameters)

    circuits = batched_build_circuits_from_params(ansatz, params_batch)

    assert len(circuits) == B
    for i, circuit in enumerate(circuits):
        assert circuit.n_qubits == ansatz.num_qubits
        # Verify circuit was built with correct params
        expected_circuit = ansatz.build_circuit(params_batch[i])
        assert len(circuit.ops) == len(expected_circuit.ops)


def test_batched_build_circuits_from_params_wrong_shape():
    """Test that wrong params shape raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    params_batch = torch.randn(5)  # 1D

    with pytest.raises(ValueError, match="must be 2D"):
        batched_build_circuits_from_params(ansatz, params_batch)


def test_apply_circuits_batch_to_states():
    """Test applying batch of circuits to batch of states."""
    B = 3
    circuits = []
    for _ in range(B):
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])
        circuits.append(circuit)

    states = torch.randn(B, 2, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    result = apply_circuits_batch_to_states(circuits, states)

    assert result.states.shape == (B, 2)
    assert result.n_qubits == 1

    # Verify each row matches single-run result
    for i in range(B):
        psi_single = _apply_circuit_to_statevector(circuits[i], states[i], device=None)
        assert torch.allclose(result.states[i], psi_single, atol=1e-10)


def test_apply_circuits_batch_to_states_broadcast():
    """Test broadcasting single state to all circuits."""
    B = 3
    circuits = []
    for _ in range(B):
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])
        circuits.append(circuit)

    # Single state
    state = torch.tensor([1.0, 0.0], dtype=torch.complex128)

    result = apply_circuits_batch_to_states(circuits, state)

    assert result.states.shape == (B, 2)
    # All results should be the same (same circuit, same input)
    for i in range(1, B):
        assert torch.allclose(result.states[0], result.states[i], atol=1e-10)


def test_apply_circuits_batch_to_states_mismatched_length():
    """Test that mismatched batch lengths raise ValueError."""
    circuits = [QuantumCircuit(n_qubits=1) for _ in range(3)]
    states = torch.randn(5, 2, dtype=torch.complex128)  # Wrong batch size
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    with pytest.raises(ValueError, match="does not match"):
        apply_circuits_batch_to_states(circuits, states)


def test_apply_circuits_batch_to_states_mismatched_n_qubits():
    """Test that mismatched n_qubits raises ValueError."""
    circuits = [QuantumCircuit(n_qubits=2) for _ in range(2)]
    states = torch.randn(2, 2, dtype=torch.complex128)  # 1 qubit states
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    with pytest.raises(ValueError, match="n_qubits"):
        apply_circuits_batch_to_states(circuits, states)


def test_apply_ansatz_batch_to_state_non_power_of_two():
    """Test that non-power-of-two initial_state raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    params_batch = torch.randn(3, ansatz.num_parameters)
    initial_state = torch.randn(3, dtype=torch.complex128)  # Not a power of 2

    with pytest.raises(ValueError, match="not a power of 2"):
        apply_ansatz_batch_to_state(
            lambda p: ansatz.build_circuit(p), params_batch, initial_state
        )


def test_apply_ansatz_batch_to_state_mismatched_circuit_n_qubits():
    """Test that mismatched circuit n_qubits raises ValueError."""
    # Use large batch to force loop path (not vectorized)
    params_batch = torch.randn(1000, 2)  # Large batch
    initial_state = torch.randn(4, dtype=torch.complex128)  # 2 qubits
    initial_state = initial_state / torch.linalg.norm(initial_state)

    # Create a factory that produces wrong n_qubits circuit
    def wrong_factory(p):
        circuit = QuantumCircuit(n_qubits=1)  # Wrong n_qubits (should be 2)
        return circuit

    with pytest.raises(ValueError, match="n_qubits"):
        apply_ansatz_batch_to_state(wrong_factory, params_batch, initial_state)


def test_apply_circuits_batch_to_states_empty():
    """Test that empty circuits list raises ValueError."""
    circuits = []
    states = torch.randn(2, dtype=torch.complex128)

    with pytest.raises(ValueError, match="non-empty"):
        apply_circuits_batch_to_states(circuits, states)


def test_apply_circuits_batch_to_states_wrong_ndim():
    """Test that wrong states ndim raises ValueError."""
    circuits = [QuantumCircuit(n_qubits=1) for _ in range(2)]
    states = torch.randn(2, 2, 2, dtype=torch.complex128)  # 3D

    with pytest.raises(ValueError, match="must be 1D"):
        apply_circuits_batch_to_states(circuits, states)

