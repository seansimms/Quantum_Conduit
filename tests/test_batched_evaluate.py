"""Tests for batched expectation evaluation utilities."""

import pytest
import torch

from qconduit.batched import (
    BatchedState,
    evaluate_expectations_batched_via_states,
    evaluate_expectations_for_params_batched,
)
from qconduit.operators import PauliSum, PauliTerm
from qconduit.variational import HardwareEfficientAnsatz
from qconduit.variational.vqe import evaluate_expectation_value


def test_evaluate_expectations_batched_via_states_single_qubit_z():
    """Test batched expectation evaluation with single-qubit Z operator."""
    # Hamiltonian: Z
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    # Create batch of states
    B = 5
    states = torch.randn(B, 2, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=1)

    # Evaluate expectations
    expectations = evaluate_expectations_batched_via_states(H, batched)

    assert expectations.shape == (B,)
    assert expectations.dtype == torch.float64

    # Verify each row matches single-run result
    from qconduit.operators.expectation import expectation_pauli_sum

    for i in range(B):
        exp_single = expectation_pauli_sum(batched.states[i], H)
        if exp_single.ndim == 0:
            exp_val = exp_single.item()
        else:
            exp_val = exp_single.item()
        assert abs(expectations[i].item() - exp_val) < 1e-10


def test_evaluate_expectations_batched_via_states_two_qubit():
    """Test batched expectation evaluation with two-qubit Hamiltonian."""
    # Hamiltonian: ZZ
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])

    # Create batch of states
    B = 4
    states = torch.randn(B, 4, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=2)

    # Evaluate expectations
    expectations = evaluate_expectations_batched_via_states(H, batched)

    assert expectations.shape == (B,)
    assert expectations.dtype == torch.float64

    # Verify each row matches single-run result
    from qconduit.operators.expectation import expectation_pauli_sum

    for i in range(B):
        exp_single = expectation_pauli_sum(batched.states[i], H)
        if exp_single.ndim == 0:
            exp_val = exp_single.item()
        else:
            exp_val = exp_single.item()
        assert abs(expectations[i].item() - exp_val) < 1e-10


def test_evaluate_expectations_batched_via_states_mismatched_n_qubits():
    """Test that mismatched n_qubits raises ValueError."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])  # 2 qubits
    states = torch.randn(3, 2, dtype=torch.complex128)  # 1 qubit
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=1)

    with pytest.raises(ValueError, match="does not match"):
        evaluate_expectations_batched_via_states(H, batched)


def test_evaluate_expectations_for_params_batched():
    """Test batched expectation evaluation for parameter batch."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    B = 10
    params_batch = torch.randn(B, ansatz.num_parameters)

    # Evaluate expectations
    energies = evaluate_expectations_for_params_batched(ansatz, params_batch, H)

    assert energies.shape == (B,)
    assert energies.dtype == torch.float64

    # Verify each row matches single-run result
    for i in range(B):
        energy_single = evaluate_expectation_value(
            ansatz, params_batch[i], H, initial_state=None
        )
        assert abs(energies[i].item() - energy_single) < 1e-8


def test_evaluate_expectations_for_params_batched_with_initial_state():
    """Test batched evaluation with custom initial state."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    B = 5
    params_batch = torch.randn(B, ansatz.num_parameters)

    # Custom initial state |+>
    psi0 = torch.tensor([1.0, 1.0], dtype=torch.complex128) / torch.sqrt(torch.tensor(2.0))

    # Evaluate expectations
    energies = evaluate_expectations_for_params_batched(
        ansatz, params_batch, H, initial_state=psi0
    )

    assert energies.shape == (B,)

    # Verify each row matches single-run result
    for i in range(B):
        energy_single = evaluate_expectation_value(
            ansatz, params_batch[i], H, initial_state=psi0
        )
        assert abs(energies[i].item() - energy_single) < 1e-8


def test_evaluate_expectations_for_params_batched_wrong_params_shape():
    """Test that wrong params shape raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    params_batch = torch.randn(5)  # 1D instead of 2D

    with pytest.raises(ValueError, match="must be 2D"):
        evaluate_expectations_for_params_batched(ansatz, params_batch, H)


def test_evaluate_expectations_for_params_batched_wrong_num_params():
    """Test that wrong number of parameters raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    params_batch = torch.randn(5, ansatz.num_parameters + 1)  # Wrong number

    with pytest.raises(ValueError, match="parameters per row"):
        evaluate_expectations_for_params_batched(ansatz, params_batch, H)


def test_evaluate_expectations_for_params_batched_mismatched_hamiltonian():
    """Test that mismatched Hamiltonian n_qubits raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])  # 2 qubits, but ansatz is 1 qubit
    params_batch = torch.randn(5, ansatz.num_parameters)

    with pytest.raises(ValueError, match="does not match"):
        evaluate_expectations_for_params_batched(ansatz, params_batch, H)


def test_evaluate_expectations_for_params_batched_zero_initial_state():
    """Test that zero initial state raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    params_batch = torch.randn(5, ansatz.num_parameters)
    psi0 = torch.zeros(2, dtype=torch.complex128)

    with pytest.raises(ValueError, match="zero norm"):
        evaluate_expectations_for_params_batched(
            ansatz, params_batch, H, initial_state=psi0
        )


def test_evaluate_expectations_for_params_batched_wrong_initial_state_shape():
    """Test that wrong initial_state shape raises ValueError."""
    ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    params_batch = torch.randn(5, ansatz.num_parameters)
    psi0 = torch.randn(2, 2, dtype=torch.complex128)  # 2D instead of 1D

    with pytest.raises(ValueError, match="must be 1D"):
        evaluate_expectations_for_params_batched(
            ansatz, params_batch, H, initial_state=psi0
        )

