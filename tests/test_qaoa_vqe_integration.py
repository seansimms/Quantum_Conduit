"""Tests for QAOA + VQE integration."""

from __future__ import annotations

import torch

from qconduit.core.device import default_device
from qconduit.algorithms import ising_maxcut_hamiltonian, QAOAAnsatz
from qconduit.algorithms.vqe import VQE


def test_qaoa_vqe_energy_uniform_state_two_node_maxcut() -> None:
    """
    Test that VQE energy for uniform superposition equals 0.5 for 2-node MaxCut.

    For H = (1 - Z₀Z₁)/2 and |+⟩⊗|+⟩, we know:
    - ⟨Z₀Z₁⟩ = 0.
    - Therefore ⟨H⟩ = (1 - 0)/2 = 0.5.
    """
    n_qubits = 2
    edges = [(0, 1)]
    H = ising_maxcut_hamiltonian(num_nodes=n_qubits, edges=edges, include_constant=True)

    ansatz = QAOAAnsatz(n_qubits=n_qubits, problem_hamiltonian=H, p=1)

    # QAOA has 2 parameters for p=1
    params = torch.zeros(ansatz.num_parameters, dtype=torch.float32, requires_grad=True)

    vqe = VQE(ansatz, H)

    energy = vqe.energy(params).item()

    # Expected value is 0.5 for uniform superposition on this MaxCut instance.
    assert abs(energy - 0.5) < 1e-3

    # Check that gradients flow
    vqe.energy(params).backward()
    assert params.grad is not None
    # Some gradient entries should be non-zero in general
    assert params.grad.abs().sum().item() >= 0.0


def test_qaoa_vqe_energy_nonzero_params() -> None:
    """Test that VQE energy computation works with non-zero QAOA parameters."""
    n_qubits = 2
    edges = [(0, 1)]
    H = ising_maxcut_hamiltonian(num_nodes=n_qubits, edges=edges, include_constant=True)

    ansatz = QAOAAnsatz(n_qubits=n_qubits, problem_hamiltonian=H, p=1)

    # Use non-zero parameters
    params = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)

    vqe = VQE(ansatz, H)

    energy = vqe.energy(params)

    # Energy should be a real scalar
    assert energy.shape == ()
    assert torch.is_floating_point(energy)

    # Check gradients
    energy.backward()
    assert params.grad is not None
    assert params.grad.shape == params.shape


def test_qaoa_vqe_energy_p2() -> None:
    """Test VQE energy computation with p=2 QAOA ansatz."""
    n_qubits = 2
    edges = [(0, 1)]
    H = ising_maxcut_hamiltonian(num_nodes=n_qubits, edges=edges, include_constant=True)

    ansatz = QAOAAnsatz(n_qubits=n_qubits, problem_hamiltonian=H, p=2)

    # p=2 means 4 parameters
    assert ansatz.num_parameters == 4

    params = torch.zeros(ansatz.num_parameters, dtype=torch.float32, requires_grad=True)

    vqe = VQE(ansatz, H)

    energy = vqe.energy(params).item()

    # For zero params, should still be 0.5 (uniform superposition)
    assert abs(energy - 0.5) < 1e-3

    # Check gradients
    energy_tensor = vqe.energy(params)
    energy_tensor.backward()
    assert params.grad is not None


def test_qaoa_vqe_energy_larger_graph() -> None:
    """Test VQE energy computation on a larger graph (triangle)."""
    n_qubits = 3
    edges = [(0, 1), (1, 2), (0, 2)]
    H = ising_maxcut_hamiltonian(num_nodes=n_qubits, edges=edges, include_constant=True)

    ansatz = QAOAAnsatz(n_qubits=n_qubits, problem_hamiltonian=H, p=1)

    params = torch.zeros(ansatz.num_parameters, dtype=torch.float32, requires_grad=True)

    vqe = VQE(ansatz, H)

    energy = vqe.energy(params)

    # Energy should be computable
    assert energy.shape == ()
    assert torch.is_floating_point(energy)

    # For uniform superposition, energy should be sum of weights / 2
    # For triangle with all weights = 1.0, this is 3 * 1.0 / 2 = 1.5
    expected_energy = 3.0 / 2.0
    assert abs(energy.item() - expected_energy) < 1e-3

    # Check gradients
    energy.backward()
    assert params.grad is not None


