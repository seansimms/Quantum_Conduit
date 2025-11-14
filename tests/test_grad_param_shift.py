"""Tests for parameter-shift gradient engine."""

from __future__ import annotations

import math

import torch

from qconduit.grad import param_shift_energy
from qconduit.layers.ansatzes import HardwareEfficientAnsatz
from qconduit.operators.pauli import PauliTerm, PauliSum
from qconduit.algorithms.vqe import VQE


def finite_diff(
    ansatz: HardwareEfficientAnsatz,
    h: PauliSum,
    theta_val: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute finite-difference gradient approximation.

    This is a helper for testing; it computes the gradient numerically
    by evaluating energy at shifted parameter values.
    """
    t_plus = theta_val.clone().detach() + eps
    t_minus = theta_val.clone().detach() - eps
    with torch.no_grad():
        e_plus = param_shift_energy(ansatz, h, t_plus, device=None).detach()
        e_minus = param_shift_energy(ansatz, h, t_minus, device=None).detach()
    return (e_plus - e_minus) / (2.0 * eps)


def test_param_shift_gradient_vs_finite_diff():
    """Test that parameter-shift gradients match finite-difference approximations."""
    # Setup: 1-qubit ansatz with PauliSum H = Z
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    term = PauliTerm(coeff=1.0, paulis=("Z",))
    h = PauliSum.from_terms([term])

    # Choose an initial parameter value
    theta = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)

    # Compute energy with parameter-shift and get gradient
    energy = param_shift_energy(ansatz, h, theta)
    energy.backward()
    grad_ps = theta.grad.detach().clone()

    # Compute finite-difference gradient
    grad_fd = finite_diff(ansatz, h, theta.detach())

    # Compare: should be close (within tolerance for numerical finite diff)
    assert torch.allclose(grad_ps, grad_fd, atol=1e-3, rtol=1e-3), (
        f"Parameter-shift gradient {grad_ps.item():.6f} does not match "
        f"finite-difference gradient {grad_fd.item():.6f}"
    )


def test_vqe_with_param_shift_optimizes():
    """Test that VQE with use_param_shift=True can optimize energy."""
    # Setup: 1-qubit ansatz with H = Z
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    term = PauliTerm(coeff=1.0, paulis=("Z",))
    h = PauliSum.from_terms([term])

    # Build VQE with parameter-shift enabled
    vqe = VQE(ansatz, h, use_param_shift=True)

    # Initialize parameters
    params = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
    optimizer = torch.optim.SGD([params], lr=0.1)

    # Record initial energy
    initial_energy = vqe.energy(params.detach()).item()

    # Run optimization loop
    for _ in range(15):
        optimizer.zero_grad()
        energy = vqe.energy(params)
        energy.backward()
        optimizer.step()

    # Record final energy
    final_energy = vqe.energy(params.detach()).item()

    # Energy should decrease (for H=Z, ground state is |0⟩ with energy -1)
    # Starting from 0.1, we should move toward better energy
    assert final_energy < initial_energy - 1e-3, (
        f"Energy did not decrease: initial={initial_energy:.6f}, "
        f"final={final_energy:.6f}"
    )


def test_param_shift_wrong_params_shape():
    """Test that param_shift_energy raises ValueError for wrong params shape."""
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    term = PauliTerm(coeff=1.0, paulis=("Z",))
    h = PauliSum.from_terms([term])

    # Try with 2D params (should fail)
    params_2d = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    with torch.no_grad():
        try:
            param_shift_energy(ansatz, h, params_2d)
            assert False, "Expected ValueError for 2D params"
        except ValueError as e:
            assert "1D tensor" in str(e) or "1 dimensions" in str(e)


def test_param_shift_with_diagonal_hamiltonian():
    """Test parameter-shift with diagonal Hamiltonian."""
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    # Diagonal Hamiltonian: H = diag([0.0, 1.0])
    h_diag = torch.tensor([0.0, 1.0], dtype=torch.float32)

    theta = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)

    # Compute energy with parameter-shift
    energy = param_shift_energy(ansatz, h_diag, theta)
    energy.backward()
    grad_ps = theta.grad.detach().clone()

    # Verify gradient is computed (not None, not NaN)
    assert grad_ps is not None
    assert not torch.isnan(grad_ps).any()
    assert grad_ps.shape == (1,)


def test_param_shift_multiple_parameters():
    """Test parameter-shift with multiple parameters."""
    # 2-qubit ansatz with 2 depth = 4 parameters
    ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
    term = PauliTerm(coeff=1.0, paulis=("Z", "I"))
    h = PauliSum.from_terms([term])

    theta = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, requires_grad=True)

    # Compute energy with parameter-shift
    energy = param_shift_energy(ansatz, h, theta)
    energy.backward()
    grad_ps = theta.grad.detach().clone()

    # Verify gradient shape matches parameters
    assert grad_ps.shape == (4,)
    assert not torch.isnan(grad_ps).any()

    # Verify all gradients are computed (not all zeros, unless truly zero)
    # At least some should be non-zero for this setup
    assert torch.any(torch.abs(grad_ps) > 1e-6)


