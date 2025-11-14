"""Tests for VQE parameter-shift gradient utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.algorithms.vqe import VQE
from qconduit.grad import autograd_gradient, vqe_parameter_shift_gradient
from qconduit.layers.ansatzes import HardwareEfficientAnsatz
from qconduit.operators import PauliSum, PauliTerm


class TestVQEParameterShiftGradient:
    """Tests for vqe_parameter_shift_gradient function."""

    def test_vqe_parameter_shift_gradient_vs_analytic_1_qubit(self):
        """Test parameter-shift gradient vs analytic for 1-qubit RX ansatz with Z Hamiltonian."""
        # Setup: 1-qubit ansatz with single RX(θ) parameter
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = Z
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # For state |ψ(θ)⟩ = RX(θ)|0⟩, expectation of Z is cos(θ)
        # So energy E(θ) = cos(θ), dE/dθ = -sin(θ)
        theta = 0.7
        params = torch.tensor([theta], dtype=torch.float64)

        # Compute gradient via parameter-shift
        grad_ps = vqe_parameter_shift_gradient(vqe, params)

        # Should be a tensor of shape (1,)
        assert grad_ps.shape == (1,)

        # Analytic derivative: grad_true = -sin(θ)
        grad_true = -math.sin(theta)

        assert abs(grad_ps[0].item() - grad_true) < 1e-4

    def test_vqe_parameter_shift_gradient_vs_autograd(self):
        """Test parameter-shift vs autograd for VQE energy."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.7], dtype=torch.float64)

        # Parameter-shift gradient
        grad_ps = vqe_parameter_shift_gradient(vqe, params)

        # Autograd gradient
        def objective(p: torch.Tensor) -> torch.Tensor:
            return vqe.energy(p)

        grad_auto = autograd_gradient(objective, params)

        # Should agree to reasonable precision
        assert torch.allclose(grad_ps, grad_auto, atol=1e-4, rtol=1e-4)

    def test_vqe_parameter_shift_gradient_subset_indices(self):
        """Test parameter-shift gradient with subset of indices for multi-parameter ansatz."""
        # 2-qubit ansatz with 2 depth = 4 parameters
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        # Hamiltonian: H = Z ⊗ I (only depends on first qubit)
        term = PauliTerm(coeff=1.0, paulis=("Z", "I"))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        # Compute full gradient
        grad_full = vqe_parameter_shift_gradient(vqe, params, indices=None)

        # Compute gradient only for first parameter
        grad_subset = vqe_parameter_shift_gradient(vqe, params, indices=[0])

        assert grad_subset.shape == (1,)
        # First component should match
        assert abs(grad_subset[0].item() - grad_full[0].item()) < 1e-5

    def test_vqe_parameter_shift_gradient_multi_param_ansatz(self):
        """Test parameter-shift gradient with multi-parameter ansatz."""
        # 2-qubit ansatz with 2 depth = 4 parameters
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        term = PauliTerm(coeff=1.0, paulis=("Z", "I"))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        # Compute gradient
        grad = vqe_parameter_shift_gradient(vqe, params)

        # Should have shape (4,)
        assert grad.shape == (4,)
        # All gradients should be finite
        assert torch.isfinite(grad).all()

    def test_vqe_parameter_shift_gradient_error_params_mismatch(self):
        """Test vqe_parameter_shift_gradient raises for params length mismatch."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # This ansatz has num_parameters = 1
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Pass params with wrong length
        params = torch.tensor([0.1, 0.2], dtype=torch.float64)  # Length 2, but ansatz expects 1

        with pytest.raises(ValueError, match="does not match"):
            vqe_parameter_shift_gradient(vqe, params)

    def test_vqe_parameter_shift_gradient_with_diagonal_hamiltonian(self):
        """Test parameter-shift gradient with diagonal Hamiltonian."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Diagonal Hamiltonian: H = diag([0.0, 1.0])
        hamiltonian = torch.tensor([0.0, 1.0], dtype=torch.float32)
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.3], dtype=torch.float64)

        # Compute gradient
        grad = vqe_parameter_shift_gradient(vqe, params)

        # Should be a tensor of shape (1,)
        assert grad.shape == (1,)
        # Gradient should be finite
        assert torch.isfinite(grad).all()

    def test_vqe_parameter_shift_gradient_reordered_indices(self):
        """Test parameter-shift gradient with reordered indices."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        term = PauliTerm(coeff=1.0, paulis=("Z", "I"))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        # Compute gradient with indices in reverse order
        grad_reversed = vqe_parameter_shift_gradient(vqe, params, indices=[3, 2, 1, 0])
        grad_normal = vqe_parameter_shift_gradient(vqe, params, indices=[0, 1, 2, 3])

        assert grad_reversed.shape == (4,)
        assert grad_normal.shape == (4,)
        # Reversed should be reverse of normal
        assert torch.allclose(grad_reversed, grad_normal.flip(0), atol=1e-5)

    def test_vqe_parameter_shift_gradient_optimization_consistency(self):
        """Test that parameter-shift gradients enable optimization."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = Z, ground state is |1⟩ with energy -1
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Start at a point with non-zero gradient
        params = torch.tensor([0.1], dtype=torch.float64)

        # Compute gradient
        grad = vqe_parameter_shift_gradient(vqe, params)

        # Gradient should be non-zero (we're not at a critical point)
        assert abs(grad[0].item()) > 1e-6

        # Verify gradient points in direction that decreases energy
        # For H=Z and RX(θ)|0⟩, energy = cos(θ), gradient = -sin(θ)
        # At θ=0.1, gradient should be negative (pointing toward larger θ)
        # This is consistent with moving toward θ=π where energy is minimum (-1)
        assert grad[0].item() < 0


