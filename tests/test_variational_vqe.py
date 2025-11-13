"""Tests for VQE implementation."""

from __future__ import annotations

import pytest
import torch

from qconduit.exact import exact_ground_state
from qconduit.operators import PauliSum, PauliTerm
from qconduit.variational import (
    HardwareEfficientAnsatz,
    VQEResult,
    evaluate_expectation_value,
    run_vqe,
)
from qconduit.core.device import default_device


class TestEvaluateExpectationValue:
    """Tests for evaluate_expectation_value."""

    def test_simple_hamiltonian_z(self) -> None:
        """Test expectation value for H = Z on |0⟩ state."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        # Zero parameters should give |0⟩ state (or close to it)
        params = torch.zeros(ansatz.num_parameters)
        expectation = evaluate_expectation_value(ansatz, params, hamiltonian)
        # For |0⟩, ⟨Z⟩ = +1
        assert abs(expectation - 1.0) < 0.1  # Allow some tolerance

    def test_simple_hamiltonian_z_rotated(self) -> None:
        """Test expectation value for H = Z on rotated state."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        # Parameters to prepare |1⟩: Rx(π) should flip to |1⟩
        params = torch.zeros(ansatz.num_parameters)
        params[0] = torch.pi  # Rx(π)
        expectation = evaluate_expectation_value(ansatz, params, hamiltonian)
        # For |1⟩, ⟨Z⟩ = -1
        assert abs(expectation - (-1.0)) < 0.1  # Allow some tolerance

    def test_initial_state_provided(self) -> None:
        """Test that providing an initial state works."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        params = torch.zeros(ansatz.num_parameters)
        # Provide |1⟩ as initial state
        initial_state = torch.tensor([0.0, 1.0], dtype=torch.complex128)
        expectation = evaluate_expectation_value(
            ansatz, params, hamiltonian, initial_state=initial_state
        )
        # Should be close to -1 for |1⟩
        assert abs(expectation - (-1.0)) < 0.1

    def test_initial_state_normalization(self) -> None:
        """Test that unnormalized initial state is normalized."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        params = torch.zeros(ansatz.num_parameters)
        # Provide unnormalized |1⟩
        initial_state = torch.tensor([0.0, 2.0], dtype=torch.complex128)
        expectation = evaluate_expectation_value(
            ansatz, params, hamiltonian, initial_state=initial_state
        )
        # Should still work (state is normalized internally)
        assert abs(expectation - (-1.0)) < 0.1

    def test_initial_state_zero_norm(self) -> None:
        """Test that zero-norm initial state raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        params = torch.zeros(ansatz.num_parameters)
        initial_state = torch.zeros(2, dtype=torch.complex128)
        with pytest.raises(ValueError, match="zero norm"):
            evaluate_expectation_value(
                ansatz, params, hamiltonian, initial_state=initial_state
            )


class TestRunVQE:
    """Tests for run_vqe."""

    def test_simple_1qubit_problem(self) -> None:
        """Test VQE on simple 1-qubit problem H = Z."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
        initial_params = torch.zeros(ansatz.num_parameters)
        # Add small random perturbation
        initial_params += 0.1 * torch.randn_like(initial_params)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            optimizer_name="adam",
            max_iterations=50,
            learning_rate=0.1,
            tol_rel=1e-4,
        )

        assert isinstance(result, VQEResult)
        assert result.optimal_value < 0.0  # Ground state energy is -1
        assert abs(result.optimal_value - (-1.0)) < 0.1  # Should be close to -1
        assert result.n_evaluations > 0
        assert len(result.history) > 0

    def test_invalid_optimizer(self) -> None:
        """Test that invalid optimizer name raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)

        with pytest.raises(ValueError, match="Unsupported optimizer_name"):
            run_vqe(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                initial_params=initial_params,
                optimizer_name="invalid",
            )

    def test_wrong_initial_params_length(self) -> None:
        """Test that wrong initial_params length raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(5)  # Wrong length

        with pytest.raises(ValueError, match="initial_params length"):
            run_vqe(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                initial_params=initial_params,
            )

    def test_invalid_max_iterations(self) -> None:
        """Test that invalid max_iterations raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            run_vqe(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                initial_params=initial_params,
                max_iterations=0,
            )

    def test_invalid_learning_rate(self) -> None:
        """Test that invalid learning_rate raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            run_vqe(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                initial_params=initial_params,
                learning_rate=0.0,
            )

    def test_invalid_tol_rel(self) -> None:
        """Test that invalid tol_rel raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)

        with pytest.raises(ValueError, match="tol_rel must be > 0"):
            run_vqe(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                initial_params=initial_params,
                tol_rel=0.0,
            )

    def test_sgd_optimizer(self) -> None:
        """Test that SGD optimizer works."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
        initial_params = torch.zeros(ansatz.num_parameters)
        initial_params += 0.1 * torch.randn_like(initial_params)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            optimizer_name="sgd",
            max_iterations=30,
            learning_rate=0.1,
        )

        assert isinstance(result, VQEResult)
        assert result.optimal_value < 0.0

    def test_convergence(self) -> None:
        """Test that VQE can converge for simple problem."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
        initial_params = torch.zeros(ansatz.num_parameters)
        initial_params += 0.1 * torch.randn_like(initial_params)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            optimizer_name="adam",
            max_iterations=100,
            learning_rate=0.1,
            tol_rel=1e-5,
        )

        # Energy should decrease from initial
        if len(result.history) > 1:
            initial_energy = result.history[0][1]
            final_energy = result.history[-1][1]
            assert final_energy <= initial_energy

    def test_result_structure(self) -> None:
        """Test that VQEResult has correct structure."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            max_iterations=10,
        )

        assert result.optimal_params.shape == (ansatz.num_parameters,)
        assert isinstance(result.optimal_value, float)
        assert isinstance(result.history, list)
        assert len(result.history) > 0
        assert isinstance(result.n_evaluations, int)
        assert result.n_evaluations > 0
        assert isinstance(result.converged, bool)

    def test_layered_entangler_ansatz(self) -> None:
        """Test VQE with LayeredEntanglerAnsatz."""
        from qconduit.variational import LayeredEntanglerAnsatz

        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        ansatz = LayeredEntanglerAnsatz(num_qubits=1, num_layers=1)
        initial_params = torch.zeros(ansatz.num_parameters)
        initial_params += 0.1 * torch.randn_like(initial_params)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            max_iterations=20,
            learning_rate=0.1,
        )

        assert isinstance(result, VQEResult)
        assert result.optimal_value <= result.history[0][1]  # Should not increase

    def test_layered_entangler_ring(self) -> None:
        """Test VQE with LayeredEntanglerAnsatz with ring entanglement."""
        from qconduit.variational import LayeredEntanglerAnsatz

        hamiltonian = PauliSum.from_terms([
            PauliTerm(1.0, ("Z", "I")),
            PauliTerm(1.0, ("I", "Z")),
        ])
        ansatz = LayeredEntanglerAnsatz(num_qubits=2, num_layers=1, ring_entanglement=True)
        initial_params = torch.zeros(ansatz.num_parameters)
        initial_params += 0.1 * torch.randn_like(initial_params)

        result = run_vqe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_params=initial_params,
            max_iterations=20,
            learning_rate=0.1,
        )

        assert isinstance(result, VQEResult)
        assert result.optimal_value <= result.history[0][1]  # Should not increase

