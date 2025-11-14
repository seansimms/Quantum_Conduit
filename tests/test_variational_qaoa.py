"""Tests for QAOA implementation."""

from __future__ import annotations

import pytest
import torch

from qconduit.exact import exact_ground_state
from qconduit.operators import PauliSum, PauliTerm
from qconduit.variational import QAOAResult, run_qaoa
from qconduit.core.device import default_device


class TestRunQAOA:
    """Tests for run_qaoa."""

    def test_simple_1qubit_ising(self) -> None:
        """Test QAOA on simple 1-qubit Ising Hamiltonian H_C = Z."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            optimizer_name="adam",
            max_iterations=50,
            learning_rate=0.1,
        )

        assert isinstance(result, QAOAResult)
        assert result.optimal_value < 0.0  # Ground state energy is -1
        assert abs(result.optimal_value - (-1.0)) < 0.2  # Should be close to -1
        assert result.n_evaluations > 0

    def test_initial_params_autodetection(self) -> None:
        """Test that initial_params=None works and is deterministic."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        result1 = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            initial_params=None,
            max_iterations=10,
        )
        result2 = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            initial_params=None,
            max_iterations=10,
        )

        # Results should be identical due to fixed seed
        assert torch.allclose(result1.optimal_params, result2.optimal_params)

    def test_initial_params_provided(self) -> None:
        """Test that providing initial_params works."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        initial_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            initial_params=initial_params,
            max_iterations=10,
        )

        assert isinstance(result, QAOAResult)
        assert result.optimal_params.shape == (2,)

    def test_wrong_initial_params_length(self) -> None:
        """Test that wrong initial_params length raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        initial_params = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)  # Wrong length

        with pytest.raises(ValueError, match="initial_params length"):
            run_qaoa(
                cost_hamiltonian=hamiltonian,
                num_qubits=1,
                depth=1,
                initial_params=initial_params,
            )

    def test_validation_num_qubits(self) -> None:
        """Test that invalid num_qubits raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            run_qaoa(
                cost_hamiltonian=hamiltonian,
                num_qubits=0,
                depth=1,
            )

    def test_validation_depth(self) -> None:
        """Test that invalid depth raises ValueError."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        with pytest.raises(ValueError, match="depth must be >= 1"):
            run_qaoa(
                cost_hamiltonian=hamiltonian,
                num_qubits=1,
                depth=0,
            )

    def test_depth_2(self) -> None:
        """Test QAOA with depth=2."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=2,
            max_iterations=30,
            learning_rate=0.1,
        )

        assert isinstance(result, QAOAResult)
        assert result.optimal_params.shape == (4,)  # 2 * depth
        assert result.optimal_value < 0.0

    def test_energy_reduction(self) -> None:
        """Test that QAOA reduces energy from initial expectation."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            max_iterations=50,
            learning_rate=0.1,
        )

        # Energy should be reduced (or at least not increased significantly)
        if len(result.history) > 1:
            initial_energy = result.history[0][1]
            final_energy = result.history[-1][1]
            # Final energy should be <= initial (allowing small numerical errors)
            assert final_energy <= initial_energy + 1e-6

    def test_sgd_optimizer(self) -> None:
        """Test that SGD optimizer works for QAOA."""
        hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=1,
            depth=1,
            optimizer_name="sgd",
            max_iterations=30,
            learning_rate=0.1,
        )

        assert isinstance(result, QAOAResult)
        assert result.optimal_value < 0.0

    def test_2qubit_problem(self) -> None:
        """Test QAOA on a 2-qubit problem."""
        # H = Z_0 + Z_1 (simple Ising)
        hamiltonian = PauliSum.from_terms([
            PauliTerm(1.0, ("Z", "I")),
            PauliTerm(1.0, ("I", "Z")),
        ])
        result = run_qaoa(
            cost_hamiltonian=hamiltonian,
            num_qubits=2,
            depth=1,
            max_iterations=30,
            learning_rate=0.1,
        )

        assert isinstance(result, QAOAResult)
        assert result.optimal_params.shape == (2,)  # 2 * depth
        # Ground state energy is -2 (both qubits in |1⟩)
        assert result.optimal_value < 0.0


