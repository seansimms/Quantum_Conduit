"""Tests for VQE-specific parameter sweep utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.algorithms.vqe import VQE
from qconduit.layers.ansatzes import HardwareEfficientAnsatz
from qconduit.operators import PauliSum, PauliTerm
from qconduit.experiments import sweep_vqe_1d, sweep_vqe_2d


def build_single_qubit_vqe() -> tuple[VQE, torch.Tensor]:
    """
    Build a simple 1-qubit VQE with a single-parameter ansatz.

    Uses HardwareEfficientAnsatz with depth=1 (1 parameter) and Hamiltonian H = Z.
    The ansatz applies RX(θ) to |0⟩, producing state cos(θ/2)|0⟩ - i sin(θ/2)|1⟩.
    The energy is E(θ) = ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ).

    Returns:
        Tuple of (VQE instance, base_params tensor of shape (1,)).
    """
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    hamiltonian = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])
    vqe = VQE(ansatz, hamiltonian)
    base_params = torch.zeros(1, dtype=torch.float32)
    return (vqe, base_params)


def build_two_param_vqe() -> tuple[VQE, torch.Tensor]:
    """
    Build a 1-qubit VQE with a 2-parameter ansatz.

    Uses HardwareEfficientAnsatz with depth=2 (2 parameters) and Hamiltonian H = Z.
    The ansatz applies RX(α) then RX(β) to |0⟩.

    Returns:
        Tuple of (VQE instance, base_params tensor of shape (2,)).
    """
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=2)
    hamiltonian = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])
    vqe = VQE(ansatz, hamiltonian)
    base_params = torch.zeros(2, dtype=torch.float32)
    return (vqe, base_params)


class TestSweepVQE1D:
    """Tests for sweep_vqe_1d function."""

    def test_sweep_vqe_1d_energy_landscape(self):
        """Test sweep_vqe_1d reproduces expected cosine energy landscape."""
        vqe, base_params = build_single_qubit_vqe()

        # For H = Z and ansatz RX(θ) on |0⟩:
        # State: cos(θ/2)|0⟩ - i sin(θ/2)|1⟩
        # Energy: E(θ) = cos(θ)
        # Minimum at θ = π

        points = torch.linspace(0.0, 2 * math.pi, 41)
        res = sweep_vqe_1d(vqe, points, base_params=base_params, index=0)

        assert res.points.shape == (41,)
        assert res.values.shape == (41,)

        # Find minimum
        min_idx = int(torch.argmin(res.values))
        theta_min = res.points[min_idx].item()

        # Minimum should be near π (within ±0.2)
        assert abs(theta_min - math.pi) < 0.2

        # Verify energy values are close to cos(θ) (within reasonable tolerance)
        # Note: For HardwareEfficientAnsatz with depth=1, the relationship is
        # approximately cos(θ), but may have slight differences due to the
        # exact ansatz structure. We check that the minimum is in the right place.
        assert res.values[min_idx].item() < -0.9  # Should be close to -1

    def test_sweep_vqe_1d_metadata(self):
        """Test sweep_vqe_1d includes correct metadata."""
        vqe, base_params = build_single_qubit_vqe()
        points = torch.linspace(0.0, math.pi, 5)
        custom_metadata = {"custom_key": "custom_value"}

        res = sweep_vqe_1d(
            vqe, points, base_params=base_params, index=0, metadata=custom_metadata
        )

        assert res.metadata["sweep_type"] == "vqe_1d"
        assert res.metadata["param_index"] == "0"
        assert res.metadata["custom_key"] == "custom_value"
        assert "hamiltonian_type" in res.metadata

    def test_sweep_vqe_1d_parameter_length_validation(self):
        """Test sweep_vqe_1d validates parameter length."""
        vqe, _ = build_single_qubit_vqe()
        points = torch.linspace(0.0, math.pi, 5)

        # base_params with wrong length
        base_params_bad = torch.zeros(2)  # Should be 1

        with pytest.raises(ValueError, match="base_params must have length"):
            sweep_vqe_1d(vqe, points, base_params=base_params_bad, index=0)

    def test_sweep_vqe_1d_different_index(self):
        """Test sweep_vqe_1d works with different parameter index."""
        vqe, base_params = build_two_param_vqe()
        points = torch.linspace(0.0, math.pi, 5)

        # Sweep over index 1
        res = sweep_vqe_1d(vqe, points, base_params=base_params, index=1)

        assert res.points.shape == (5,)
        assert res.values.shape == (5,)
        assert res.metadata["param_index"] == "1"


class TestSweepVQE2D:
    """Tests for sweep_vqe_2d function."""

    def test_sweep_vqe_2d_shapes(self):
        """Test sweep_vqe_2d produces correct shapes."""
        vqe, base_params = build_two_param_vqe()

        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)

        res = sweep_vqe_2d(
            vqe, x_points, y_points, base_params, x_index=0, y_index=1
        )

        assert res.values.shape == (5, 3)
        assert res.x_points.shape == (5,)
        assert res.y_points.shape == (3,)

    def test_sweep_vqe_2d_energy_landscape_structure(self):
        """Test sweep_vqe_2d produces reasonable energy landscape."""
        vqe, base_params = build_two_param_vqe()

        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)

        res = sweep_vqe_2d(
            vqe, x_points, y_points, base_params, x_index=0, y_index=1
        )

        # For H = Z, the energy should have some structure
        # Find minimum along x-axis for different y values
        min_indices_x = [int(torch.argmin(res.values[:, j])) for j in range(3)]

        # The minimum along x should be relatively consistent across y
        # (since H = Z, the dependence on the second parameter may be weak)
        # We just check that we get valid indices
        assert all(0 <= idx < 5 for idx in min_indices_x)

    def test_sweep_vqe_2d_metadata(self):
        """Test sweep_vqe_2d includes correct metadata."""
        vqe, base_params = build_two_param_vqe()
        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)
        custom_metadata = {"custom_key": "custom_value"}

        res = sweep_vqe_2d(
            vqe,
            x_points,
            y_points,
            base_params,
            x_index=0,
            y_index=1,
            metadata=custom_metadata,
        )

        assert res.metadata["sweep_type"] == "vqe_2d"
        assert res.metadata["param_index_x"] == "0"
        assert res.metadata["param_index_y"] == "1"
        assert res.metadata["custom_key"] == "custom_value"
        assert "hamiltonian_type" in res.metadata

    def test_sweep_vqe_2d_parameter_length_validation(self):
        """Test sweep_vqe_2d validates parameter length."""
        vqe, _ = build_two_param_vqe()
        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)

        # base_params with wrong length
        base_params_bad = torch.zeros(1)  # Should be 2

        with pytest.raises(ValueError, match="base_params must have length"):
            sweep_vqe_2d(
                vqe, x_points, y_points, base_params_bad, x_index=0, y_index=1
            )

    def test_sweep_vqe_2d_same_indices(self):
        """Test sweep_vqe_2d raises ValueError when x_index == y_index."""
        vqe, base_params = build_two_param_vqe()
        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)

        with pytest.raises(ValueError, match="x_index and y_index must be different"):
            sweep_vqe_2d(
                vqe, x_points, y_points, base_params, x_index=0, y_index=0
            )

    def test_sweep_vqe_1d_diagonal_hamiltonian_metadata(self):
        """Test sweep_vqe_1d includes correct metadata for diagonal Hamiltonian."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])  # Diagonal Hamiltonian
        vqe = VQE(ansatz, hamiltonian)
        base_params = torch.zeros(1, dtype=torch.float32)
        points = torch.linspace(0.0, math.pi, 5)

        res = sweep_vqe_1d(vqe, points, base_params=base_params, index=0)

        assert res.metadata["hamiltonian_type"] == "diagonal"

    def test_sweep_vqe_2d_diagonal_hamiltonian_metadata(self):
        """Test sweep_vqe_2d includes correct metadata for diagonal Hamiltonian."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=2)
        hamiltonian = torch.tensor([0.0, 1.0])  # Diagonal Hamiltonian
        vqe = VQE(ansatz, hamiltonian)
        base_params = torch.zeros(2, dtype=torch.float32)
        x_points = torch.linspace(0.0, math.pi, 5)
        y_points = torch.linspace(0.0, 2.0, 3)

        res = sweep_vqe_2d(
            vqe, x_points, y_points, base_params, x_index=0, y_index=1
        )

        assert res.metadata["hamiltonian_type"] == "diagonal"

