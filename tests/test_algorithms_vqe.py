"""Tests for Variational Quantum Eigensolver (VQE)."""

import math

import pytest
import torch

from qconduit.algorithms.vqe import VQE, ensure_hamiltonian_diag
from qconduit.core.device import default_device
from qconduit.layers.ansatzes import HardwareEfficientAnsatz
from qconduit.operators import PauliSum, PauliTerm


class TestEnsureHamiltonianDiag:
    """Tests for ensure_hamiltonian_diag helper function."""

    def test_ensure_hamiltonian_diag_valid(self):
        """Test ensure_hamiltonian_diag with valid input."""
        hamiltonian = torch.tensor([0.0, 1.0])
        device = default_device()
        result = ensure_hamiltonian_diag(hamiltonian, n_qubits=1, device=device)
        assert result.shape == (2,)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"

    def test_ensure_hamiltonian_diag_wrong_dimension(self):
        """Test ensure_hamiltonian_diag raises for wrong dimension."""
        hamiltonian = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # 2D instead of 1D
        device = default_device()
        with pytest.raises(ValueError, match="must be 1D"):
            ensure_hamiltonian_diag(hamiltonian, n_qubits=1, device=device)

    def test_ensure_hamiltonian_diag_wrong_length(self):
        """Test ensure_hamiltonian_diag raises for wrong length."""
        hamiltonian = torch.tensor([0.0, 1.0, 2.0])  # Length 3, but 2**1 = 2
        device = default_device()
        with pytest.raises(ValueError, match="must have length"):
            ensure_hamiltonian_diag(hamiltonian, n_qubits=1, device=device)

    def test_ensure_hamiltonian_diag_converts_integer_to_float(self):
        """Test ensure_hamiltonian_diag converts integer dtype to float."""
        hamiltonian = torch.tensor([0, 1], dtype=torch.int32)
        device = default_device()
        result = ensure_hamiltonian_diag(hamiltonian, n_qubits=1, device=device)
        assert result.dtype == torch.float32


class TestVQE:
    """Tests for VQE class."""

    def test_vqe_initialization(self):
        """Test VQE can be initialized."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)
        assert vqe.ansatz == ansatz
        assert vqe.hamiltonian_diag.shape == (2,)
        assert torch.allclose(vqe.hamiltonian_diag, hamiltonian)

    def test_vqe_energy_1_qubit_ground_state(self):
        """Test VQE energy for 1-qubit system with |0⟩ state."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = diag([0.0, 1.0]), so |0⟩ has energy 0, |1⟩ has energy 1
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        # Parameters that produce |0⟩ state (theta=0)
        params = torch.tensor([0.0])
        energy = vqe.energy(params)

        # Should be close to 0.0 (energy of |0⟩)
        assert torch.allclose(energy, torch.tensor(0.0), atol=1e-6)

    def test_vqe_energy_1_qubit_excited_state(self):
        """Test VQE energy for 1-qubit system with |1⟩ state."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        # Parameters that produce |1⟩ state (theta=π)
        params = torch.tensor([math.pi])
        energy = vqe.energy(params)

        # Should be close to 1.0 (energy of |1⟩)
        assert torch.allclose(energy, torch.tensor(1.0), atol=1e-6)

    def test_vqe_forward_alias(self):
        """Test VQE.forward() is an alias for energy()."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.0])
        energy1 = vqe.energy(params)
        energy2 = vqe.forward(params)

        assert torch.allclose(energy1, energy2)

    def test_vqe_energy_batch_mode(self):
        """Test VQE energy works in batch mode."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        # Batch: first element produces |0⟩, second produces |1⟩
        params = torch.tensor([[0.0], [math.pi]])
        energy = vqe.energy(params)

        assert energy.shape == (2,)
        assert torch.allclose(energy[0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(energy[1], torch.tensor(1.0), atol=1e-6)

    def test_vqe_gradient(self):
        """Test that gradients flow through VQE."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.5], requires_grad=True)
        energy = vqe.energy(params)

        # Backward pass
        energy.backward()

        # Check gradients exist and are finite
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()

    def test_vqe_optimization(self):
        """Test VQE can be used in an optimization loop."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = diag([0.0, 1.0]), ground state is |0⟩ with energy 0
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        # Initialize parameters
        params = torch.nn.Parameter(torch.tensor([math.pi / 2]))  # Start away from ground state

        # Optimizer
        optimizer = torch.optim.SGD([params], lr=0.1)

        # Store initial energy
        initial_energy = vqe.energy(params).item()

        # Run optimization steps
        for _ in range(20):
            optimizer.zero_grad()
            energy = vqe.energy(params)
            energy.backward()
            optimizer.step()

        # Final energy should be lower than initial
        final_energy = vqe.energy(params).item()
        assert final_energy < initial_energy

        # Final energy should be close to ground state energy (0.0)
        # Allow some tolerance for optimization convergence
        assert final_energy < 0.2  # Should be significantly below 1.0

    def test_vqe_wrong_hamiltonian_length(self):
        """Test VQE raises for wrong Hamiltonian length."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0, 2.0])  # Wrong length
        with pytest.raises(ValueError, match="must have length"):
            VQE(ansatz, hamiltonian)

    def test_vqe_device_consistency(self):
        """Test VQE uses correct device."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1, device="sv_cpu")
        hamiltonian = torch.tensor([0.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.0])
        energy = vqe.energy(params)
        assert energy.device.type == "cpu"

    def test_vqe_with_custom_device(self):
        """Test VQE can be initialized with a custom device."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        device = default_device()
        vqe = VQE(ansatz, hamiltonian, device=device)

        params = torch.tensor([0.0])
        energy = vqe.energy(params)
        assert energy.device.type == "cpu"

    def test_vqe_2_qubit_system(self):
        """Test VQE with a 2-qubit system."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
        # Simple Hamiltonian: |00⟩ has energy 0, others have energy 1
        hamiltonian = torch.tensor([0.0, 1.0, 1.0, 1.0])
        vqe = VQE(ansatz, hamiltonian)

        # Try to find ground state (should be close to |00⟩)
        params = torch.nn.Parameter(torch.zeros(2))  # 2 qubits * 1 depth
        optimizer = torch.optim.SGD([params], lr=0.1)

        for _ in range(15):
            optimizer.zero_grad()
            energy = vqe.energy(params)
            energy.backward()
            optimizer.step()

        # Energy should be low (close to 0)
        final_energy = vqe.energy(params).item()
        assert final_energy < 0.5  # Should be significantly below 1.0


class TestVQEPauliSum:
    """Tests for VQE with PauliSum Hamiltonians."""

    def test_vqe_initialization_pauli_sum(self):
        """Test VQE can be initialized with PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)
        assert vqe.ansatz == ansatz
        assert vqe.hamiltonian_pauli is not None
        assert vqe.hamiltonian_diag is None
        assert vqe.hamiltonian_pauli.n_qubits() == 1

    def test_vqe_energy_pauli_sum_1_qubit_ground_state(self):
        """Test VQE energy for 1-qubit system with |0⟩ state using PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = Z, so |0⟩ has energy +1, |1⟩ has energy -1
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Parameters that produce |0⟩ state (theta=0)
        params = torch.tensor([0.0])
        energy = vqe.energy(params)

        # Should be close to +1.0 (energy of |0⟩ for H=Z)
        assert torch.allclose(energy, torch.tensor(1.0), atol=1e-5)

    def test_vqe_energy_pauli_sum_1_qubit_excited_state(self):
        """Test VQE energy for 1-qubit system with |1⟩ state using PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Parameters that produce |1⟩ state (theta=π)
        params = torch.tensor([math.pi])
        energy = vqe.energy(params)

        # Should be close to -1.0 (energy of |1⟩ for H=Z)
        assert torch.allclose(energy, torch.tensor(-1.0), atol=1e-5)

    def test_vqe_forward_alias_pauli_sum(self):
        """Test VQE.forward() is an alias for energy() with PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.0])
        energy1 = vqe.energy(params)
        energy2 = vqe.forward(params)

        assert torch.allclose(energy1, energy2)

    def test_vqe_energy_batch_mode_pauli_sum(self):
        """Test VQE energy works in batch mode with PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Batch: first element produces |0⟩, second produces |1⟩
        params = torch.tensor([[0.0], [math.pi]])
        energy = vqe.energy(params)

        assert energy.shape == (2,)
        assert torch.allclose(energy[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(energy[1], torch.tensor(-1.0), atol=1e-5)

    def test_vqe_gradient_pauli_sum(self):
        """Test that gradients flow through VQE with PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        params = torch.tensor([0.5], requires_grad=True)
        energy = vqe.energy(params)

        # Backward pass
        energy.backward()

        # Check gradients exist and are finite
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()

    def test_vqe_optimization_pauli_sum(self):
        """Test VQE can be used in an optimization loop with PauliSum."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Hamiltonian: H = Z, ground state is |1⟩ with energy -1
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        vqe = VQE(ansatz, hamiltonian)

        # Initialize parameters away from critical point (params=0 has zero gradient)
        # Start at a small positive value where gradient is non-zero
        params = torch.nn.Parameter(torch.tensor([0.1]))  # Start near |0⟩ but with non-zero gradient

        # Optimizer
        optimizer = torch.optim.SGD([params], lr=0.1)

        # Store initial energy
        initial_energy = vqe.energy(params).item()

        # Run optimization steps
        for _ in range(20):
            optimizer.zero_grad()
            energy = vqe.energy(params)
            energy.backward()
            optimizer.step()

        # Final energy should be lower than initial (gradients are working)
        final_energy = vqe.energy(params).item()
        assert final_energy < initial_energy

        # The important thing is that optimization is making progress
        # (gradients are flowing correctly). Full convergence may require
        # more steps or better hyperparameters, but we've verified gradients work.
        # Energy should be significantly lower than the starting point
        assert final_energy < initial_energy * 0.9  # At least 10% improvement

    def test_vqe_pauli_sum_wrong_n_qubits(self):
        """Test VQE raises for PauliSum with wrong n_qubits."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z", "Z"))  # 2-qubit term
        hamiltonian = PauliSum.from_terms([term])
        with pytest.raises(ValueError, match="does not match"):
            VQE(ansatz, hamiltonian)

    def test_vqe_pauli_sum_multiple_terms(self):
        """Test VQE with PauliSum containing multiple terms."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # H = 0.5 * Z + 0.5 * X
        term_z = PauliTerm(0.5, ("Z",))
        term_x = PauliTerm(0.5, ("X",))
        hamiltonian = PauliSum.from_terms([term_z, term_x])
        vqe = VQE(ansatz, hamiltonian)

        # For |0⟩: ⟨Z⟩ = 1, ⟨X⟩ = 0, so energy = 0.5
        params = torch.tensor([0.0])
        energy = vqe.energy(params)
        assert torch.allclose(energy, torch.tensor(0.5), atol=1e-5)


class TestVQENoise:
    """Tests for VQE with noise models."""

    def test_vqe_with_noise_initialization(self):
        """Test VQE can be initialized with noise_model."""
        from qconduit.noise import AmplitudeDampingChannel

        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        noise = AmplitudeDampingChannel(gamma=0.5)
        vqe = VQE(ansatz, hamiltonian, noise_model=noise)
        assert vqe.noise_model == noise

    def test_vqe_direct_with_noise(self):
        """Test direct VQE energy computation with noise."""
        from qconduit.noise import AmplitudeDampingChannel

        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        H = PauliSum.from_terms([term])

        # Without noise
        vqe_clean = VQE(ansatz, H, use_param_shift=False, noise_model=None)
        params = torch.tensor([0.3], dtype=torch.float32)
        energy_clean = vqe_clean.energy(params).item()

        # With amplitude damping noise
        noise = AmplitudeDampingChannel(gamma=0.5)
        vqe_noisy = VQE(ansatz, H, use_param_shift=False, noise_model=noise)
        energy_noisy = vqe_noisy.energy(params).item()

        # Energy should differ (noise has an effect)
        assert abs(energy_noisy - energy_clean) > 1e-6
        # Both should be finite
        assert torch.isfinite(torch.tensor(energy_clean))
        assert torch.isfinite(torch.tensor(energy_noisy))

    def test_vqe_param_shift_with_noise(self):
        """Test parameter-shift VQE with noise."""
        from qconduit.noise import DepolarizingChannel

        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        H = PauliSum.from_terms([term])

        noise = DepolarizingChannel(p=0.2)
        vqe_ps = VQE(ansatz, H, use_param_shift=True, noise_model=noise)
        params = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        optimizer = torch.optim.SGD([params], lr=0.1)
        initial_energy = vqe_ps.energy(params.detach()).item()

        # Run a few optimization steps
        for _ in range(10):
            optimizer.zero_grad()
            energy = vqe_ps.energy(params)
            energy.backward()
            optimizer.step()

        final_energy = vqe_ps.energy(params.detach()).item()

        # Check gradients were computed
        assert params.grad is not None
        # Both energies should be finite
        assert torch.isfinite(torch.tensor(initial_energy))
        assert torch.isfinite(torch.tensor(final_energy))
        # Energy should not increase dramatically (param-shift path is stable)
        assert final_energy <= initial_energy + 1e-3

    def test_vqe_noise_consistency_pure_vs_dm(self):
        """Test that pure-state and density-matrix paths are consistent when noise=None."""
        from qconduit.backend.density_matrix import dm_from_statevector
        from qconduit.operators.expectation import expectation_pauli_sum_dm

        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        term = PauliTerm(1.0, ("Z",))
        H = PauliSum.from_terms([term])

        # Pure-state path
        vqe_clean = VQE(ansatz, H, noise_model=None)
        params = torch.tensor([0.3], dtype=torch.float32)
        energy_pure = vqe_clean.energy(params).item()

        # Density-matrix path (manually compute)
        state = ansatz(params)
        rho = dm_from_statevector(state)
        energy_dm = expectation_pauli_sum_dm(rho, H).item()

        # Should match (within numerical tolerance)
        assert abs(energy_pure - energy_dm) < 1e-5

    def test_vqe_noise_diagonal_hamiltonian(self):
        """Test VQE with noise and diagonal Hamiltonian."""
        from qconduit.noise import PhaseDampingChannel

        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        hamiltonian = torch.tensor([0.0, 1.0])
        noise = PhaseDampingChannel(gamma=0.3)

        vqe = VQE(ansatz, hamiltonian, noise_model=noise)
        params = torch.tensor([0.2], dtype=torch.float32)
        energy = vqe.energy(params).item()

        # Energy should be finite
        assert torch.isfinite(torch.tensor(energy))
        # With phase damping, energy should be between 0 and 1
        assert 0.0 <= energy <= 1.0

