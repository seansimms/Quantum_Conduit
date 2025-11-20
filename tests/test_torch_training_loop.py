"""End-to-end tests for PyTorch training loops with QuantumModule."""

import torch

from qconduit.operators import PauliSum, PauliTerm
from qconduit.torch import QuantumModule
from qconduit.variational import HardwareEfficientAnsatz


class TestTorchTrainingLoop:
    """End-to-end tests for training loops."""

    def test_simple_training_loop_ground_state(self):
        """
        Test a simple training loop that finds the ground state.

        Uses a 1-qubit system with H=Z. The ground state is |1⟩ with energy -1.0.
        We use an ansatz that can prepare |1⟩ with a single Rx parameter.
        """
        # Create ansatz: HardwareEfficientAnsatz with 1 qubit, 1 layer
        # This has 2 parameters: [theta_rx, theta_rz]
        # For |1⟩ state, we need Rx(π) applied to |0⟩
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)

        # Hamiltonian: H = Z
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        # Initialize with small parameters (close to |0⟩)
        init_params = torch.tensor([0.1, 0.0], dtype=torch.float64)

        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        optimizer = torch.optim.Adam(module.parameters(), lr=0.1)

        # Track energy over iterations
        energies = []
        initial_energy = None

        # Training loop
        for iteration in range(50):
            optimizer.zero_grad()
            energy = module()
            energies.append(float(energy.item()))

            if iteration == 0:
                initial_energy = energies[0]

            energy.backward()
            optimizer.step()

        # Verify energy decreased
        final_energy = energies[-1]
        assert final_energy < initial_energy, (
            f"Energy did not decrease: initial={initial_energy:.6f}, "
            f"final={final_energy:.6f}"
        )

        # Energy should be closer to ground state (-1.0) than initial state
        # For |0⟩ state, energy is +1.0; for |1⟩ state, energy is -1.0
        # We don't require exact convergence, but energy should improve
        assert final_energy < 0.0, (
            f"Final energy {final_energy:.6f} should be negative (ground state is -1.0)"
        )

    def test_training_loop_deterministic(self):
        """Test that training loop is deterministic with fixed seed."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        # Initialize with same parameters
        init_params = torch.tensor([0.1, 0.0], dtype=torch.float64)

        # Run training twice with same initialization
        energies1 = []
        module1 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params.clone(),
            gradient_method="parameter_shift",
        )
        optimizer1 = torch.optim.Adam(module1.parameters(), lr=0.1)

        for _ in range(10):
            optimizer1.zero_grad()
            energy = module1()
            energies1.append(float(energy.item()))
            energy.backward()
            optimizer1.step()

        energies2 = []
        module2 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params.clone(),
            gradient_method="parameter_shift",
        )
        optimizer2 = torch.optim.Adam(module2.parameters(), lr=0.1)

        for _ in range(10):
            optimizer2.zero_grad()
            energy = module2()
            energies2.append(float(energy.item()))
            energy.backward()
            optimizer2.step()

        # Energies should match (deterministic)
        for e1, e2 in zip(energies1, energies2):
            assert abs(e1 - e2) < 1e-10, (
                f"Training is not deterministic: {e1} != {e2}"
            )

        # Final parameters should match
        assert torch.allclose(module1.params, module2.params, atol=1e-10)

    def test_state_dict_roundtrip(self):
        """Test state_dict and load_state_dict roundtrip."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        # Create module and train for a few steps
        module1 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=torch.tensor([0.3, 0.4], dtype=torch.float64),
            gradient_method="parameter_shift",
        )

        optimizer1 = torch.optim.Adam(module1.parameters(), lr=0.1)
        for _ in range(5):
            optimizer1.zero_grad()
            energy = module1()
            energy.backward()
            optimizer1.step()

        # Get state dict
        state_dict = module1.state_dict()

        # Create new module and load state dict
        module2 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="parameter_shift",
        )
        module2.load_state_dict(state_dict)

        # Parameters should match exactly
        assert torch.allclose(module1.params, module2.params, atol=1e-10)

        # Forward pass should give same energy
        energy1 = module1()
        energy2 = module2()
        assert abs(float(energy1.item()) - float(energy2.item())) < 1e-10

    def test_training_with_different_optimizers(self):
        """Test that different optimizers work with QuantumModule."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        init_params = torch.tensor([0.1, 0.0], dtype=torch.float64)

        # Test SGD
        module_sgd = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params.clone(),
            gradient_method="parameter_shift",
        )
        optimizer_sgd = torch.optim.SGD(module_sgd.parameters(), lr=0.1)

        for _ in range(5):
            optimizer_sgd.zero_grad()
            energy = module_sgd()
            energy.backward()
            optimizer_sgd.step()

        # Test Adam
        module_adam = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params.clone(),
            gradient_method="parameter_shift",
        )
        optimizer_adam = torch.optim.Adam(module_adam.parameters(), lr=0.1)

        for _ in range(5):
            optimizer_adam.zero_grad()
            energy = module_adam()
            energy.backward()
            optimizer_adam.step()

        # Both should have changed parameters
        assert not torch.allclose(module_sgd.params, init_params, atol=1e-10)
        assert not torch.allclose(module_adam.params, init_params, atol=1e-10)

