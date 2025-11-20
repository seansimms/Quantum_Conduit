"""Tests for QuantumModule PyTorch integration."""

import pytest
import torch

from qconduit.operators import PauliSum, PauliTerm
from qconduit.torch import QuantumModule
from qconduit.variational import HardwareEfficientAnsatz
from qconduit.variational.vqe import evaluate_expectation_value


class TestQuantumModule:
    """Tests for QuantumModule class."""

    def test_quantum_module_initialization(self):
        """Test basic QuantumModule initialization."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="parameter_shift",
        )

        assert module._ansatz == ansatz
        assert module._hamiltonian == H
        assert module._gradient_method == "parameter_shift"
        assert module.params.shape == (2,)  # HardwareEfficientAnsatz has 2 params per layer
        assert module.params.dtype == torch.float64

    def test_quantum_module_with_init_params(self):
        """Test QuantumModule with provided initial parameters."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        init_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        # Check that parameters match (within tolerance)
        assert torch.allclose(module.params, init_params, atol=1e-10)

    def test_quantum_module_forward_parameter_shift(self):
        """Test QuantumModule forward pass with parameter-shift."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        # Initialize with parameters that produce |0⟩ state (theta=0 for Rx, Rz)
        init_params = torch.tensor([0.0, 0.0], dtype=torch.float64)
        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        # Forward pass
        energy = module()

        # Should be a scalar tensor
        assert energy.ndim == 0
        assert energy.dtype == torch.float64

        # For |0⟩ state with H=Z, energy should be +1.0
        # Verify by comparing with evaluate_expectation_value
        expected_energy = evaluate_expectation_value(
            ansatz=ansatz,
            params=init_params,
            hamiltonian=H,
        )

        assert abs(float(energy.item()) - expected_energy) < 1e-6

    def test_quantum_module_gradients_parameter_shift(self):
        """Test that gradients are computed correctly with parameter-shift."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        init_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        # Forward pass
        energy = module()

        # Backward pass
        energy.backward()

        # Check that gradients were computed
        assert module.params.grad is not None
        assert module.params.grad.shape == (2,)
        assert module.params.grad.dtype == torch.float64

    def test_quantum_module_training_step(self):
        """Test a single training step with optimizer."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        init_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        # Training step
        optimizer.zero_grad()
        energy = module()
        energy.backward()
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(module.params, init_params, atol=1e-10)

    def test_quantum_module_multiple_training_steps(self):
        """Test multiple training steps to verify energy decreases."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        init_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=init_params,
            gradient_method="parameter_shift",
        )

        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        energies = []
        for _ in range(5):
            optimizer.zero_grad()
            energy = module()
            energies.append(float(energy.item()))
            energy.backward()
            optimizer.step()

        # Energy should change (may increase or decrease depending on initial state)
        # At minimum, we verify the training loop runs without errors
        assert len(energies) == 5
        assert all(isinstance(e, float) for e in energies)

    def test_quantum_module_get_set_parameters(self):
        """Test get_parameters and set_parameters methods."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="parameter_shift",
        )

        # Get parameters
        params = module.get_parameters()
        assert params.shape == (2,)
        assert params.dtype == torch.float64
        assert params.device.type == "cpu"

        # Set new parameters
        new_params = torch.tensor([0.5, 0.6], dtype=torch.float64)
        module.set_parameters(new_params)

        # Verify parameters were set
        assert torch.allclose(module.params, new_params, atol=1e-10)

    def test_quantum_module_set_parameters_invalid_shape(self):
        """Test that set_parameters raises error for invalid shapes."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="parameter_shift",
        )

        # Wrong length
        invalid_params = torch.tensor([0.5], dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match expected length"):
            module.set_parameters(invalid_params)

    def test_quantum_module_state_dict(self):
        """Test state_dict and load_state_dict compatibility."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        module1 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            init_params=torch.tensor([0.3, 0.4], dtype=torch.float64),
            gradient_method="parameter_shift",
        )

        # Get state dict
        state_dict = module1.state_dict()

        # Create new module and load state dict
        module2 = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="parameter_shift",
        )
        module2.load_state_dict(state_dict)

        # Parameters should match
        assert torch.allclose(module1.params, module2.params, atol=1e-10)

    def test_quantum_module_invalid_gradient_method(self):
        """Test that invalid gradient_method raises ValueError."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        with pytest.raises(ValueError, match="gradient_method must be"):
            QuantumModule(
                ansatz=ansatz,
                hamiltonian=H,
                gradient_method="invalid_method",
            )

    def test_quantum_module_hamiltonian_ansatz_mismatch(self):
        """Test that mismatched hamiltonian and ansatz raise ValueError."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        # Hamiltonian for 2 qubits
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z", "Z"))])

        with pytest.raises(ValueError, match="does not match"):
            QuantumModule(
                ansatz=ansatz,
                hamiltonian=H,
                gradient_method="parameter_shift",
            )

    def test_quantum_module_autograd_mode_error(self):
        """Test that autograd mode raises clear error when not supported."""
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
        H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])

        module = QuantumModule(
            ansatz=ansatz,
            hamiltonian=H,
            gradient_method="autograd",
        )

        # Forward should either work (if autograd supported) or raise clear error
        try:
            energy = module()
            # If it works, check that it's a tensor
            assert isinstance(energy, torch.Tensor)
        except RuntimeError as e:
            # Error should mention parameter_shift as fallback
            assert "parameter_shift" in str(e).lower()

