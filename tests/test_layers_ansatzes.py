"""Tests for parametric ansätze."""

import math

import pytest
import torch

from qconduit.backend.statevector import measure_probs
from qconduit.layers.ansatzes import HardwareEfficientAnsatz, ParametricAnsatz


class TestParametricAnsatz:
    """Tests for ParametricAnsatz base class."""

    def test_parametric_ansatz_cannot_instantiate_directly(self):
        """Test that ParametricAnsatz cannot be used without overriding forward."""
        ansatz = ParametricAnsatz(n_qubits=1)
        params = torch.tensor([0.0])
        with pytest.raises(NotImplementedError, match="must be implemented by subclasses"):
            ansatz(params)

    def test_parametric_ansatz_initialization(self):
        """Test ParametricAnsatz can be initialized."""
        ansatz = ParametricAnsatz(n_qubits=2)
        assert ansatz.n_qubits == 2
        assert ansatz.device.name == "sv_cpu"


class TestHardwareEfficientAnsatz:
    """Tests for HardwareEfficientAnsatz."""

    def test_hardware_efficient_ansatz_initialization(self):
        """Test HardwareEfficientAnsatz can be initialized."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=3)
        assert ansatz.n_qubits == 2
        assert ansatz.depth == 3
        assert ansatz.num_parameters == 6  # 2 qubits * 3 depth

    def test_hardware_efficient_ansatz_invalid_n_qubits(self):
        """Test HardwareEfficientAnsatz raises for invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            HardwareEfficientAnsatz(n_qubits=0, depth=1)

    def test_hardware_efficient_ansatz_invalid_depth(self):
        """Test HardwareEfficientAnsatz raises for invalid depth."""
        with pytest.raises(ValueError, match="depth must be >= 1"):
            HardwareEfficientAnsatz(n_qubits=2, depth=0)

    def test_hardware_efficient_ansatz_num_parameters(self):
        """Test num_parameters is computed correctly."""
        ansatz1 = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        assert ansatz1.num_parameters == 1

        ansatz2 = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        assert ansatz2.num_parameters == 4

        ansatz3 = HardwareEfficientAnsatz(n_qubits=3, depth=5)
        assert ansatz3.num_parameters == 15

    def test_hardware_efficient_ansatz_1_qubit_1_layer_theta_zero(self):
        """Test 1-qubit, 1-layer ansatz with theta=0 produces |0⟩."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        params = torch.tensor([0.0])
        state = ansatz(params)

        # Should be |0⟩ state
        probs = measure_probs(state, n_qubits=1)
        assert torch.allclose(probs[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(0.0), atol=1e-6)

    def test_hardware_efficient_ansatz_1_qubit_1_layer_theta_pi(self):
        """Test 1-qubit, 1-layer ansatz with theta=π produces |1⟩."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        params = torch.tensor([math.pi])
        state = ansatz(params)

        # Should be |1⟩ state (up to global phase)
        probs = measure_probs(state, n_qubits=1)
        assert torch.allclose(probs[0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(1.0), atol=1e-6)

    def test_hardware_efficient_ansatz_batch_mode(self):
        """Test ansatz works in batch mode."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        # Batch of 2: first with theta=0, second with theta=π
        params = torch.tensor([[0.0], [math.pi]])
        state = ansatz(params)

        assert state.shape == (2, 2)  # (batch_size, 2**n_qubits)

        # First batch element should be |0⟩
        probs0 = measure_probs(state[0:1], n_qubits=1)
        assert torch.allclose(probs0[0, 0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(probs0[0, 1], torch.tensor(0.0), atol=1e-6)

        # Second batch element should be |1⟩
        probs1 = measure_probs(state[1:2], n_qubits=1)
        assert torch.allclose(probs1[0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(probs1[0, 1], torch.tensor(1.0), atol=1e-6)

    def test_hardware_efficient_ansatz_wrong_num_parameters(self):
        """Test ansatz raises for wrong number of parameters."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        # Should have 4 parameters, but provide 3
        params = torch.tensor([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Expected params.shape"):
            ansatz(params)

    def test_hardware_efficient_ansatz_gradient(self):
        """Test that gradients flow through the ansatz."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
        params = torch.tensor([0.1, 0.2], requires_grad=True)

        # Compute a simple loss
        state = ansatz(params)
        loss = state.abs().pow(2).sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are finite
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        assert params.grad.shape == params.shape

    def test_hardware_efficient_ansatz_2_qubits_entanglement(self):
        """Test that 2-qubit ansatz creates entanglement."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
        # Use parameters that should create a Bell-like state
        params = torch.tensor([math.pi / 2, math.pi / 2])
        state = ansatz(params)

        # Check that state is not a product state
        # For a product state, we'd have |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩
        # An entangled state will have non-zero amplitudes for multiple basis states
        probs = measure_probs(state, n_qubits=2)
        # At least two basis states should have non-zero probability
        non_zero_probs = (probs > 1e-6).sum().item()
        assert non_zero_probs >= 2, "State should be entangled"

    def test_hardware_efficient_ansatz_device_consistency(self):
        """Test ansatz uses correct device."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1, device="sv_cpu")
        params = torch.tensor([0.0])
        state = ansatz(params)
        assert state.device.type == "cpu"

