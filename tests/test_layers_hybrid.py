"""Tests for hybrid quantum-classical blocks."""

import pytest
import torch

from qconduit.layers.hybrid import QuantumBlock


class TestQuantumBlock:
    """Tests for QuantumBlock."""

    def test_quantum_block_initialization(self):
        """Test QuantumBlock can be initialized."""
        block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        assert block.n_qubits == 2
        assert block.in_features == 3
        assert block.ansatz.n_qubits == 2
        assert block.ansatz.depth == 1
        assert block.encoder.in_features == 3
        assert block.encoder.out_features == block.ansatz.num_parameters

    def test_quantum_block_invalid_in_features(self):
        """Test QuantumBlock raises for invalid in_features."""
        with pytest.raises(ValueError, match="in_features must be >= 1"):
            QuantumBlock(n_qubits=2, depth=1, in_features=0)

    def test_quantum_block_forward_shape(self):
        """Test QuantumBlock forward pass produces correct output shape."""
        block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        batch_size = 4
        x = torch.randn(batch_size, 3)
        output = block(x)

        assert output.shape == (batch_size, 2)  # (batch_size, n_qubits)
        assert output.dtype == torch.float32

    def test_quantum_block_forward_wrong_input_shape(self):
        """Test QuantumBlock raises for wrong input shape."""
        block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        x = torch.randn(4, 5)  # Wrong last dimension
        with pytest.raises(ValueError, match="Expected x.shape"):
            block(x)

    def test_quantum_block_forward_output_range(self):
        """Test QuantumBlock output is in valid range for Z expectations."""
        block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        x = torch.randn(4, 3)
        output = block(x)

        # Z expectation values should be in [-1, 1]
        assert torch.all(output >= -1.0 - 1e-6)
        assert torch.all(output <= 1.0 + 1e-6)

    def test_quantum_block_gradient(self):
        """Test that gradients flow through QuantumBlock to encoder parameters."""
        block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        x = torch.randn(4, 3)
        target = torch.randn(4, 2)

        # Compute MSE loss
        output = block(x)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check that encoder parameters have gradients
        assert block.encoder.weight.grad is not None
        assert block.encoder.bias.grad is not None

        # Check gradients are finite
        assert torch.isfinite(block.encoder.weight.grad).all()
        assert torch.isfinite(block.encoder.bias.grad).all()

        # Check gradient shapes
        assert block.encoder.weight.grad.shape == block.encoder.weight.shape
        assert block.encoder.bias.grad.shape == block.encoder.bias.shape

    def test_quantum_block_device_consistency(self):
        """Test QuantumBlock uses correct device."""
        block = QuantumBlock(n_qubits=1, depth=1, in_features=2, device="sv_cpu")
        x = torch.randn(2, 2)
        output = block(x)
        assert output.device.type == "cpu"

    def test_quantum_block_differentiable_end_to_end(self):
        """Test QuantumBlock is fully differentiable end-to-end."""
        block = QuantumBlock(n_qubits=2, depth=2, in_features=4)
        x = torch.randn(3, 4, requires_grad=True)

        output = block(x)
        # Sum of outputs as a simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_quantum_block_multiple_forward_calls(self):
        """Test QuantumBlock can be called multiple times."""
        block = QuantumBlock(n_qubits=1, depth=1, in_features=2)
        x1 = torch.randn(2, 2)
        x2 = torch.randn(3, 2)

        output1 = block(x1)
        output2 = block(x2)

        assert output1.shape == (2, 1)
        assert output2.shape == (3, 1)

