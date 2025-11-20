"""Tests for parameter-shift gradient computation."""

import math

import pytest
import torch

from qconduit.torch.param_shift import (
    batched_parameter_shift_gradients,
    parameter_shift_gradients,
)


class TestParameterShiftGradients:
    """Tests for parameter_shift_gradients function."""

    def test_simple_cosine_function(self):
        """Test parameter-shift on a simple analytic function."""
        # Define f(θ) = cos(θ[0])
        # Analytical gradient: df/dθ = -sin(θ[0])

        def evaluate_fn(theta: torch.Tensor) -> float:
            return math.cos(float(theta[0].item()))

        theta = torch.tensor([0.3], dtype=torch.float64)
        grad = parameter_shift_gradients(evaluate_fn, theta, shift=math.pi / 2.0)

        # Expected gradient: -sin(0.3)
        expected_grad = -math.sin(0.3)

        assert grad.shape == (1,)
        assert grad.dtype == torch.float64
        assert abs(grad[0].item() - expected_grad) < 1e-8

    def test_multi_parameter_function(self):
        """Test parameter-shift on a multi-parameter function."""
        # Define f(θ) = cos(θ[0]) + sin(θ[1])
        # Gradients: df/dθ[0] = -sin(θ[0]), df/dθ[1] = cos(θ[1])

        def evaluate_fn(theta: torch.Tensor) -> float:
            return math.cos(float(theta[0].item())) + math.sin(float(theta[1].item()))

        theta = torch.tensor([0.3, 0.5], dtype=torch.float64)
        grad = parameter_shift_gradients(evaluate_fn, theta, shift=math.pi / 2.0)

        expected_grad_0 = -math.sin(0.3)
        expected_grad_1 = math.cos(0.5)

        assert grad.shape == (2,)
        assert grad.dtype == torch.float64
        assert abs(grad[0].item() - expected_grad_0) < 1e-8
        assert abs(grad[1].item() - expected_grad_1) < 1e-8

    def test_zero_gradient(self):
        """Test parameter-shift on a function with zero gradient."""
        # Define f(θ) = θ[0]^2 (but we'll evaluate at θ=0)
        # At θ=0, gradient should be 0

        def evaluate_fn(theta: torch.Tensor) -> float:
            val = float(theta[0].item())
            return val * val  # θ^2

        theta = torch.tensor([0.0], dtype=torch.float64)
        grad = parameter_shift_gradients(evaluate_fn, theta, shift=math.pi / 2.0)

        # For f(θ) = θ^2, gradient at θ=0 is 0
        # But parameter-shift with shift=π/2 gives:
        # grad = 0.5 * (f(π/2) - f(-π/2)) = 0.5 * ((π/2)^2 - (π/2)^2) = 0
        assert grad.shape == (1,)
        assert abs(grad[0].item()) < 1e-8

    def test_invalid_params_shape(self):
        """Test that invalid parameter shapes raise ValueError."""
        def evaluate_fn(theta: torch.Tensor) -> float:
            return 0.0

        # 2D tensor should raise error
        theta_2d = torch.tensor([[0.3]], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be 1D"):
            parameter_shift_gradients(evaluate_fn, theta_2d)


class TestBatchedParameterShiftGradients:
    """Tests for batched_parameter_shift_gradients function."""

    def test_batched_gradients(self):
        """Test batched gradient computation."""
        # Define f(θ) = cos(θ[0])
        def evaluate_fn(theta: torch.Tensor) -> float:
            return math.cos(float(theta[0].item()))

        # Batch of 3 parameter vectors
        theta_batch = torch.tensor(
            [[0.3], [0.5], [0.7]], dtype=torch.float64
        )

        grad_batch = batched_parameter_shift_gradients(
            evaluate_fn, theta_batch, shift=math.pi / 2.0
        )

        assert grad_batch.shape == (3, 1)
        assert grad_batch.dtype == torch.float64

        # Check each gradient
        expected_0 = -math.sin(0.3)
        expected_1 = -math.sin(0.5)
        expected_2 = -math.sin(0.7)

        assert abs(grad_batch[0, 0].item() - expected_0) < 1e-8
        assert abs(grad_batch[1, 0].item() - expected_1) < 1e-8
        assert abs(grad_batch[2, 0].item() - expected_2) < 1e-8

    def test_batched_multi_parameter(self):
        """Test batched gradients for multi-parameter functions."""
        def evaluate_fn(theta: torch.Tensor) -> float:
            return math.cos(float(theta[0].item())) + math.sin(float(theta[1].item()))

        theta_batch = torch.tensor(
            [[0.3, 0.5], [0.1, 0.2]], dtype=torch.float64
        )

        grad_batch = batched_parameter_shift_gradients(
            evaluate_fn, theta_batch, shift=math.pi / 2.0
        )

        assert grad_batch.shape == (2, 2)

        # Check first batch element
        assert abs(grad_batch[0, 0].item() - (-math.sin(0.3))) < 1e-8
        assert abs(grad_batch[0, 1].item() - math.cos(0.5)) < 1e-8

        # Check second batch element
        assert abs(grad_batch[1, 0].item() - (-math.sin(0.1))) < 1e-8
        assert abs(grad_batch[1, 1].item() - math.cos(0.2)) < 1e-8

    def test_batched_invalid_shape(self):
        """Test that invalid batch shapes raise ValueError."""
        def evaluate_fn(theta: torch.Tensor) -> float:
            return 0.0

        # 1D tensor should raise error
        theta_1d = torch.tensor([0.3], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be 2D"):
            batched_parameter_shift_gradients(evaluate_fn, theta_1d)

