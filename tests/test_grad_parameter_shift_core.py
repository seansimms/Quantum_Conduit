"""Tests for core parameter-shift gradient utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.grad import (
    ParameterShiftRule,
    autograd_gradient,
    parameter_shift_gradient,
    parameter_shift_single,
)


class TestParameterShiftRule:
    """Tests for ParameterShiftRule dataclass."""

    def test_default_rule(self):
        """Test default parameter-shift rule."""
        from qconduit.grad.parameter_shift import default_parameter_shift_rule

        rule = default_parameter_shift_rule()
        assert rule.shift == pytest.approx(math.pi / 2)
        assert rule.prefactor == 0.5

    def test_custom_rule(self):
        """Test creating custom parameter-shift rule."""
        rule = ParameterShiftRule(shift=1.0, prefactor=0.25)
        assert rule.shift == 1.0
        assert rule.prefactor == 0.25

    def test_rule_immutable(self):
        """Test that ParameterShiftRule is immutable."""
        rule = ParameterShiftRule(shift=1.0, prefactor=0.5)
        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
            rule.shift = 2.0  # type: ignore


class TestParameterShiftSingle:
    """Tests for parameter_shift_single function."""

    def test_parameter_shift_single_vs_analytic_cos(self):
        """Test parameter_shift_single vs analytic derivative for cos function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(θ) = cos(θ)."""
            assert params.shape == (1,)
            return torch.cos(params[0])

        theta = 0.7
        params = torch.tensor([theta], dtype=torch.float64)

        # Compute gradient via parameter shift
        grad_ps = parameter_shift_single(objective, params, index=0)

        # Analytic derivative: d/dθ cos(θ) = -sin(θ)
        grad_true = -math.sin(theta)

        assert abs(grad_ps - grad_true) < 1e-6

    def test_parameter_shift_single_multi_param(self):
        """Test parameter_shift_single with multi-parameter function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = cos(x) + 3*sin(y)."""
            # Use functions compatible with parameter-shift rule (cos/sin)
            x, y = params[0], params[1]
            return torch.cos(x) + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        # Test gradient w.r.t. x: ∂/∂x = -sin(x)
        grad_x = parameter_shift_single(objective, params, index=0)
        grad_x_true = -math.sin(0.3)
        assert abs(grad_x - grad_x_true) < 1e-6

        # Test gradient w.r.t. y: ∂/∂y = 3*cos(y)
        grad_y = parameter_shift_single(objective, params, index=1)
        grad_y_true = 3.0 * math.cos(-0.4)
        assert abs(grad_y - grad_y_true) < 1e-6

    def test_parameter_shift_single_custom_rule(self):
        """Test parameter_shift_single with custom shift rule."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return torch.cos(params[0])

        params = torch.tensor([1.0], dtype=torch.float64)
        # Use a smaller shift for finite difference approximation
        rule = ParameterShiftRule(shift=0.1, prefactor=0.5)

        # For f(x) = cos(x), gradient at x=1 is -sin(1)
        # With finite shift 0.1: 0.5 * (cos(1.1) - cos(0.9)) ≈ -sin(1)
        grad = parameter_shift_single(objective, params, index=0, rule=rule)
        # Using central difference approximation
        expected = 0.5 * (math.cos(1.1) - math.cos(0.9))
        # Should match the formula exactly (this is what parameter_shift_single computes)
        assert abs(grad - expected) < 1e-6

    def test_parameter_shift_single_error_ndim(self):
        """Test parameter_shift_single raises for non-1D params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params.sum()

        params_2d = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            parameter_shift_single(objective, params_2d, index=0)

    def test_parameter_shift_single_error_index_out_of_range(self):
        """Test parameter_shift_single raises for index out of range."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        params = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="out of range"):
            parameter_shift_single(objective, params, index=1)

    def test_parameter_shift_single_error_non_scalar_output(self):
        """Test parameter_shift_single raises for non-scalar objective output."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params  # Returns vector, not scalar

        params = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="must return a scalar tensor"):
            parameter_shift_single(objective, params, index=0)


class TestParameterShiftGradient:
    """Tests for parameter_shift_gradient function."""

    def test_parameter_shift_gradient_multi_param(self):
        """Test parameter_shift_gradient on multi-parameter function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = cos(x) + 3*sin(y)."""
            # Use functions compatible with parameter-shift rule
            x, y = params[0], params[1]
            return torch.cos(x) + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        # Compute full gradient
        grad = parameter_shift_gradient(objective, params, indices=None)

        # True gradient: [-sin(x), 3*cos(y)] = [-sin(0.3), 3*cos(-0.4)]
        assert grad.shape == (2,)
        assert abs(grad[0].item() - (-math.sin(0.3))) < 1e-6
        assert abs(grad[1].item() - 3.0 * math.cos(-0.4)) < 1e-6

    def test_parameter_shift_gradient_subset_indices(self):
        """Test parameter_shift_gradient with subset of indices."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = x² + 3*sin(y)."""
            x, y = params[0], params[1]
            return x**2 + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        # Compute gradient only for y (index 1)
        grad = parameter_shift_gradient(objective, params, indices=[1])

        assert grad.shape == (1,)
        grad_y_true = 3.0 * math.cos(-0.4)
        assert abs(grad[0].item() - grad_y_true) < 1e-6

    def test_parameter_shift_gradient_reordered_indices(self):
        """Test parameter_shift_gradient with reordered indices."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = cos(x) + 3*sin(y)."""
            x, y = params[0], params[1]
            return torch.cos(x) + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        # Compute gradient with indices in reverse order
        grad = parameter_shift_gradient(objective, params, indices=[1, 0])

        assert grad.shape == (2,)
        # First entry should be gradient w.r.t. y
        assert abs(grad[0].item() - 3.0 * math.cos(-0.4)) < 1e-6
        # Second entry should be gradient w.r.t. x
        assert abs(grad[1].item() - (-math.sin(0.3))) < 1e-6

    def test_parameter_shift_gradient_error_ndim(self):
        """Test parameter_shift_gradient raises for non-1D params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params.sum()

        params_2d = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            parameter_shift_gradient(objective, params_2d)

    def test_parameter_shift_gradient_error_empty_params(self):
        """Test parameter_shift_gradient raises for empty params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        params = torch.tensor([], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be non-empty"):
            parameter_shift_gradient(objective, params)

    def test_parameter_shift_gradient_error_empty_indices(self):
        """Test parameter_shift_gradient raises for empty indices."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        params = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be non-empty"):
            parameter_shift_gradient(objective, params, indices=[])

    def test_parameter_shift_gradient_error_invalid_index(self):
        """Test parameter_shift_gradient raises for invalid index."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        params = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="out of range"):
            parameter_shift_gradient(objective, params, indices=[1])


class TestAutogradGradient:
    """Tests for autograd_gradient function."""

    def test_autograd_gradient_vs_analytic_cos(self):
        """Test autograd_gradient vs analytic derivative for cos function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(θ) = cos(θ)."""
            return torch.cos(params[0])

        theta = 0.7
        params = torch.tensor([theta], dtype=torch.float64)

        # Compute gradient via autograd
        grad_auto = autograd_gradient(objective, params)

        # Analytic derivative: d/dθ cos(θ) = -sin(θ)
        grad_true = -math.sin(theta)

        assert abs(grad_auto[0].item() - grad_true) < 1e-6

    def test_autograd_gradient_multi_param(self):
        """Test autograd_gradient with multi-parameter function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = x² + 3*sin(y)."""
            x, y = params[0], params[1]
            return x**2 + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        grad = autograd_gradient(objective, params)

        assert grad.shape == (2,)
        assert abs(grad[0].item() - 0.6) < 1e-6
        assert abs(grad[1].item() - 3.0 * math.cos(-0.4)) < 1e-6

    def test_autograd_gradient_error_ndim(self):
        """Test autograd_gradient raises for non-1D params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params.sum()

        params_2d = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="must be a 1D tensor"):
            autograd_gradient(objective, params_2d)

    def test_autograd_gradient_error_non_scalar_output(self):
        """Test autograd_gradient raises for non-scalar objective output."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params  # Returns vector, not scalar

        params = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="must return a scalar tensor"):
            autograd_gradient(objective, params)


class TestParameterShiftVsAutograd:
    """Tests comparing parameter-shift and autograd gradients."""

    def test_parameter_shift_vs_autograd_simple(self):
        """Test parameter-shift vs autograd for simple function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: f(x, y) = cos(x) + 3*sin(y)."""
            # Use functions compatible with parameter-shift rule
            x, y = params[0], params[1]
            return torch.cos(x) + 3.0 * torch.sin(y)

        params = torch.tensor([0.3, -0.4], dtype=torch.float64)

        # Compute gradients
        grad_ps = parameter_shift_gradient(objective, params)
        grad_auto = autograd_gradient(objective, params)

        # Should agree to high precision (parameter-shift is exact for cos/sin)
        assert torch.allclose(grad_ps, grad_auto, atol=1e-6, rtol=1e-6)

    def test_parameter_shift_vs_autograd_cos(self):
        """Test parameter-shift vs autograd for cos function."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return torch.cos(params[0])

        params = torch.tensor([0.7], dtype=torch.float64)

        grad_ps = parameter_shift_gradient(objective, params)
        grad_auto = autograd_gradient(objective, params)

        assert torch.allclose(grad_ps, grad_auto, atol=1e-6, rtol=1e-6)

