"""Tests for adiabatic schedule functions."""

from __future__ import annotations

import pytest
import torch

from qconduit.adiabatic import (
    linear_schedule,
    polynomial_schedule,
    sample_schedule,
)


def test_linear_schedule_basic():
    """Test basic linear schedule properties."""
    sched = linear_schedule(num_steps=5)

    assert sched.shape == (5,)
    assert sched[0].item() == pytest.approx(0.0, abs=1e-10)
    assert sched[-1].item() == pytest.approx(1.0, abs=1e-10)

    # Check monotonic non-decreasing
    assert torch.all(sched[1:] >= sched[:-1])

    # Check values match expected [0.0, 0.25, 0.5, 0.75, 1.0]
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    assert torch.allclose(sched, expected, atol=1e-10)


def test_linear_schedule_invalid_num_steps():
    """Test that linear_schedule raises ValueError for invalid num_steps."""
    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        linear_schedule(1)

    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        linear_schedule(0)

    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        linear_schedule(-1)


def test_linear_schedule_device():
    """Test that linear_schedule respects device parameter."""
    device = torch.device("cpu")
    sched = linear_schedule(num_steps=3, device=device)
    assert sched.device == device


def test_polynomial_schedule_power_two():
    """Test polynomial schedule with power=2.0."""
    sched = polynomial_schedule(num_steps=5, power=2.0)

    assert sched.shape == (5,)
    assert sched[0].item() == pytest.approx(0.0, abs=1e-10)
    assert sched[-1].item() == pytest.approx(1.0, abs=1e-10)

    # Check monotonic non-decreasing
    assert torch.all(sched[1:] >= sched[:-1])

    # Compare with linear schedule squared
    linear = linear_schedule(num_steps=5)
    expected = linear.pow(2.0)
    assert torch.allclose(sched, expected, atol=1e-10)


def test_polynomial_schedule_power_three():
    """Test polynomial schedule with power=3.0."""
    sched = polynomial_schedule(num_steps=4, power=3.0)

    assert sched.shape == (4,)
    assert sched[0].item() == pytest.approx(0.0, abs=1e-10)
    assert sched[-1].item() == pytest.approx(1.0, abs=1e-10)

    # Check monotonic
    assert torch.all(sched[1:] >= sched[:-1])

    # Compare with linear schedule cubed
    linear = linear_schedule(num_steps=4)
    expected = linear.pow(3.0)
    assert torch.allclose(sched, expected, atol=1e-10)


def test_polynomial_schedule_invalid_power():
    """Test that polynomial_schedule raises ValueError for invalid power."""
    with pytest.raises(ValueError, match="power must be >= 1.0"):
        polynomial_schedule(num_steps=5, power=0.5)

    with pytest.raises(ValueError, match="power must be >= 1.0"):
        polynomial_schedule(num_steps=5, power=0.0)

    with pytest.raises(ValueError, match="power must be >= 1.0"):
        polynomial_schedule(num_steps=5, power=-1.0)


def test_polynomial_schedule_invalid_num_steps():
    """Test that polynomial_schedule raises ValueError for invalid num_steps."""
    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        polynomial_schedule(num_steps=1, power=2.0)


def test_polynomial_schedule_power_one():
    """Test that power=1.0 gives linear schedule."""
    sched_poly = polynomial_schedule(num_steps=5, power=1.0)
    sched_linear = linear_schedule(num_steps=5)

    assert torch.allclose(sched_poly, sched_linear, atol=1e-10)


def test_sample_schedule_valid():
    """Test sample_schedule with a valid schedule function."""

    def my_schedule(t: torch.Tensor) -> torch.Tensor:
        return t**3

    s = sample_schedule(my_schedule, num_steps=4)

    assert s.shape == (4,)
    assert s[0].item() == pytest.approx(0.0, abs=1e-6)
    assert s[-1].item() == pytest.approx(1.0, abs=1e-6)

    # Should equal linear_schedule cubed
    linear = linear_schedule(num_steps=4)
    expected = linear.pow(3.0)
    assert torch.allclose(s, expected, atol=1e-10)


def test_sample_schedule_wrong_shape():
    """Test that sample_schedule raises ValueError for wrong output shape."""

    def bad_schedule(t: torch.Tensor) -> torch.Tensor:
        return t[:-1]  # Wrong length

    with pytest.raises(ValueError, match="schedule_fn must return a tensor of shape"):
        sample_schedule(bad_schedule, num_steps=4)


def test_sample_schedule_bad_values():
    """Test that sample_schedule raises ValueError for values outside [0,1]."""

    def bad_values(t: torch.Tensor) -> torch.Tensor:
        return t * 2.0  # Values > 1

    with pytest.raises(ValueError, match="schedule_fn must return values in \\[0, 1\\]"):
        sample_schedule(bad_values, num_steps=4)


def test_sample_schedule_bad_endpoint_zero():
    """Test that sample_schedule raises ValueError for wrong s(0)."""

    def bad_endpoint(t: torch.Tensor) -> torch.Tensor:
        return t + 0.1  # s(0) != 0, and values > 1

    # The function will fail on the range check first
    with pytest.raises(ValueError, match="schedule_fn must return values in"):
        sample_schedule(bad_endpoint, num_steps=4)


def test_sample_schedule_bad_endpoint_one():
    """Test that sample_schedule raises ValueError for wrong s(1)."""

    def bad_endpoint(t: torch.Tensor) -> torch.Tensor:
        return t * 0.9  # s(1) != 1

    with pytest.raises(ValueError, match="schedule_fn must return s\\(1\\)"):
        sample_schedule(bad_endpoint, num_steps=4)


def test_sample_schedule_invalid_num_steps():
    """Test that sample_schedule raises ValueError for invalid num_steps."""

    def my_schedule(t: torch.Tensor) -> torch.Tensor:
        return t

    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        sample_schedule(my_schedule, num_steps=1)


def test_sample_schedule_negative_values():
    """Test that sample_schedule raises ValueError for negative values."""

    def negative_values(t: torch.Tensor) -> torch.Tensor:
        return t - 0.1  # Some values < 0

    with pytest.raises(ValueError, match="schedule_fn must return values in \\[0, 1\\]"):
        sample_schedule(negative_values, num_steps=4)

