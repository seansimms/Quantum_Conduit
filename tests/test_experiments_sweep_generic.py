"""Tests for generic parameter sweep utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.experiments import (
    SweepResult1D,
    SweepResult2D,
    run_1d_sweep,
    run_2d_sweep,
)


class TestSweepResult1D:
    """Tests for SweepResult1D dataclass."""

    def test_sweep_result_1d_valid(self):
        """Test SweepResult1D with valid inputs."""
        points = torch.linspace(0.0, 1.0, 5)
        values = torch.zeros(5)
        metadata = {"test": "ok"}
        res = SweepResult1D(points=points, values=values, metadata=metadata)

        assert res.points.shape == (5,)
        assert res.values.shape == (5,)
        assert res.metadata == {"test": "ok"}

    def test_sweep_result_1d_invalid_points_shape(self):
        """Test SweepResult1D raises ValueError for invalid points shape."""
        points_bad = torch.zeros((2, 2))  # 2D instead of 1D
        values_good = torch.zeros(4)

        with pytest.raises(ValueError, match="points must be 1D tensor"):
            SweepResult1D(points=points_bad, values=values_good, metadata={})

    def test_sweep_result_1d_invalid_values_shape(self):
        """Test SweepResult1D raises ValueError for invalid values shape."""
        points_good = torch.zeros(3)
        values_bad = torch.zeros((2, 2))  # 2D instead of 1D

        with pytest.raises(ValueError, match="values must be 1D tensor"):
            SweepResult1D(points=points_good, values=values_bad, metadata={})

    def test_sweep_result_1d_mismatched_lengths(self):
        """Test SweepResult1D raises ValueError for mismatched lengths."""
        points_good = torch.zeros(3)
        values_bad = torch.zeros(4)  # Different length

        with pytest.raises(ValueError, match="must have the same length"):
            SweepResult1D(points=points_good, values=values_bad, metadata={})

    def test_sweep_result_1d_metadata_copy(self):
        """Test that metadata is copied (frozen dataclass protection)."""
        points = torch.zeros(3)
        values = torch.zeros(3)
        metadata = {"key": "value"}
        res = SweepResult1D(points=points, values=values, metadata=metadata)

        # Modify original metadata
        metadata["new_key"] = "new_value"

        # Result metadata should be unchanged
        assert "new_key" not in res.metadata
        assert res.metadata == {"key": "value"}


class TestSweepResult2D:
    """Tests for SweepResult2D dataclass."""

    def test_sweep_result_2d_valid(self):
        """Test SweepResult2D with valid inputs."""
        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)
        values = torch.zeros(3, 2)
        metadata = {"test": "ok"}
        res = SweepResult2D(
            x_points=x_points, y_points=y_points, values=values, metadata=metadata
        )

        assert res.x_points.shape == (3,)
        assert res.y_points.shape == (2,)
        assert res.values.shape == (3, 2)
        assert res.metadata == {"test": "ok"}

    def test_sweep_result_2d_invalid_x_points_shape(self):
        """Test SweepResult2D raises ValueError for invalid x_points shape."""
        x_points_bad = torch.zeros((2, 2))  # 2D instead of 1D
        y_points_good = torch.zeros(2)
        values_good = torch.zeros(2, 2)

        with pytest.raises(ValueError, match="x_points must be 1D tensor"):
            SweepResult2D(
                x_points=x_points_bad,
                y_points=y_points_good,
                values=values_good,
                metadata={},
            )

    def test_sweep_result_2d_invalid_y_points_shape(self):
        """Test SweepResult2D raises ValueError for invalid y_points shape."""
        x_points_good = torch.zeros(3)
        y_points_bad = torch.zeros((2, 2))  # 2D instead of 1D
        values_good = torch.zeros(3, 2)

        with pytest.raises(ValueError, match="y_points must be 1D tensor"):
            SweepResult2D(
                x_points=x_points_good,
                y_points=y_points_bad,
                values=values_good,
                metadata={},
            )

    def test_sweep_result_2d_invalid_values_shape(self):
        """Test SweepResult2D raises ValueError for invalid values shape."""
        x_points_good = torch.zeros(3)
        y_points_good = torch.zeros(2)
        values_bad = torch.zeros(3)  # 1D instead of 2D

        with pytest.raises(ValueError, match="values must be 2D tensor"):
            SweepResult2D(
                x_points=x_points_good,
                y_points=y_points_good,
                values=values_bad,
                metadata={},
            )

    def test_sweep_result_2d_mismatched_shapes(self):
        """Test SweepResult2D raises ValueError for mismatched shapes."""
        x_points_good = torch.zeros(3)
        y_points_good = torch.zeros(2)
        values_bad = torch.zeros(3, 3)  # Wrong second dimension

        with pytest.raises(ValueError, match="values must have shape"):
            SweepResult2D(
                x_points=x_points_good,
                y_points=y_points_good,
                values=values_bad,
                metadata={},
            )


class TestRun1DSweep:
    """Tests for run_1d_sweep function."""

    def test_run_1d_sweep_simple_scalar_function(self):
        """Test run_1d_sweep on a simple analytic scalar function."""
        # Define objective: f(x) = cos(x)
        def objective(params: torch.Tensor) -> torch.Tensor:
            assert params.shape == (1,)
            return torch.cos(params[0])

        points = torch.linspace(0.0, math.pi, 4)
        res = run_1d_sweep(objective, points)

        assert res.points.shape == (4,)
        assert res.values.shape == (4,)
        assert torch.allclose(res.values, torch.cos(res.points), atol=1e-6)

    def test_run_1d_sweep_with_base_params(self):
        """Test run_1d_sweep with base_params."""
        # Define objective: f(params) = params[0]^2 + params[1]
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] ** 2 + params[1]

        base_params = torch.tensor([0.0, 1.0])
        points = torch.linspace(0.0, 2.0, 5)

        # Sweep over index 0
        res = run_1d_sweep(objective, points, base_params=base_params, index=0)

        assert res.points.shape == (5,)
        assert res.values.shape == (5,)
        # Expected: values[i] = points[i]^2 + 1.0
        expected = res.points ** 2 + 1.0
        assert torch.allclose(res.values, expected, atol=1e-6)

    def test_run_1d_sweep_with_metadata(self):
        """Test run_1d_sweep with metadata."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        points = torch.linspace(0.0, 1.0, 3)
        metadata = {"param_name": "theta", "x_label": "Parameter"}
        res = run_1d_sweep(objective, points, metadata=metadata)

        assert res.metadata == metadata

    def test_run_1d_sweep_invalid_points_2d(self):
        """Test run_1d_sweep raises ValueError for 2D points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        points_bad = torch.zeros((2, 2))  # 2D instead of 1D

        with pytest.raises(ValueError, match="points must be 1D tensor"):
            run_1d_sweep(objective, points_bad)

    def test_run_1d_sweep_empty_points(self):
        """Test run_1d_sweep raises ValueError for empty points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        points_empty = torch.tensor([])

        with pytest.raises(ValueError, match="points must be non-empty"):
            run_1d_sweep(objective, points_empty)

    def test_run_1d_sweep_invalid_base_params_index(self):
        """Test run_1d_sweep raises ValueError for invalid index."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        base_params = torch.tensor([0.0, 1.0])
        points = torch.linspace(0.0, 1.0, 3)

        with pytest.raises(ValueError, match="index must be in"):
            run_1d_sweep(objective, points, base_params=base_params, index=2)

    def test_run_1d_sweep_base_params_none_index_not_zero(self):
        """Test run_1d_sweep raises ValueError when base_params is None but index != 0."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        points = torch.linspace(0.0, 1.0, 3)

        with pytest.raises(ValueError, match="If base_params is None, index must be 0"):
            run_1d_sweep(objective, points, base_params=None, index=1)

    def test_run_1d_sweep_non_scalar_return(self):
        """Test run_1d_sweep raises ValueError for non-scalar objective return."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params  # Returns 1D tensor instead of scalar

        points = torch.linspace(0.0, 1.0, 3)

        with pytest.raises(ValueError, match="objective must return a scalar tensor"):
            run_1d_sweep(objective, points)

    def test_run_1d_sweep_detach_false(self):
        """Test run_1d_sweep with detach=False still returns detached results."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] ** 2

        points = torch.linspace(0.0, 1.0, 3, requires_grad=True)
        res = run_1d_sweep(objective, points, detach=False)

        # Results should still be detached (no gradients)
        assert not res.values.requires_grad
        assert not res.points.requires_grad

    def test_run_1d_sweep_invalid_base_params_2d(self):
        """Test run_1d_sweep raises ValueError for 2D base_params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0]

        points = torch.linspace(0.0, 1.0, 3)
        base_params_bad = torch.zeros((2, 2))  # 2D instead of 1D

        with pytest.raises(ValueError, match="base_params must be 1D tensor"):
            run_1d_sweep(objective, points, base_params=base_params_bad, index=0)


class TestRun2DSweep:
    """Tests for run_2d_sweep function."""

    def test_run_2d_sweep_simple_2d_function(self):
        """Test run_2d_sweep on a simple 2D function."""
        # Define objective: f(x, y) = x + 2.0 * y
        def objective(params: torch.Tensor) -> torch.Tensor:
            x, y = params[0], params[1]
            return x + 2.0 * y

        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(-1.0, 1.0, 2)

        res = run_2d_sweep(objective, x_points, y_points)

        assert res.values.shape == (3, 2)
        # Verify values[i, j] = x_points[i] + 2 * y_points[j]
        for i in range(3):
            for j in range(2):
                expected = x_points[i].item() + 2.0 * y_points[j].item()
                assert abs(res.values[i, j].item() - expected) < 1e-6

    def test_run_2d_sweep_with_base_params(self):
        """Test run_2d_sweep with base_params."""
        # Define objective: f(params) = params[0]^2 + params[1] + params[2]
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] ** 2 + params[1] + params[2]

        base_params = torch.tensor([0.0, 0.0, 1.0])
        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        # Sweep over indices 0 and 1
        res = run_2d_sweep(
            objective, x_points, y_points, base_params=base_params, x_index=0, y_index=1
        )

        assert res.values.shape == (3, 2)
        # Expected: values[i, j] = x_points[i]^2 + y_points[j] + 1.0
        for i in range(3):
            for j in range(2):
                expected = x_points[i].item() ** 2 + y_points[j].item() + 1.0
                assert abs(res.values[i, j].item() - expected) < 1e-6

    def test_run_2d_sweep_invalid_x_points_2d(self):
        """Test run_2d_sweep raises ValueError for 2D x_points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points_bad = torch.zeros((2, 2))  # 2D instead of 1D
        y_points_good = torch.zeros(2)

        with pytest.raises(ValueError, match="x_points must be 1D tensor"):
            run_2d_sweep(objective, x_points_bad, y_points_good)

    def test_run_2d_sweep_invalid_y_points_2d(self):
        """Test run_2d_sweep raises ValueError for 2D y_points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points_good = torch.zeros(3)
        y_points_bad = torch.zeros((2, 2))  # 2D instead of 1D

        with pytest.raises(ValueError, match="y_points must be 1D tensor"):
            run_2d_sweep(objective, x_points_good, y_points_bad)

    def test_run_2d_sweep_empty_x_points(self):
        """Test run_2d_sweep raises ValueError for empty x_points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points_empty = torch.tensor([])
        y_points_good = torch.zeros(2)

        with pytest.raises(ValueError, match="x_points must be non-empty"):
            run_2d_sweep(objective, x_points_empty, y_points_good)

    def test_run_2d_sweep_empty_y_points(self):
        """Test run_2d_sweep raises ValueError for empty y_points."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points_good = torch.zeros(3)
        y_points_empty = torch.tensor([])

        with pytest.raises(ValueError, match="y_points must be non-empty"):
            run_2d_sweep(objective, x_points_good, y_points_empty)

    def test_run_2d_sweep_invalid_base_params_index(self):
        """Test run_2d_sweep raises ValueError for invalid indices."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        base_params = torch.tensor([0.0, 1.0])
        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        with pytest.raises(ValueError, match="x_index must be in"):
            run_2d_sweep(
                objective, x_points, y_points, base_params=base_params, x_index=2, y_index=0
            )

    def test_run_2d_sweep_same_indices(self):
        """Test run_2d_sweep raises ValueError when x_index == y_index."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        base_params = torch.tensor([0.0, 1.0, 2.0])
        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        with pytest.raises(ValueError, match="x_index and y_index must be different"):
            run_2d_sweep(
                objective, x_points, y_points, base_params=base_params, x_index=0, y_index=0
            )

    def test_run_2d_sweep_non_scalar_return(self):
        """Test run_2d_sweep raises ValueError for non-scalar objective return."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params  # Returns 1D tensor instead of scalar

        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        with pytest.raises(ValueError, match="objective must return a scalar tensor"):
            run_2d_sweep(objective, x_points, y_points)

    def test_run_2d_sweep_detach_false(self):
        """Test run_2d_sweep with detach=False still returns detached results."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points = torch.linspace(0.0, 1.0, 3, requires_grad=True)
        y_points = torch.linspace(0.0, 1.0, 2, requires_grad=True)
        res = run_2d_sweep(objective, x_points, y_points, detach=False)

        # Results should still be detached (no gradients)
        assert not res.values.requires_grad
        assert not res.x_points.requires_grad
        assert not res.y_points.requires_grad

    def test_run_2d_sweep_invalid_base_params_2d(self):
        """Test run_2d_sweep raises ValueError for 2D base_params."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)
        base_params_bad = torch.zeros((2, 2))  # 2D instead of 1D

        with pytest.raises(ValueError, match="base_params must be 1D tensor"):
            run_2d_sweep(
                objective, x_points, y_points, base_params=base_params_bad, x_index=0, y_index=1
            )

    def test_run_2d_sweep_base_params_none_invalid_indices(self):
        """Test run_2d_sweep raises ValueError when base_params is None but indices are not 0,1."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        with pytest.raises(ValueError, match="If base_params is None"):
            run_2d_sweep(objective, x_points, y_points, base_params=None, x_index=0, y_index=2)

    def test_run_2d_sweep_invalid_y_index(self):
        """Test run_2d_sweep raises ValueError for invalid y_index."""
        def objective(params: torch.Tensor) -> torch.Tensor:
            return params[0] + params[1]

        base_params = torch.tensor([0.0, 1.0])
        x_points = torch.linspace(0.0, 1.0, 3)
        y_points = torch.linspace(0.0, 1.0, 2)

        with pytest.raises(ValueError, match="y_index must be in"):
            run_2d_sweep(
                objective, x_points, y_points, base_params=base_params, x_index=0, y_index=2
            )

