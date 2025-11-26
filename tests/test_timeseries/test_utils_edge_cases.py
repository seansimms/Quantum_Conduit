"""Comprehensive edge case tests for timeseries/utils to improve coverage to ≥85%."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.utils import (
    acf,
    acov,
    difference,
    durbin_levinson,
    invert_difference,
    lag_matrix,
    pacf,
)


class TestDifferenceEdgeCases:
    """Edge case tests for difference() function."""

    def test_difference_zero_d(self):
        """Test differencing with d=0 (no differencing)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = difference(x, d=0)
        np.testing.assert_array_equal(result, x)

    def test_difference_seasonal_negative(self):
        """Test differencing with negative seasonal raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="seasonal must be >= 0"):
            difference(x, d=0, seasonal=-1)

    def test_difference_short_series_seasonal(self):
        """Test seasonal differencing with insufficient data."""
        x = np.array([1.0, 2.0])
        result = difference(x, d=0, seasonal=3)
        assert len(result) == 0

    def test_difference_combined_d_and_seasonal(self):
        """Test combined regular and seasonal differencing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = difference(x, d=1, seasonal=3)
        # Should apply d=1 first, then seasonal=3
        assert len(result) > 0

    def test_difference_empty_array(self):
        """Test differencing with empty array."""
        x = np.array([])
        result = difference(x, d=1)
        assert len(result) == 0

    def test_difference_single_element(self):
        """Test differencing with single element."""
        x = np.array([5.0])
        result = difference(x, d=1)
        assert len(result) == 0

    def test_difference_constant_series(self):
        """Test differencing constant series."""
        x = np.ones(10) * 5.0
        result = difference(x, d=1)
        np.testing.assert_allclose(result, np.zeros(9))

    def test_difference_2d_raises(self):
        """Test that 2D input raises ValueError."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            difference(x, d=1)


class TestInvertDifferenceEdgeCases:
    """Edge case tests for invert_difference() function."""

    def test_invert_difference_roundtrip_d2(self):
        """Test roundtrip with d=2."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dx = difference(x, d=2)
        x_recon = invert_difference(x, dx, d=2)
        # Should match from index 2 onwards
        np.testing.assert_allclose(x[2:], x_recon[2:], rtol=1e-10)

    def test_invert_difference_shape_mismatch(self):
        """Test invert_difference with incompatible shapes."""
        orig = np.array([1.0, 2.0, 3.0, 4.0])
        diffed = np.array([1.0, 1.0])  # Wrong length
        with pytest.raises(ValueError, match="Incompatible lengths"):
            invert_difference(orig, diffed, d=1)

    def test_invert_difference_negative_d(self):
        """Test invert_difference with negative d raises error."""
        orig = np.array([1.0, 2.0, 3.0])
        diffed = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="d must be >= 0"):
            invert_difference(orig, diffed, d=-1)

    def test_invert_difference_negative_seasonal(self):
        """Test invert_difference with negative seasonal raises error."""
        orig = np.array([1.0, 2.0, 3.0, 4.0])
        diffed = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="seasonal must be >= 0"):
            invert_difference(orig, diffed, d=0, seasonal=-1)

    def test_invert_difference_2d_orig(self):
        """Test invert_difference with 2D orig raises error."""
        orig = np.array([[1.0, 2.0], [3.0, 4.0]])
        diffed = np.array([1.0])
        with pytest.raises(ValueError, match="must be 1D"):
            invert_difference(orig, diffed, d=1)

    def test_invert_difference_2d_diffed(self):
        """Test invert_difference with 2D diffed raises error."""
        orig = np.array([1.0, 2.0, 3.0])
        diffed = np.array([[1.0], [1.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            invert_difference(orig, diffed, d=1)

    def test_invert_difference_empty(self):
        """Test invert_difference with empty arrays."""
        orig = np.array([1.0])
        diffed = np.array([])
        result = invert_difference(orig, diffed, d=1)
        assert len(result) == len(orig)


class TestLagMatrixEdgeCases:
    """Edge case tests for lag_matrix() function."""

    def test_lag_matrix_p_equals_n_minus_one(self):
        """Test lag matrix with p = n-1."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        X = lag_matrix(x, p=3)
        assert X.shape == (1, 3)
        # Row contains x[p-1-j] for j=0..p-1, so x[2], x[1], x[0]
        # Based on implementation: X[:, j] = x[p - 1 - j : n - 1 - j]
        # For j=0: x[2:3] = [3.0]
        # For j=1: x[1:2] = [2.0]
        # For j=2: x[0:1] = [1.0]
        np.testing.assert_allclose(X[0], [3.0, 2.0, 1.0])

    def test_lag_matrix_p_negative(self):
        """Test lag matrix with negative p raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="p must be >= 1"):
            lag_matrix(x, p=-1)

    def test_lag_matrix_p_zero(self):
        """Test lag matrix with p=0 raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="p must be >= 1"):
            lag_matrix(x, p=0)

    def test_lag_matrix_2d_raises(self):
        """Test lag matrix with 2D input raises error."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            lag_matrix(x, p=1)

    def test_lag_matrix_insufficient_data(self):
        """Test lag matrix with insufficient data."""
        x = np.array([1.0])
        with pytest.raises(ValueError, match="Need at least"):
            lag_matrix(x, p=2)


class TestAcovEdgeCases:
    """Edge case tests for acov() function."""

    def test_acov_nlags_zero(self):
        """Test autocovariance with nlags=0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = acov(x, nlags=0)
        assert len(gamma) == 1
        assert gamma[0] > 0  # Variance should be positive

    def test_acov_nlags_negative(self):
        """Test autocovariance with negative nlags raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nlags must be >= 0"):
            acov(x, nlags=-1)

    def test_acov_constant_series(self):
        """Test autocovariance of constant series."""
        x = np.ones(100) * 5.0
        gamma = acov(x, nlags=5)
        # All autocovariances should be zero
        np.testing.assert_allclose(gamma, np.zeros(6), atol=1e-10)

    def test_acov_2d_raises(self):
        """Test autocovariance with 2D input raises error."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            acov(x, nlags=1)

    def test_acov_single_element(self):
        """Test autocovariance with single element raises error."""
        x = np.array([5.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            acov(x, nlags=1)

    def test_acov_large_nlags(self):
        """Test autocovariance with nlags >= n-1."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = acov(x, nlags=10)  # nlags > n-1
        assert len(gamma) == 11
        # Higher lags should be zero or very small
        assert abs(gamma[-1]) < 1e-10


class TestAcfEdgeCases:
    """Edge case tests for acf() function."""

    def test_acf_constant_series(self):
        """Test ACF of constant series returns zeros."""
        x = np.ones(100) * 5.0
        rho = acf(x, nlags=5)
        # Should return zeros (variance is zero)
        np.testing.assert_allclose(rho, np.zeros(6), atol=1e-10)

    def test_acf_nlags_negative(self):
        """Test ACF with negative nlags raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nlags must be >= 0"):
            acf(x, nlags=-1)

    def test_acf_2d_raises(self):
        """Test ACF with 2D input raises error."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            acf(x, nlags=1)

    def test_acf_nlags_zero(self):
        """Test ACF with nlags=0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        rho = acf(x, nlags=0)
        assert len(rho) == 1
        assert abs(rho[0] - 1.0) < 1e-10  # ρ(0) = 1


class TestPacfEdgeCases:
    """Edge case tests for pacf() function."""

    def test_pacf_nlags_one(self):
        """Test PACF with nlags=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        pacf_vals = pacf(x, nlags=1)
        assert len(pacf_vals) == 1

    def test_pacf_nlags_negative(self):
        """Test PACF with negative nlags raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nlags must be >= 1"):
            pacf(x, nlags=-1)

    def test_pacf_nlags_zero(self):
        """Test PACF with nlags=0 raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nlags must be >= 1"):
            pacf(x, nlags=0)

    def test_pacf_constant_series(self):
        """Test PACF of constant series returns zeros."""
        x = np.ones(100) * 5.0
        pacf_vals = pacf(x, nlags=5)
        # Should return zeros (variance is zero)
        np.testing.assert_allclose(pacf_vals, np.zeros(5), atol=1e-10)

    def test_pacf_2d_raises(self):
        """Test PACF with 2D input raises error."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            pacf(x, nlags=1)

    def test_pacf_single_element(self):
        """Test PACF with single element raises error."""
        x = np.array([5.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            pacf(x, nlags=1)


class TestDurbinLevinsonEdgeCases:
    """Edge case tests for durbin_levinson() function."""

    def test_durbin_levinson_empty(self):
        """Test Durbin-Levinson with empty array raises error."""
        r = np.array([])
        with pytest.raises(ValueError, match="Need at least 2"):
            durbin_levinson(r)

    def test_durbin_levinson_single_element(self):
        """Test Durbin-Levinson with single element raises error."""
        r = np.array([1.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            durbin_levinson(r)

    def test_durbin_levinson_zero_variance(self):
        """Test Durbin-Levinson with zero variance raises error."""
        r = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="r\\[0\\] .* must be > 0"):
            durbin_levinson(r)

    def test_durbin_levinson_negative_variance(self):
        """Test Durbin-Levinson with negative variance raises error."""
        r = np.array([-1.0, 0.5])
        with pytest.raises(ValueError, match="r\\[0\\] .* must be > 0"):
            durbin_levinson(r)

    def test_durbin_levinson_2d_raises(self):
        """Test Durbin-Levinson with 2D input raises error."""
        r = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            durbin_levinson(r)

    def test_durbin_levinson_near_singular(self):
        """Test Durbin-Levinson with near-singular system."""
        # Create near-singular autocovariance (very small variance)
        r = np.array([1e-10, 0.5e-10, 0.25e-10])
        # Should use ridge regularization
        phi = durbin_levinson(r)
        assert len(phi) == 2
        assert np.all(np.isfinite(phi))

    def test_durbin_levinson_ar2(self):
        """Test Durbin-Levinson for AR(2) case."""
        # Theoretical autocovariances for AR(2) with φ₁=0.5, φ₂=0.3
        phi1, phi2 = 0.5, 0.3
        sigma2 = 1.0
        # Solve Yule-Walker to get theoretical autocovariances
        # γ(0) = σ² / (1 - φ₁*ρ(1) - φ₂*ρ(2))
        # For simplicity, use approximate values
        gamma0 = 2.0
        gamma1 = phi1 * gamma0 / (1 - phi2)
        gamma2 = phi1 * gamma1 + phi2 * gamma0
        r = np.array([gamma0, gamma1, gamma2])
        phi = durbin_levinson(r)
        assert len(phi) == 2
        # Should recover approximate AR coefficients
        assert abs(phi[0] - phi1) < 0.3  # Allow tolerance for approximation

