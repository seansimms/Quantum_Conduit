"""Tests for timeseries utility functions."""

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


class TestDifference:
    """Tests for difference() function."""

    def test_difference_basic(self):
        """Test basic first differencing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = difference(x, d=1)
        expected = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_difference_double(self):
        """Test double differencing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = difference(x, d=2)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_difference_seasonal(self):
        """Test seasonal differencing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = difference(x, d=0, seasonal=4)
        expected = np.array([4.0, 4.0, 4.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_difference_empty(self):
        """Test differencing with insufficient data."""
        x = np.array([1.0])
        result = difference(x, d=1)
        assert len(result) == 0

    def test_difference_invalid(self):
        """Test differencing with invalid inputs."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            difference(x, d=1)

        with pytest.raises(ValueError, match="d must be >= 0"):
            difference(x.flatten(), d=-1)


class TestInvertDifference:
    """Tests for invert_difference() function."""

    def test_invert_difference_roundtrip(self):
        """Test that differencing and inversion are inverses."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dx = difference(x, d=1)
        x_recon = invert_difference(x, dx, d=1)
        # Should match original from index 1 onwards
        np.testing.assert_allclose(x[1:], x_recon[1:], rtol=1e-10)

    def test_invert_difference_seasonal(self):
        """Test inversion of seasonal differencing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        dx = difference(x, d=0, seasonal=4)
        x_recon = invert_difference(x, dx, d=0, seasonal=4)
        # Should match original from index 4 onwards
        np.testing.assert_allclose(x[4:], x_recon[4:], rtol=1e-10)


class TestLagMatrix:
    """Tests for lag_matrix() function."""

    def test_lag_matrix_basic(self):
        """Test basic lag matrix construction."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = lag_matrix(x, p=2)
        expected = np.array([[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        np.testing.assert_allclose(X, expected)

    def test_lag_matrix_p1(self):
        """Test lag matrix with p=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        X = lag_matrix(x, p=1)
        expected = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(X, expected)

    def test_lag_matrix_invalid(self):
        """Test lag matrix with invalid inputs."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Need at least"):
            lag_matrix(x, p=3)


class TestAcovAcf:
    """Tests for autocovariance and autocorrelation functions."""

    def test_acov_white_noise(self):
        """Test autocovariance of white noise."""
        np.random.seed(42)
        x = np.random.randn(1000)
        gamma = acov(x, nlags=5)
        # γ(0) should be variance (approx 1 for standard normal)
        assert abs(gamma[0] - 1.0) < 0.1
        # Higher lags should be near zero
        assert abs(gamma[1]) < 0.1

    def test_acf_ar1(self):
        """Test ACF of AR(1) process."""
        np.random.seed(42)
        n = 500
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        rho = acf(x, nlags=3)
        # ρ(0) = 1
        assert abs(rho[0] - 1.0) < 0.01
        # ρ(1) ≈ φ
        assert abs(rho[1] - phi) < 0.1
        # ρ(2) ≈ φ²
        assert abs(rho[2] - phi**2) < 0.15

    def test_acov_invalid(self):
        """Test autocovariance with invalid inputs."""
        x = np.array([1.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            acov(x, nlags=1)


class TestPacf:
    """Tests for partial autocorrelation function."""

    def test_pacf_ar1(self):
        """Test PACF of AR(1) process."""
        np.random.seed(42)
        n = 500
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        pacf_vals = pacf(x, nlags=5)
        # PACF(1) should be close to φ
        assert abs(pacf_vals[0] - phi) < 0.1
        # PACF(k) for k > 1 should be near zero
        assert abs(pacf_vals[1]) < 0.15


class TestDurbinLevinson:
    """Tests for Durbin-Levinson recursion."""

    def test_durbin_levinson_ar1(self):
        """Test Durbin-Levinson for AR(1) case."""
        # Theoretical autocovariances for AR(1) with φ=0.7, σ²=1
        # γ(0) = σ²/(1-φ²) ≈ 1.96
        # γ(k) = φ^k * γ(0)
        phi_true = 0.7
        gamma0 = 1.0 / (1 - phi_true**2)
        r = np.array([gamma0, phi_true * gamma0, phi_true**2 * gamma0])
        phi_est = durbin_levinson(r)
        assert abs(phi_est[0] - phi_true) < 0.01

    def test_durbin_levinson_invalid(self):
        """Test Durbin-Levinson with invalid inputs."""
        r = np.array([1.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            durbin_levinson(r)

        r = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="r\\[0\\] .* must be > 0"):
            durbin_levinson(r)

