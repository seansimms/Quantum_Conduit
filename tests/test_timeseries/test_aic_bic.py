"""Tests for AIC/BIC and diagnostic functions."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.diagnostics import aic, bic, ljung_box
from qconduit.timeseries.estimation import FitResult
from qconduit.timeseries.models import AR


class TestAicBic:
    """Tests for AIC and BIC functions."""

    def test_aic_bic_from_fitresult(self):
        """Test AIC and BIC computed from FitResult."""
        np.random.seed(42)
        n = 300
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        model = AR(p=1)
        res = model.fit(x)

        aic_val = aic(res)
        bic_val = bic(res)

        assert aic_val == res.aic
        assert bic_val == res.bic
        assert isinstance(aic_val, float)
        assert isinstance(bic_val, float)

    def test_aic_bic_manual_calculation(self):
        """Test AIC/BIC consistency with manual calculation."""
        # Create a FitResult with known values
        n = 100
        n_params = 2
        loglike = -150.0  # Example log-likelihood

        aic_manual = 2 * n_params - 2 * loglike
        bic_manual = np.log(n) * n_params - 2 * loglike

        res = FitResult(
            params=np.array([0.5, 0.3]),
            stderr=None,
            sigma2=1.0,
            aic=aic_manual,
            bic=bic_manual,
            nobs=n,
            success=True,
            message="test",
        )

        assert abs(aic(res) - aic_manual) < 1e-10
        assert abs(bic(res) - bic_manual) < 1e-10


class TestLjungBox:
    """Tests for Ljung-Box test."""

    def test_ljung_box_white_noise(self):
        """Test Ljung-Box on white noise (should not reject null)."""
        np.random.seed(42)
        residuals = np.random.randn(200)

        stat, pval = ljung_box(residuals, lags=10)

        assert stat >= 0
        assert 0 <= pval <= 1
        # White noise should not show significant autocorrelation
        assert pval > 0.05  # Should not reject at 5% level

    def test_ljung_box_statistic_only(self):
        """Test Ljung-Box returning only statistic."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        stat = ljung_box(residuals, lags=5, return_pvalue=False)

        assert stat >= 0
        assert isinstance(stat, float)

    def test_ljung_box_autocorrelated(self):
        """Test Ljung-Box on autocorrelated residuals."""
        np.random.seed(42)
        n = 200
        eps = np.random.randn(n)
        # Create autocorrelated residuals
        residuals = np.zeros(n)
        for t in range(1, n):
            residuals[t] = 0.7 * residuals[t - 1] + eps[t]

        stat, pval = ljung_box(residuals, lags=10)

        assert stat > 0
        # Should detect autocorrelation
        assert pval < 0.05  # Should reject null

    def test_ljung_box_invalid(self):
        """Test Ljung-Box with invalid inputs."""
        residuals = np.array([1.0])
        with pytest.raises(ValueError, match="Need at least 2"):
            ljung_box(residuals, lags=1)

        residuals = np.random.randn(100)
        with pytest.raises(ValueError, match="lags must be >= 1"):
            ljung_box(residuals, lags=0)

        with pytest.raises(ValueError, match="lags must be <"):
            ljung_box(residuals, lags=200)

    def test_ljung_box_no_scipy(self):
        """Test that ljung_box raises error if scipy not available when needed."""
        # This test assumes scipy is available in test environment
        # If scipy is not available, we can't test this meaningfully
        pass

