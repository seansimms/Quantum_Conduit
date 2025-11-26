"""Additional edge case tests for estimation functions to improve coverage."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.estimation import (
    FitResult,
    conditional_sum_squares_arma,
    ols_ar,
    yule_walker,
)
from qconduit.timeseries.models import AR, ARMA


class TestEstimationEdgeCases:
    """Tests for edge cases in estimation functions."""

    def test_yule_walker_near_zero_variance(self):
        """Test Yule-Walker with near-zero variance."""
        x = np.ones(100) * 0.5  # Constant series
        with pytest.raises(ValueError, match="near-zero variance"):
            yule_walker(x, p=1)

    def test_ols_ar_with_constant_term(self):
        """Test OLS AR with constant term."""
        np.random.seed(42)
        n = 200
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 1.0 + 0.7 * x[t - 1] + eps[t]  # With intercept

        res = ols_ar(x, p=1, trend="c")
        assert res.success
        assert len(res.params) == 1  # AR params only

    def test_ols_ar_insufficient_data(self):
        """Test OLS AR with insufficient data."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Need at least"):
            ols_ar(x, p=2)

    def test_css_arma_without_scipy(self):
        """Test CSS ARMA without scipy (should use fallback)."""
        np.random.seed(42)
        n = 200
        eps = np.random.normal(size=n + 1)
        x = eps[1:] + 0.3 * eps[:-1]

        # This will use CSS fallback if scipy not available
        params, success, _ = conditional_sum_squares_arma(
            x, ar_order=0, ma_order=1, method="css"
        )
        assert len(params) == 1

    def test_css_arma_invalid_orders(self):
        """Test CSS ARMA with invalid orders."""
        x = np.random.randn(100)
        with pytest.raises(ValueError, match="At least one"):
            conditional_sum_squares_arma(x, ar_order=0, ma_order=0)

        with pytest.raises(ValueError, match="Need at least"):
            conditional_sum_squares_arma(x, ar_order=50, ma_order=50)

    def test_fitresult_attributes(self):
        """Test FitResult dataclass attributes."""
        res = FitResult(
            params=np.array([0.5]),
            stderr=np.array([0.1]),
            sigma2=1.0,
            aic=100.0,
            bic=105.0,
            nobs=100,
            success=True,
            message="test",
        )
        assert res.params[0] == 0.5
        assert res.success


class TestModelsEdgeCases:
    """Tests for edge cases in model classes."""

    def test_ar_fit_with_constant_trend(self):
        """Test AR fitting with constant trend option."""
        np.random.seed(42)
        n = 200
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + eps[t]

        model = AR(p=1, trend="c", method="ols")
        res = model.fit(x)
        assert res.success

    def test_arma_not_fitted_error(self):
        """Test that unfitted models raise appropriate errors."""
        model = ARMA(p=1, q=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(steps=10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.simulate(nsim=100)

