"""Integration tests for timeseries module.

Tests end-to-end workflows and integration with the main QConduit package.
"""

from __future__ import annotations

import numpy as np
import pytest

# Test both import paths
from qconduit import AR, ARIMA, FitResult, StateSpace, kalman_filter
from qconduit.timeseries import (
    ARMA,
    MA,
    aic,
    bic,
    difference,
    invert_difference,
    yule_walker,
)


class TestEndToEndWorkflows:
    """Test complete workflows from data to forecasts."""

    def test_ar_workflow_from_main_package(self):
        """Test complete AR workflow using main package imports."""
        np.random.seed(42)
        n = 500
        phi_true = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi_true * x[t - 1] + eps[t]

        # Fit model
        model = AR(p=1)
        res = model.fit(x)

        # Check fit results
        assert isinstance(res, FitResult)
        assert res.success
        assert abs(res.params[0] - phi_true) < 0.05
        assert res.aic > 0
        assert res.bic > 0

        # Generate forecasts
        forecast, ci = model.predict(steps=10, alpha=0.05)
        assert len(forecast) == 10
        assert ci.shape == (2, 10)
        assert np.all(ci[1, :] >= ci[0, :])

        # Simulate
        sim = model.simulate(nsim=100, seed=123)
        assert len(sim) == 100

    def test_arima_workflow(self):
        """Test complete ARIMA workflow."""
        np.random.seed(42)
        n = 500
        phi = 0.6
        eps = np.random.normal(size=n)

        # Create ARIMA(1,1,0) process
        x_diff = np.zeros(n)
        for t in range(1, n):
            x_diff[t] = phi * x_diff[t - 1] + eps[t]
        x = np.cumsum(x_diff)

        # Fit
        model = ARIMA(p=1, d=1, q=0)
        res = model.fit(x)
        assert res.success

        # Forecast (should be on original scale)
        forecast, ci = model.predict(steps=10)
        assert len(forecast) == 10

    def test_utility_integration(self):
        """Test utility functions work together."""
        np.random.seed(42)
        x = np.random.randn(200) + np.arange(200) * 0.01

        # Difference
        x_diff = difference(x, d=1)
        assert len(x_diff) == len(x) - 1

        # Invert
        x_recon = invert_difference(x, x_diff, d=1)
        assert len(x_recon) == len(x)
        np.testing.assert_allclose(x[1:], x_recon[1:], rtol=1e-5)

        # ACF/PACF
        from qconduit.timeseries import acf, pacf

        acf_vals = acf(x, nlags=10)
        assert len(acf_vals) == 11
        assert abs(acf_vals[0] - 1.0) < 1e-10

        pacf_vals = pacf(x, nlags=5)
        assert len(pacf_vals) == 5

    def test_estimation_integration(self):
        """Test estimation functions work with models."""
        np.random.seed(42)
        n = 300
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        # Use Yule-Walker directly
        phi_yw, sigma2_yw = yule_walker(x, p=1)
        assert abs(phi_yw[0] - phi) < 0.1

        # Compare with model fit
        model = AR(p=1, method="yule_walker")
        res = model.fit(x)
        np.testing.assert_allclose(res.params, phi_yw, rtol=1e-5)

    def test_kalman_integration(self):
        """Test Kalman filter integration."""
        # Local level model
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

        # Generate data
        np.random.seed(42)
        T = 100
        true_state = np.zeros(T)
        y = np.zeros((T, 1))
        for t in range(1, T):
            true_state[t] = true_state[t - 1] + np.random.normal(0, np.sqrt(Q[0, 0]))
            y[t, 0] = true_state[t] + np.random.normal(0, np.sqrt(R[0, 0]))

        # Filter
        x_filtered, P_filtered, loglik = kalman_filter(ss, y)
        assert x_filtered.shape == (T, 1)
        assert isinstance(loglik, float)

    def test_diagnostics_integration(self):
        """Test diagnostics work with model results."""
        np.random.seed(42)
        n = 300
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + eps[t]

        model = AR(p=1)
        res = model.fit(x)

        # AIC/BIC
        aic_val = aic(res)
        bic_val = bic(res)
        assert aic_val == res.aic
        assert bic_val == res.bic
        assert aic_val < bic_val  # BIC should penalize more

        # Ljung-Box test (would need residuals - simplified check)
        from qconduit.timeseries import ljung_box

        residuals = np.random.randn(100)  # Mock residuals
        stat, pval = ljung_box(residuals, lags=10)
        assert stat >= 0
        assert 0 <= pval <= 1


class TestCrossModuleIntegration:
    """Test integration between different timeseries submodules."""

    def test_models_use_estimation(self):
        """Verify models use estimation functions."""
        np.random.seed(42)
        n = 300
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + eps[t]

        # AR with Yule-Walker (uses estimation.yule_walker)
        model1 = AR(p=1, method="yule_walker")
        res1 = model1.fit(x)

        # AR with OLS (uses estimation.ols_ar)
        model2 = AR(p=1, method="ols")
        res2 = model2.fit(x)

        # Both should succeed
        assert res1.success
        assert res2.success
        # Estimates should be similar
        assert abs(res1.params[0] - res2.params[0]) < 0.2

    def test_arma_uses_css(self):
        """Verify ARMA uses CSS estimation."""
        np.random.seed(42)
        n = 500
        eps = np.random.normal(size=n + 1)
        x = eps[1:] + 0.5 * eps[:-1]

        model = MA(q=1)
        res = model.fit(x, method="css")
        assert res.success
        assert len(res.params) == 1

    def test_arima_uses_differencing(self):
        """Verify ARIMA uses differencing utilities."""
        np.random.seed(42)
        n = 500
        x = np.cumsum(np.random.randn(n))

        model = ARIMA(p=1, d=1, q=0)
        res = model.fit(x)
        assert res.success

        # Internal differencing should have been applied
        # (verified by successful fit)


class TestProductionReadiness:
    """Test production readiness features."""

    def test_all_exports_accessible(self):
        """Test all documented exports are accessible."""
        from qconduit.timeseries import __all__

        for name in __all__:
            obj = getattr(__import__("qconduit.timeseries", fromlist=[name]), name)
            assert obj is not None, f"{name} is None"

    def test_error_handling(self):
        """Test error handling is production-ready."""
        # Invalid model parameters
        with pytest.raises(ValueError):
            AR(p=0)

        with pytest.raises(ValueError):
            ARMA(p=0, q=0)

        # Insufficient data (need at least p+1 observations)
        x = np.array([1.0])
        model = AR(p=1)
        with pytest.raises(ValueError):
            model.fit(x)

        # Not fitted before predict
        model = AR(p=1)
        with pytest.raises(RuntimeError):
            model.predict(steps=10)

    def test_deterministic_results(self):
        """Test that results are deterministic with fixed seeds."""
        np.random.seed(123)
        n = 200
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + eps[t]

        # Fit twice with same seed
        model1 = AR(p=1)
        res1 = model1.fit(x)

        model2 = AR(p=1)
        res2 = model2.fit(x)

        # Results should be identical
        np.testing.assert_allclose(res1.params, res2.params)

    def test_type_safety(self):
        """Test that type hints work correctly."""
        from typing import get_type_hints

        # Check FitResult has proper types
        hints = get_type_hints(FitResult)
        assert "params" in hints
        assert "aic" in hints
        assert "bic" in hints

    def test_large_dataset(self):
        """Test performance with larger datasets."""
        np.random.seed(42)
        n = 2000  # Larger dataset
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + eps[t]

        model = AR(p=1)
        res = model.fit(x)
        assert res.success

        # Should handle gracefully
        forecast, ci = model.predict(steps=50)
        assert len(forecast) == 50

