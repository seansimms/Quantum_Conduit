"""Tests for AR models."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.models import AR


class TestAR:
    """Tests for AR model class."""

    def test_ar1_yule_walker(self):
        """Test AR(1) fitting with Yule-Walker."""
        np.random.seed(42)
        n = 500
        phi_true = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi_true * x[t - 1] + eps[t]

        model = AR(p=1, method="yule_walker")
        res = model.fit(x)

        assert res.success
        assert abs(res.params[0] - phi_true) < 0.10  # More lenient tolerance for OLS
        assert res.nobs == n
        assert res.sigma2 > 0

    def test_ar1_ols(self):
        """Test AR(1) fitting with OLS."""
        np.random.seed(42)
        n = 300
        phi_true = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi_true * x[t - 1] + eps[t]

        model = AR(p=1, method="ols")
        res = model.fit(x)

        assert res.success
        assert abs(res.params[0] - phi_true) < 0.06  # More lenient tolerance for OLS
        assert res.aic is not None
        assert res.bic is not None

    def test_ar2_fitting(self):
        """Test AR(2) fitting."""
        np.random.seed(42)
        n = 500
        phi1 = 0.5
        phi2 = 0.3
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(2, n):
            x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]

        model = AR(p=2)
        res = model.fit(x)

        assert res.success
        assert len(res.params) == 2
        assert abs(res.params[0] - phi1) < 0.1

    def test_ar_predict(self):
        """Test AR forecasting."""
        np.random.seed(42)
        n = 300
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        model = AR(p=1)
        model.fit(x)

        forecast, ci = model.predict(steps=10, alpha=0.05)
        assert len(forecast) == 10
        assert ci.shape == (2, 10)
        assert np.all(ci[1, :] >= ci[0, :])  # Upper >= lower

    def test_ar_simulate(self):
        """Test AR simulation."""
        np.random.seed(42)
        n = 300
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        model = AR(p=1)
        model.fit(x)

        sim = model.simulate(nsim=200, burn=50, seed=123)
        assert len(sim) == 200
        # Check variance is reasonable
        assert 0.5 < np.var(sim) < 3.0

    def test_ar_invalid(self):
        """Test AR model with invalid inputs."""
        with pytest.raises(ValueError, match="p must be >= 1"):
            AR(p=0)

        with pytest.raises(ValueError, match="method must be"):
            AR(p=1, method="invalid")

    def test_ar_not_fitted(self):
        """Test that prediction requires fitting."""
        model = AR(p=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(steps=10)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.simulate(nsim=100)

