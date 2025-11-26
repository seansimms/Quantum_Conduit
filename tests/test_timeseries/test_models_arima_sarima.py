"""Tests for ARIMA and SARIMA models."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.models import ARIMA, SARIMA


class TestARIMA:
    """Tests for ARIMA model class."""

    def test_arima110_fitting(self):
        """Test ARIMA(1,1,0) fitting."""
        np.random.seed(42)
        n = 500
        phi = 0.7
        eps = np.random.normal(size=n)

        # Simulate ARIMA(1,1,0) by integrating AR(1)
        x_diff = np.zeros(n)
        for t in range(1, n):
            x_diff[t] = phi * x_diff[t - 1] + eps[t]
        x = np.cumsum(x_diff)

        model = ARIMA(p=1, d=1, q=0)
        res = model.fit(x)

        assert res.success
        assert len(res.params) == 1
        # AR parameter estimate should be close to true value
        assert abs(res.params[0] - phi) < 0.15

    def test_arima_predict(self):
        """Test ARIMA forecasting."""
        np.random.seed(42)
        n = 500
        phi = 0.7
        eps = np.random.normal(size=n)

        x_diff = np.zeros(n)
        for t in range(1, n):
            x_diff[t] = phi * x_diff[t - 1] + eps[t]
        x = np.cumsum(x_diff)

        model = ARIMA(p=1, d=1, q=0)
        model.fit(x)
        forecast, ci = model.predict(steps=10)

        assert len(forecast) == 10
        assert ci.shape == (2, 10)

    def test_arima_no_differencing(self):
        """Test ARIMA with d=0 (equivalent to ARMA)."""
        np.random.seed(42)
        n = 500
        phi = 0.7
        eps = np.random.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        model = ARIMA(p=1, d=0, q=0)
        res = model.fit(x)

        assert res.success
        assert abs(res.params[0] - phi) < 0.1

    def test_arima_invalid(self):
        """Test ARIMA with invalid orders."""
        with pytest.raises(ValueError, match="At least one"):
            ARIMA(p=0, d=0, q=0)

        with pytest.raises(ValueError, match="d must be >= 0"):
            ARIMA(p=1, d=-1, q=0)


class TestSARIMA:
    """Tests for SARIMA model class."""

    def test_sarima_basic(self):
        """Test basic SARIMA fitting."""
        np.random.seed(42)
        n = 200
        # Create series with seasonal component
        t = np.arange(n)
        x = (
            np.random.randn(n)
            + 0.5 * np.sin(2 * np.pi * t / 12)
            + 0.1 * np.cumsum(np.random.randn(n))
        )

        model = SARIMA(p=1, d=0, q=1, P=0, D=1, Q=0, s=12)
        res = model.fit(x, method="css")

        assert res.success
        assert res.aic is not None
        assert res.bic is not None

    def test_sarima_predict(self):
        """Test SARIMA forecasting."""
        np.random.seed(42)
        n = 200
        t = np.arange(n)
        x = (
            np.random.randn(n)
            + 0.5 * np.sin(2 * np.pi * t / 12)
            + 0.1 * np.cumsum(np.random.randn(n))
        )

        model = SARIMA(p=1, d=0, q=1, P=0, D=1, Q=0, s=12)
        model.fit(x)
        forecast, ci = model.predict(steps=12)

        assert len(forecast) == 12
        assert ci.shape == (2, 12)

    def test_sarima_invalid(self):
        """Test SARIMA with invalid parameters."""
        with pytest.raises(ValueError, match="s .* must be > 0"):
            SARIMA(p=1, d=0, q=1, P=0, D=0, Q=0, s=0)

