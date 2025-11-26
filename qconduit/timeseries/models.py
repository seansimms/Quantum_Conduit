"""Time-series model classes: AR, MA, ARMA, ARIMA, SARIMA.

This module provides class-based interfaces for fitting and forecasting
time-series models. All models support .fit(), .predict(), and .simulate()
methods.

References:
    - Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
    - Hamilton (1994): Time Series Analysis
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .estimation import (
    FitResult,
    conditional_sum_squares_arma,
    mle_arma,
    ols_ar,
    yule_walker,
)
from .utils import difference

try:
    from scipy import optimize  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AR:
    """Autoregressive (AR) model.

    Models the time series as:
        x_t = c + φ_1 x_{t-1} + ... + φ_p x_{t-p} + ε_t

    where ε_t ~ N(0, σ²).

    Args:
        p: AR order. Must be >= 1.
        trend: Trend specification. Options:
            - "n": No intercept
            - "c": Constant term
        method: Estimation method. Options:
            - "yule_walker": Yule-Walker equations (default)
            - "ols": Ordinary least squares

    Example:
        >>> np.random.seed(0)
        >>> n = 300
        >>> phi = 0.7
        >>> eps = np.random.normal(size=n)
        >>> x = np.zeros(n)
        >>> for t in range(1, n):
        ...     x[t] = phi * x[t-1] + eps[t]
        >>> model = AR(p=1)
        >>> res = model.fit(x)
        >>> fcast, ci = model.predict(steps=10, alpha=0.05)
    """

    def __init__(self, p: int, trend: str = "n", method: str = "yule_walker") -> None:
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if trend not in ["n", "c"]:
            raise ValueError(f"trend must be 'n' or 'c', got {trend}")
        if method not in ["yule_walker", "ols"]:
            raise ValueError(f"method must be 'yule_walker' or 'ols', got {method}")

        self.p = p
        self.trend = trend
        self.method = method
        self.fit_result: Optional[FitResult] = None
        self._fitted_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> FitResult:
        """Fit the AR model to data.

        Args:
            x: 1D time series array, shape (n,).

        Returns:
            FitResult with parameter estimates, AIC/BIC, etc.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")

        if self.method == "yule_walker":
            phi, sigma2 = yule_walker(x, p=self.p)
            # Compute log-likelihood for AIC/BIC
            n = len(x)
            loglike = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
            n_params = self.p
            if self.trend == "c":
                # Yule-Walker doesn't handle trend, approximate
                n_params += 1
            aic = 2 * n_params - 2 * loglike
            bic = np.log(n) * n_params - 2 * loglike

            self.fit_result = FitResult(
                params=phi,
                stderr=None,
                sigma2=sigma2,
                aic=aic,
                bic=bic,
                nobs=n,
                success=True,
                message="Yule-Walker estimation successful",
            )
        else:  # OLS
            self.fit_result = ols_ar(x, p=self.p, trend=self.trend)

        self._fitted_x = x.copy()
        return self.fit_result

    def predict(
        self, steps: int, start: Optional[np.ndarray] = None, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts and confidence intervals.

        Args:
            steps: Number of steps ahead to forecast.
            start: Optional starting values for forecasting. If None, uses
                the last p values from the fitted series.
            alpha: Significance level for confidence intervals (default 0.05).

        Returns:
            Tuple of (forecast, conf_int) where:
            - forecast: Forecast values, shape (steps,).
            - conf_int: Confidence intervals, shape (2, steps).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self._fitted_x is None:
            raise RuntimeError("Fitted data not available")

        # Get last p values for initialization
        if start is None:
            start = self._fitted_x[-self.p :]
        else:
            start = np.asarray(start)
            if len(start) < self.p:
                raise ValueError(f"start must have at least {self.p} values")

        phi = self.fit_result.params
        sigma2 = self.fit_result.sigma2

        # Generate forecasts recursively
        forecast = np.zeros(steps)
        forecast_mse = np.zeros(steps)  # Forecast mean squared error

        # Initialize with last p values
        x_last = start[-self.p :].copy()

        for h in range(steps):
            # Forecast: x_{t+h} = φ_1 x_{t+h-1} + ... + φ_p x_{t+h-p}
            forecast[h] = np.dot(phi, x_last[-self.p :])

            # Forecast MSE accumulates innovation variance
            # For AR(p), MSE(h) = σ² for h=1, increases for h>1
            if h == 0:
                forecast_mse[h] = sigma2
            else:
                # Recursive formula for forecast MSE
                forecast_mse[h] = sigma2 * (1 + np.sum(phi[: min(h, self.p)] ** 2))

            # Update x_last for next iteration
            x_last = np.append(x_last, forecast[h])[-self.p :]

        # Confidence intervals (Gaussian approximation)
        z_score = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        se = np.sqrt(forecast_mse)
        conf_int = np.zeros((2, steps))
        conf_int[0, :] = forecast - z_score * se
        conf_int[1, :] = forecast + z_score * se

        return forecast, conf_int

    def simulate(
        self, nsim: int, burn: int = 0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate from the fitted AR model.

        Args:
            nsim: Number of observations to simulate.
            burn: Number of burn-in observations (discarded).
            seed: Random seed for reproducibility.

        Returns:
            Simulated series, shape (nsim,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before simulation")

        rng = np.random.RandomState(seed)
        phi = self.fit_result.params
        sigma = np.sqrt(self.fit_result.sigma2)

        # Total length including burn-in
        total_n = nsim + burn + self.p

        # Initialize
        x = np.zeros(total_n)
        eps = rng.normal(scale=sigma, size=total_n)

        # Generate with initial zeros
        for t in range(self.p, total_n):
            x[t] = np.dot(phi, x[t - self.p : t][::-1]) + eps[t]

        # Return requested portion (after burn-in)
        return x[self.p + burn :]


class MA:
    """Moving average (MA) model.

    Models the time series as:
        x_t = μ + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}

    where ε_t ~ N(0, σ²).

    Args:
        q: MA order. Must be >= 1.

    Example:
        >>> np.random.seed(0)
        >>> n = 500
        >>> theta = 0.5
        >>> eps = np.random.normal(size=n)
        >>> x = eps[1:] + theta * eps[:-1]
        >>> model = MA(q=1)
        >>> res = model.fit(x)
    """

    def __init__(self, q: int) -> None:
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        self.q = q
        self.fit_result: Optional[FitResult] = None
        self._fitted_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, method: str = "css") -> FitResult:
        """Fit the MA model to data.

        Args:
            x: 1D time series array.
            method: Estimation method. Options:
                - "css": Conditional sum-of-squares (default)
                - "mle": Maximum likelihood (requires scipy)

        Returns:
            FitResult with parameter estimates.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")

        # MA is ARMA(0, q)
        if method == "mle":
            if not HAS_SCIPY:
                raise RuntimeError("MLE requires scipy. Use method='css'.")
            params, success, loglike = mle_arma(x, ar_order=0, ma_order=self.q)
        else:
            params, success, css = conditional_sum_squares_arma(
                x, ar_order=0, ma_order=self.q, method="scipy" if HAS_SCIPY else "css"
            )
            # Approximate log-likelihood from CSS
            n = len(x)
            sigma2 = css / n
            loglike = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        # Compute AIC/BIC
        n = len(x)
        n_params = self.q
        aic = 2 * n_params - 2 * loglike
        bic = np.log(n) * n_params - 2 * loglike

        # Estimate sigma2 from innovations
        from .estimation import _innovations_css

        ma_params = params
        _, css = _innovations_css(x, np.array([]), ma_params)
        sigma2 = css / n

        self.fit_result = FitResult(
            params=params,
            stderr=None,
            sigma2=sigma2,
            aic=aic,
            bic=bic,
            nobs=n,
            success=success,
            message=f"MA estimation via {method}",
        )
        self._fitted_x = x.copy()
        return self.fit_result

    def predict(
        self, steps: int, start: Optional[np.ndarray] = None, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts.

        For MA models, forecasts beyond lag q are constant (mean).
        """
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before prediction")

        # MA(q) forecasts: beyond lag q, forecast is zero (mean)
        forecast = np.zeros(steps)
        mu = np.mean(self._fitted_x) if self._fitted_x is not None else 0.0

        # Forecast MSE: σ² for steps <= q, then increases
        sigma2 = self.fit_result.sigma2
        theta = self.fit_result.params

        forecast_mse = np.zeros(steps)
        for h in range(steps):
            if h < self.q:
                # Short-term: depends on MA structure
                forecast_mse[h] = sigma2 * (1 + np.sum(theta[: h + 1] ** 2))
            else:
                # Long-term: converges to unconditional variance
                forecast_mse[h] = sigma2 * (1 + np.sum(theta**2))

        forecast.fill(mu)

        z_score = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        se = np.sqrt(forecast_mse)
        conf_int = np.zeros((2, steps))
        conf_int[0, :] = forecast - z_score * se
        conf_int[1, :] = forecast + z_score * se

        return forecast, conf_int

    def simulate(
        self, nsim: int, burn: int = 0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate from the fitted MA model."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before simulation")

        rng = np.random.RandomState(seed)
        theta = self.fit_result.params
        sigma = np.sqrt(self.fit_result.sigma2)

        total_n = nsim + burn + self.q
        eps = rng.normal(scale=sigma, size=total_n)
        x = np.zeros(total_n)

        for t in range(self.q, total_n):
            x[t] = eps[t] + np.dot(theta, eps[t - self.q : t][::-1])

        return x[self.q + burn :]


class ARMA:
    """Autoregressive moving average (ARMA) model.

    Models the time series as:
        x_t = c + φ_1 x_{t-1} + ... + φ_p x_{t-p} + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}

    Args:
        p: AR order. Must be >= 0.
        q: MA order. Must be >= 0.
        At least one of p or q must be > 0.

    Example:
        >>> np.random.seed(0)
        >>> n = 500
        >>> eps = np.random.normal(size=n)
        >>> x = np.zeros(n)
        >>> for t in range(1, n):
        ...     x[t] = 0.7 * x[t-1] + eps[t] + 0.3 * eps[t-1]
        >>> model = ARMA(p=1, q=1)
        >>> res = model.fit(x)
    """

    def __init__(self, p: int, q: int) -> None:
        if p < 0:
            raise ValueError(f"p must be >= 0, got {p}")
        if q < 0:
            raise ValueError(f"q must be >= 0, got {q}")
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be > 0")

        self.p = p
        self.q = q
        self.fit_result: Optional[FitResult] = None
        self._fitted_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, method: str = "css") -> FitResult:
        """Fit the ARMA model.

        Args:
            x: 1D time series array.
            method: Estimation method. Options:
                - "css": Conditional sum-of-squares (default)
                - "mle": Maximum likelihood (requires scipy)

        Returns:
            FitResult with parameter estimates.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")

        if method == "mle":
            if not HAS_SCIPY:
                raise RuntimeError("MLE requires scipy. Use method='css'.")
            params, success, loglike = mle_arma(
                x, ar_order=self.p, ma_order=self.q
            )
        else:
            params, success, css = conditional_sum_squares_arma(
                x,
                ar_order=self.p,
                ma_order=self.q,
                method="scipy" if HAS_SCIPY else "css",
            )
            n = len(x)
            sigma2 = css / n
            loglike = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        n = len(x)
        n_params = self.p + self.q
        aic = 2 * n_params - 2 * loglike
        bic = np.log(n) * n_params - 2 * loglike

        from .estimation import _innovations_css

        ar_params = params[: self.p] if self.p > 0 else np.array([])
        ma_params = params[self.p :] if self.q > 0 else np.array([])
        _, css = _innovations_css(x, ar_params, ma_params)
        sigma2 = css / n

        self.fit_result = FitResult(
            params=params,
            stderr=None,
            sigma2=sigma2,
            aic=aic,
            bic=bic,
            nobs=n,
            success=success,
            message=f"ARMA estimation via {method}",
        )
        self._fitted_x = x.copy()
        return self.fit_result

    def predict(
        self, steps: int, start: Optional[np.ndarray] = None, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts from ARMA model."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Simplified forecasting: recursive filtering
        # In practice, this requires innovation reconstruction
        forecast = np.zeros(steps)
        mu = np.mean(self._fitted_x) if self._fitted_x is not None else 0.0

        sigma2 = self.fit_result.sigma2
        params = self.fit_result.params
        ma_params = params[self.p :] if self.q > 0 else np.array([])

        # Forecast MSE (simplified)
        forecast_mse = np.full(steps, sigma2 * (1 + np.sum(ma_params**2)))

        forecast.fill(mu)

        z_score = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        se = np.sqrt(forecast_mse)
        conf_int = np.zeros((2, steps))
        conf_int[0, :] = forecast - z_score * se
        conf_int[1, :] = forecast + z_score * se

        return forecast, conf_int

    def simulate(
        self, nsim: int, burn: int = 0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate from the fitted ARMA model."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before simulation")

        rng = np.random.RandomState(seed)
        params = self.fit_result.params
        ar_params = params[: self.p] if self.p > 0 else np.array([])
        ma_params = params[self.p :] if self.q > 0 else np.array([])
        sigma = np.sqrt(self.fit_result.sigma2)

        total_n = nsim + burn + max(self.p, self.q)
        eps = rng.normal(scale=sigma, size=total_n)
        x = np.zeros(total_n)

        for t in range(max(self.p, self.q), total_n):
            ar_part = np.dot(ar_params, x[t - self.p : t][::-1]) if self.p > 0 else 0.0
            ma_part = (
                np.dot(ma_params, eps[t - self.q : t][::-1]) if self.q > 0 else 0.0
            )
            x[t] = ar_part + eps[t] + ma_part

        return x[max(self.p, self.q) + burn :]


class ARIMA:
    """Autoregressive integrated moving average (ARIMA) model.

    ARIMA(p, d, q) applies d differences to the data, then fits ARMA(p, q)
    to the differenced series.

    Args:
        p: AR order.
        d: Differencing order. Must be >= 0.
        q: MA order.

    Example:
        >>> np.random.seed(0)
        >>> # Simulate ARIMA(1,1,0) by integrating AR(1)
        >>> n = 500
        >>> eps = np.random.normal(size=n)
        >>> x_diff = np.zeros(n)
        >>> for t in range(1, n):
        ...     x_diff[t] = 0.7 * x_diff[t-1] + eps[t]
        >>> x = np.cumsum(x_diff)
        >>> model = ARIMA(p=1, d=1, q=0)
        >>> res = model.fit(x)
    """

    def __init__(self, p: int, d: int, q: int) -> None:
        if p < 0:
            raise ValueError(f"p must be >= 0, got {p}")
        if d < 0:
            raise ValueError(f"d must be >= 0, got {d}")
        if q < 0:
            raise ValueError(f"q must be >= 0, got {q}")
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be > 0")

        self.p = p
        self.d = d
        self.q = q
        self.fit_result: Optional[FitResult] = None
        self._fitted_x: Optional[np.ndarray] = None
        self._original_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, method: str = "css") -> FitResult:
        """Fit the ARIMA model.

        Args:
            x: 1D time series array.
            method: Estimation method ("css" or "mle").

        Returns:
            FitResult with parameter estimates.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")

        self._original_x = x.copy()

        # Apply differencing
        if self.d > 0:
            x_diff = difference(x, d=self.d)
        else:
            x_diff = x

        # Fit ARMA on differenced data
        arma = ARMA(p=self.p, q=self.q)
        self.fit_result = arma.fit(x_diff, method=method)
        self._fitted_x = x_diff.copy()

        return self.fit_result

    def predict(
        self, steps: int, start: Optional[np.ndarray] = None, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts, transforming back to original level."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Forecast on differenced scale
        arma = ARMA(p=self.p, q=self.q)
        arma.fit_result = self.fit_result
        arma._fitted_x = self._fitted_x

        fcast_diff, ci_diff = arma.predict(steps=steps, start=start, alpha=alpha)

        # Transform back to original level
        if self.d > 0 and self._original_x is not None:
            # Invert differencing
            # Simplified: accumulate differences
            last_value = self._original_x[-1]
            forecast = np.zeros(steps)
            forecast[0] = last_value + fcast_diff[0]
            for h in range(1, steps):
                forecast[h] = forecast[h - 1] + fcast_diff[h]

            # Adjust confidence intervals similarly
            conf_int = np.zeros((2, steps))
            conf_int[0, 0] = last_value + ci_diff[0, 0]
            conf_int[1, 0] = last_value + ci_diff[1, 0]
            for h in range(1, steps):
                conf_int[0, h] = conf_int[0, h - 1] + ci_diff[0, h]
                conf_int[1, h] = conf_int[1, h - 1] + ci_diff[1, h]
        else:
            forecast = fcast_diff
            conf_int = ci_diff

        return forecast, conf_int


class SARIMA:
    """Seasonal ARIMA model.

    SARIMA(p, d, q)(P, D, Q, s) combines:
    - Regular ARIMA(p, d, q)
    - Seasonal ARIMA(P, D, Q) with period s

    Args:
        p: Regular AR order.
        d: Regular differencing order.
        q: Regular MA order.
        P: Seasonal AR order.
        D: Seasonal differencing order.
        Q: Seasonal MA order.
        s: Seasonal period. Must be > 0.

    Note:
        Full seasonal parameter estimation is computationally intensive.
        Uses CSS method for estimation.

    Example:
        >>> np.random.seed(0)
        >>> n = 200
        >>> x = np.random.randn(n) + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
        >>> model = SARIMA(p=1, d=0, q=1, P=1, D=1, Q=0, s=12)
        >>> res = model.fit(x)
    """

    def __init__(
        self, p: int, d: int, q: int, P: int, D: int, Q: int, s: int
    ) -> None:
        if s < 1:
            raise ValueError(f"s (seasonal period) must be > 0, got {s}")
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.fit_result: Optional[FitResult] = None
        self._fitted_x: Optional[np.ndarray] = None
        self._original_x: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, method: str = "css") -> FitResult:
        """Fit the SARIMA model.

        Applies regular and seasonal differencing, then fits combined ARMA model.
        Uses CSS estimation (MLE may be slow for seasonal models).

        Args:
            x: 1D time series array.
            method: Estimation method. "css" is recommended for seasonal models.

        Returns:
            FitResult with parameter estimates.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D array, got shape {x.shape}")

        self._original_x = x.copy()

        # Apply regular differencing
        if self.d > 0:
            x_diff = difference(x, d=self.d)
        else:
            x_diff = x

        # Apply seasonal differencing
        if self.D > 0:
            for _ in range(self.D):
                if len(x_diff) < self.s + 1:
                    raise ValueError("Insufficient data after seasonal differencing")
                x_diff = difference(x_diff, d=0, seasonal=self.s)

        # Fit combined ARMA model
        # Note: Full SARIMA fitting is complex. Here we use a simplified approach:
        # fit ARMA(p+s*P, q+s*Q) on the differenced data as approximation.
        # For true SARIMA, need multiplicative structure.
        # This is a simplified implementation.
        total_ar = self.p + self.s * self.P
        total_ma = self.q + self.s * self.Q

        arma = ARMA(p=total_ar, q=total_ma)
        self.fit_result = arma.fit(x_diff, method=method)
        self._fitted_x = x_diff.copy()

        return self.fit_result

    def predict(
        self, steps: int, start: Optional[np.ndarray] = None, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts, transforming back to original level."""
        if self.fit_result is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Forecast on differenced scale (simplified)
        arma = ARMA(p=self.p + self.s * self.P, q=self.q + self.s * self.Q)
        arma.fit_result = self.fit_result
        arma._fitted_x = self._fitted_x

        fcast_diff, ci_diff = arma.predict(steps=steps, start=start, alpha=alpha)

        # Transform back (simplified inversion)
        if (self.d > 0 or self.D > 0) and self._original_x is not None:
            # Simplified: use last value and accumulate
            last_value = self._original_x[-1]
            forecast = last_value + np.cumsum(fcast_diff)
            conf_int = np.zeros((2, steps))
            conf_int[0, :] = last_value + np.cumsum(ci_diff[0, :])
            conf_int[1, :] = last_value + np.cumsum(ci_diff[1, :])
        else:
            forecast = fcast_diff
            conf_int = ci_diff

        return forecast, conf_int

