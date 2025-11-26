"""Time-series forecasting models for QConduit.

This module provides textbook implementations of classical time-series models
including AR, MA, ARMA, ARIMA, SARIMA, and Kalman filtering.

Example:
    >>> from qconduit.timeseries.models import AR, ARIMA
    >>> import numpy as np
    >>>
    >>> # AR(1) example
    >>> np.random.seed(0)
    >>> n = 300
    >>> phi = 0.7
    >>> eps = np.random.normal(size=n)
    >>> x = np.zeros(n)
    >>> for t in range(1, n):
    ...     x[t] = phi * x[t-1] + eps[t]
    >>>
    >>> model = AR(p=1)
    >>> res = model.fit(x)
    >>> fcast, ci = model.predict(steps=10, alpha=0.05)
    >>> print(f"Estimated phi: {res.params[0]:.3f}")
    >>> print(f"AIC: {res.aic:.2f}")

References:
    - Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
    - Hamilton (1994): Time Series Analysis
    - Durbin & Koopman (2012): Time Series Analysis by State Space Methods
"""

from __future__ import annotations

from .diagnostics import aic, bic, ljung_box
from .estimation import (
    FitResult,
    conditional_sum_squares_arma,
    mle_arma,
    ols_ar,
    yule_walker,
)
from .kalman import StateSpace, kalman_filter, kalman_predict, kalman_smoother
from .models import AR, ARIMA, ARMA, MA, SARIMA
from .utils import (
    acf,
    acov,
    difference,
    durbin_levinson,
    invert_difference,
    lag_matrix,
    pacf,
)

__all__ = [
    # Models
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    "SARIMA",
    # Estimation
    "FitResult",
    "yule_walker",
    "ols_ar",
    "conditional_sum_squares_arma",
    "mle_arma",
    # Kalman filtering
    "StateSpace",
    "kalman_filter",
    "kalman_smoother",
    "kalman_predict",
    # Utilities
    "difference",
    "invert_difference",
    "lag_matrix",
    "acov",
    "acf",
    "pacf",
    "durbin_levinson",
    # Diagnostics
    "aic",
    "bic",
    "ljung_box",
]
