"""Diagnostic functions for time-series model evaluation.

This module provides tools for model selection (AIC/BIC) and residual
diagnostics (Ljung-Box test).

References:
    - Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
    - Ljung & Box (1978): "On a measure of lack of fit in time series models"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .estimation import FitResult

try:
    from scipy import stats

    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False


def aic(fitresult: FitResult) -> float:
    """Compute Akaike Information Criterion (AIC).

    AIC = 2k - 2ln(L)
    where k is the number of parameters and L is the likelihood.

    Args:
        fitresult: FitResult from model fitting.

    Returns:
        AIC value.
    """
    return fitresult.aic


def bic(fitresult: FitResult) -> float:
    """Compute Bayesian Information Criterion (BIC).

    BIC = ln(n)k - 2ln(L)
    where k is the number of parameters, n is sample size, and L is the likelihood.

    Args:
        fitresult: FitResult from model fitting.

    Returns:
        BIC value.
    """
    return fitresult.bic


def ljung_box(
    residuals: np.ndarray, lags: Optional[int] = None, return_pvalue: bool = True
) -> tuple[float, float] | float:
    """Ljung-Box test for residual autocorrelation.

    Tests the null hypothesis that residuals are independently distributed
    (no autocorrelation). The test statistic follows a chi-square distribution
    under the null.

    Args:
        residuals: 1D array of residuals, shape (n,).
        lags: Number of lags to test. If None, uses min(10, n/5).
        return_pvalue: If True, returns (statistic, pvalue). If False, returns
            only statistic.

    Returns:
        If return_pvalue=True: tuple of (statistic, pvalue).
        If return_pvalue=False: statistic only.

    Raises:
        ValueError: If residuals is not 1D or has insufficient observations.
        RuntimeError: If scipy.stats is required but not available.

    Example:
        >>> np.random.seed(0)
        >>> residuals = np.random.randn(100)
        >>> stat, pval = ljung_box(residuals, lags=10)
        >>> pval > 0.05  # Should not reject null for white noise
        True
    """
    residuals = np.asarray(residuals)
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1D array, got shape {residuals.shape}")

    n = len(residuals)
    if n < 2:
        raise ValueError(f"Need at least 2 residuals, got {n}")

    if lags is None:
        lags = min(10, n // 5)
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")
    if lags >= n:
        raise ValueError(f"lags must be < n={n}, got {lags}")

    # Compute sample autocorrelations
    from .utils import acf

    acf_vals = acf(residuals, nlags=lags)

    # Ljung-Box statistic
    # Q = n(n+2) * Σ_{k=1}^m [ρ(k)² / (n-k)]
    Q = 0.0
    for k in range(1, lags + 1):
        if k < len(acf_vals):
            rho_k = acf_vals[k]
            Q += rho_k**2 / (n - k)

    Q = n * (n + 2) * Q

    if return_pvalue:
        if not HAS_SCIPY_STATS:
            raise RuntimeError(
                "scipy.stats is required for p-value computation. "
                "Set return_pvalue=False to get statistic only."
            )
        # Chi-square distribution with lags degrees of freedom
        pvalue = 1 - stats.chi2.cdf(Q, df=lags)
        return Q, pvalue

    return Q

