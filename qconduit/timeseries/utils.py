"""Utility functions for time-series analysis.

This module provides core utility functions used throughout the timeseries module,
including differencing, lag matrix construction, autocovariance/autocorrelation
functions, and partial autocorrelation via Durbin-Levinson recursion.

References:
    - Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
    - Hamilton (1994): Time Series Analysis
"""

from __future__ import annotations

import numpy as np


def difference(x: np.ndarray, d: int = 1, seasonal: int = 0) -> np.ndarray:
    """Apply differencing to a time series.

    Applies d times first differencing: Δx_t = x_t - x_{t-1},
    and optionally seasonal differencing with lag `seasonal`.

    Args:
        x: 1D array of time series values, shape (n,).
        d: Number of times to apply first differencing. Default is 1.
            Must be >= 0.
        seasonal: Seasonal differencing period. If > 0, applies
            seasonal differencing after regular differencing: x_t - x_{t-seasonal}.
            Default is 0 (no seasonal differencing).

    Returns:
        Differenced time series. Length is n - d*1 - seasonal if seasonal > 0.

    Raises:
        ValueError: If d < 0, seasonal < 0, or input is not 1D.

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> diff(x, d=1)
        array([1., 1., 1., 1.])
        >>> diff(x, d=2)
        array([0., 0., 0.])
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    if seasonal < 0:
        raise ValueError(f"seasonal must be >= 0, got {seasonal}")

    result = x.copy()

    # Apply regular differencing d times
    for _ in range(d):
        if len(result) < 2:
            return np.array([])
        result = result[1:] - result[:-1]

    # Apply seasonal differencing if requested
    if seasonal > 0:
        if len(result) < seasonal + 1:
            return np.array([])
        result = result[seasonal:] - result[:-seasonal]

    return result


def invert_difference(
    orig: np.ndarray, diffed: np.ndarray, d: int = 1, seasonal: int = 0
) -> np.ndarray:
    """Reconstruct original series from differenced series.

    Inverts the differencing operation by accumulating the differences
    back to level using the initial values from the original series.

    Args:
        orig: Original (undifferenced) series, shape (n,).
        diffed: Differenced series, shape (n_diff,).
        d: Number of regular differences that were applied.
        seasonal: Seasonal differencing period that was applied.

    Returns:
        Reconstructed series, shape (n,).

    Raises:
        ValueError: If shapes are incompatible or parameters invalid.

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> dx = difference(x, d=1)
        >>> x_recon = invert_difference(x, dx, d=1)
        >>> np.allclose(x[1:], x_recon[1:])
        True
    """
    orig = np.asarray(orig)
    diffed = np.asarray(diffed)

    if orig.ndim != 1:
        raise ValueError(f"orig must be 1D array, got shape {orig.shape}")
    if diffed.ndim != 1:
        raise ValueError(f"diffed must be 1D array, got shape {diffed.shape}")
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    if seasonal < 0:
        raise ValueError(f"seasonal must be >= 0, got {seasonal}")

    n = len(orig)
    n_diff = len(diffed)

    # Expected length after differencing
    expected_n_diff = n - d - (seasonal if seasonal > 0 else 0)
    if n_diff != expected_n_diff:
        raise ValueError(
            f"Incompatible lengths: orig length {n}, diffed length {n_diff}, "
            f"expected diffed length {expected_n_diff} for d={d}, seasonal={seasonal}"
        )

    # Simple reconstruction: accumulate differences starting from original values
    # For d=1: x[t] = x[t-1] + diffed[t-1], so x[t] = x[0] + sum(diffed[:t])

    if seasonal == 0 and d == 1:
        # Simple first difference case
        recon = np.zeros(n, dtype=orig.dtype)
        recon[0] = orig[0]
        recon[1:] = orig[0] + np.cumsum(diffed)
        return recon

    # General case: handle step by step
    result = diffed.copy()

    # Invert seasonal differencing first if applied
    if seasonal > 0:
        recon_seasonal = np.zeros(len(result) + seasonal, dtype=result.dtype)
        recon_seasonal[:seasonal] = orig[:seasonal]
        for i in range(len(result)):
            recon_seasonal[i + seasonal] = recon_seasonal[i] + result[i]
        result = recon_seasonal

    # Invert regular differencing d times
    for _ in range(d):
        if len(result) == 0:
            break
        recon = np.zeros(len(result) + 1, dtype=result.dtype)
        recon[0] = orig[0]  # Always start from first original value
        recon[1:] = recon[0] + np.cumsum(result)
        result = recon

    # Ensure correct length
    if len(result) < n:
        # Pad with last value
        pad_val = result[-1] if len(result) > 0 else orig[0]
        result = np.concatenate([result, np.full(n - len(result), pad_val)])
    elif len(result) > n:
        result = result[:n]

    return result


def lag_matrix(x: np.ndarray, p: int) -> np.ndarray:
    """Build design matrix with lags 1 through p for OLS estimation.

    Constructs a matrix where each row contains the lagged values
    needed to predict x[t] from x[t-1], ..., x[t-p].

    Args:
        x: 1D time series array, shape (n,).
        p: Number of lags to include. Must be >= 1.

    Returns:
        Design matrix of shape (n-p, p) where row i contains
        x[i+p-1], x[i+p-2], ..., x[i] (lags in reverse order).

    Raises:
        ValueError: If p < 1, n < p+1, or x is not 1D.

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> lag_matrix(x, p=2)
        array([[2., 1.],
               [3., 2.],
               [4., 3.]])
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    n = len(x)
    if n < p + 1:
        raise ValueError(f"Need at least p+1={p+1} observations, got {n}")

    X = np.zeros((n - p, p))
    for j in range(p):
        X[:, j] = x[p - 1 - j : n - 1 - j]
    return X


def acov(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute unbiased autocovariance estimates.

    Computes sample autocovariance:
        γ(k) = (1/(n-k)) * Σ_{t=k+1}^n (x_t - x̄)(x_{t-k} - x̄)

    for k = 0, 1, ..., nlags.

    Args:
        x: 1D time series array, shape (n,).
        nlags: Maximum lag to compute. Must be >= 0.

    Returns:
        Array of autocovariances [γ(0), γ(1), ..., γ(nlags)],
        shape (nlags+1,).

    Raises:
        ValueError: If nlags < 0, n < 2, or x is not 1D.

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> acov(x, nlags=2)
        array([2.5, 1.25, 0.])
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if nlags < 0:
        raise ValueError(f"nlags must be >= 0, got {nlags}")
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    x_centered = x - np.mean(x)
    acov_vals = np.zeros(nlags + 1)

    for k in range(nlags + 1):
        if k == 0:
            acov_vals[k] = np.mean(x_centered**2)
        else:
            if n - k > 0:
                acov_vals[k] = np.mean(x_centered[k:] * x_centered[: n - k])
            else:
                acov_vals[k] = 0.0

    return acov_vals


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute autocorrelation function (ACF).

    Computes sample autocorrelation as normalized autocovariance:
        ρ(k) = γ(k) / γ(0)

    for k = 0, 1, ..., nlags.

    Args:
        x: 1D time series array, shape (n,).
        nlags: Maximum lag to compute. Must be >= 0.

    Returns:
        Array of autocorrelations [ρ(0), ρ(1), ..., ρ(nlags)],
        shape (nlags+1,). Note: ρ(0) = 1.0.

    Raises:
        ValueError: If nlags < 0 or x is not 1D.

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> acf(x, nlags=2)
        array([1., 0.5, 0.])
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if nlags < 0:
        raise ValueError(f"nlags must be >= 0, got {nlags}")

    gamma = acov(x, nlags)
    gamma0 = gamma[0]
    if abs(gamma0) < 1e-12:
        # Variance is essentially zero, return zeros
        return np.zeros(nlags + 1)

    return gamma / gamma0


def durbin_levinson(r: np.ndarray) -> np.ndarray:
    """Solve Toeplitz Yule-Walker system via Durbin-Levinson recursion.

    Solves the Toeplitz system:
        R * φ = r[1:]

    where R is the Toeplitz matrix with first row [r[0], r[1], ..., r[p-1]]
    and r is the autocovariance vector [γ(0), γ(1), ..., γ(p)].

    Returns the AR coefficients φ = [φ_1, ..., φ_p].

    Args:
        r: Autocovariance vector [γ(0), γ(1), ..., γ(p)],
            shape (p+1,). Must have r[0] > 0.

    Returns:
        AR coefficients [φ_1, ..., φ_p], shape (p,).

    Raises:
        ValueError: If r is empty, r[0] <= 0, or r is not 1D.

    References:
        Durbin (1960): "The fitting of time-series models"
        Levinson (1947): "The Wiener RMS error criterion"

    Example:
        >>> # For AR(1) with φ=0.7, theoretical autocovariances
        >>> r = np.array([1.0, 0.7, 0.49])
        >>> durbin_levinson(r)
        array([0.7, 0.])
    """
    r = np.asarray(r)
    if r.ndim != 1:
        raise ValueError(f"r must be 1D array, got shape {r.shape}")
    if len(r) < 2:
        raise ValueError(f"Need at least 2 autocovariances, got {len(r)}")
    if r[0] <= 0:
        raise ValueError(f"r[0] (variance) must be > 0, got {r[0]}")

    p = len(r) - 1
    if p == 0:
        return np.array([])

    # Initialize recursion
    phi = np.zeros(p)
    v = r[0]  # Prediction error variance

    # Durbin-Levinson recursion
    for k in range(1, p + 1):
        # Compute reflection coefficient
        num = r[k] - np.dot(phi[: k - 1], r[k - 1 : 0 : -1])
        denom = v
        if abs(denom) < 1e-12:
            # Near-singular system, use ridge
            denom = v + 1e-12
        ak = num / denom

        # Update coefficients
        phi_old = phi[: k - 1].copy()
        phi[k - 1] = ak
        phi[: k - 1] = phi_old - ak * phi_old[::-1]

        # Update prediction error variance
        v = v * (1 - ak**2)

    return phi


def pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute partial autocorrelation function (PACF) via Durbin-Levinson.

    The PACF at lag k is the last coefficient from fitting an AR(k) model,
    computed using the Durbin-Levinson recursion.

    Args:
        x: 1D time series array, shape (n,).
        nlags: Maximum lag to compute. Must be >= 0.

    Returns:
        Array of partial autocorrelations [π(1), π(2), ..., π(nlags)],
        shape (nlags,). Note: PACF starts at lag 1 (no lag-0 term).

    Raises:
        ValueError: If nlags < 1 or x is not 1D.

    Example:
        >>> # For AR(1) process, PACF should be zero after lag 1
        >>> np.random.seed(0)
        >>> x = np.random.randn(100)
        >>> pacf_vals = pacf(x, nlags=5)
        >>> len(pacf_vals)
        5
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}")

    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # Compute autocovariances
    gamma = acov(x, nlags=nlags)
    gamma0 = gamma[0]
    if abs(gamma0) < 1e-12:
        return np.zeros(nlags)

    # Normalize to autocorrelations
    r = gamma / gamma0

    # Compute PACF by solving AR(k) for each k
    pacf_vals = np.zeros(nlags)
    for k in range(1, nlags + 1):
        r_k = r[: k + 1]
        phi_k = durbin_levinson(r_k)
        if len(phi_k) > 0:
            pacf_vals[k - 1] = phi_k[-1]
        else:
            pacf_vals[k - 1] = 0.0

    return pacf_vals

