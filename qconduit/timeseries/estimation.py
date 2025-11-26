"""Parameter estimation routines for time-series models.

This module provides classical estimation methods including:
- Yule-Walker estimation for AR models
- OLS estimation for AR models
- Conditional sum-of-squares (CSS) for ARMA models
- Maximum likelihood estimation (MLE) when scipy is available

References:
    - Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
    - Hamilton (1994): Time Series Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy import optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .utils import acov, durbin_levinson, lag_matrix


@dataclass
class FitResult:
    """Result of fitting a time-series model.

    Attributes:
        params: Estimated parameters, shape (n_params,).
        stderr: Standard errors of parameters, shape (n_params,), or None.
        sigma2: Estimated innovation variance (σ²).
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        nobs: Number of observations used in fitting.
        success: Whether the estimation converged successfully.
        message: Status message from the estimation routine.
    """

    params: np.ndarray
    stderr: Optional[np.ndarray]
    sigma2: float
    aic: float
    bic: float
    nobs: int
    success: bool
    message: str


def yule_walker(
    x: np.ndarray, p: int, method: str = "unbiased"
) -> Tuple[np.ndarray, float]:
    """Estimate AR(p) parameters via Yule-Walker equations.

    Solves the Yule-Walker system using Durbin-Levinson recursion
    to obtain AR coefficients φ = [φ_1, ..., φ_p] and innovation variance σ².

    Args:
        x: 1D time series array, shape (n,).
        p: AR order. Must be >= 1.
        method: Covariance estimation method. Currently only "unbiased" is
            implemented (uses 1/(n-k) divisor in autocovariance).

    Returns:
        Tuple of (phi, sigma2) where:
        - phi: AR coefficients [φ_1, ..., φ_p], shape (p,).
        - sigma2: Innovation variance estimate.

    Raises:
        ValueError: If p < 1, n < p+1, or x is not 1D.

    Example:
        >>> np.random.seed(0)
        >>> # Simulate AR(1) with φ=0.7
        >>> n = 500
        >>> eps = np.random.normal(size=n)
        >>> x = np.zeros(n)
        >>> for t in range(1, n):
        ...     x[t] = 0.7 * x[t-1] + eps[t]
        >>> phi, sigma2 = yule_walker(x, p=1)
        >>> abs(phi[0] - 0.7) < 0.1
        True
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    n = len(x)
    if n < p + 1:
        raise ValueError(f"Need at least p+1={p+1} observations, got {n}")

    if method != "unbiased":
        raise ValueError(f"method must be 'unbiased', got {method}")

    # Compute autocovariances up to lag p
    gamma = acov(x, nlags=p)

    # Check variance
    if gamma[0] < 1e-12:
        raise ValueError("Series has near-zero variance")

    # Solve Yule-Walker system via Durbin-Levinson
    phi = durbin_levinson(gamma)

    # Compute innovation variance
    # σ² = γ(0) - φ' * [γ(1), ..., γ(p)]
    sigma2 = gamma[0] - np.dot(phi, gamma[1 : p + 1])
    sigma2 = max(sigma2, 1e-12)  # Ensure positive

    return phi, sigma2


def ols_ar(x: np.ndarray, p: int, trend: str = "n") -> FitResult:
    """Estimate AR(p) parameters via ordinary least squares (OLS).

    Fits the model:
        x_t = c + φ_1 x_{t-1} + ... + φ_p x_{t-p} + ε_t

    using OLS regression on the lag matrix.

    Args:
        x: 1D time series array, shape (n,).
        p: AR order. Must be >= 1.
        trend: Trend specification. Options:
            - "n": No intercept (default)
            - "c": Constant term
            - "ct": Constant + linear trend (not implemented yet)

    Returns:
        FitResult containing parameter estimates, standard errors, AIC/BIC.

    Raises:
        ValueError: If p < 1, n < p+2, trend is invalid, or x is not 1D.

    Example:
        >>> np.random.seed(0)
        >>> n = 300
        >>> eps = np.random.normal(size=n)
        >>> x = np.zeros(n)
        >>> for t in range(1, n):
        ...     x[t] = 0.7 * x[t-1] + eps[t]
        >>> res = ols_ar(x, p=1)
        >>> res.success
        True
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    n = len(x)
    if n < p + 2:
        raise ValueError(f"Need at least p+2={p+2} observations, got {n}")

    if trend not in ["n", "c"]:
        raise ValueError(f"trend must be 'n' or 'c', got {trend}")

    # Build design matrix
    X = lag_matrix(x, p)

    # Response vector (aligned with X)
    y = x[p:]

    # Add intercept if requested
    if trend == "c":
        X = np.column_stack([np.ones(len(X)), X])

    # Solve normal equations: (X'X) β = X'y
    XtX = X.T @ X
    Xty = X.T @ y

    # Add small ridge for numerical stability
    ridge = 1e-12
    XtX += ridge * np.eye(len(XtX))

    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        beta = np.linalg.pinv(XtX) @ Xty

    # Compute residuals and variance
    y_pred = X @ beta
    residuals = y - y_pred
    n_obs = len(residuals)
    sigma2 = np.mean(residuals**2)

    # Standard errors from covariance matrix
    # se = sqrt(diag(σ² * (X'X)^(-1)))
    try:
        cov_beta = sigma2 * np.linalg.inv(XtX)
        stderr = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        stderr = None

    # Extract AR coefficients (exclude intercept if present)
    if trend == "c":
        params = beta[1:]
        if stderr is not None:
            stderr = stderr[1:]
    else:
        params = beta

    # Compute log-likelihood (Gaussian approximation)
    loglike = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(sigma2) + 1)

    # AIC and BIC
    n_params = len(params) + (1 if trend == "c" else 0)
    aic = 2 * n_params - 2 * loglike
    bic = np.log(n_obs) * n_params - 2 * loglike

    return FitResult(
        params=params,
        stderr=stderr,
        sigma2=sigma2,
        aic=aic,
        bic=bic,
        nobs=n_obs,
        success=True,
        message="OLS estimation successful",
    )


def _innovations_css(
    x: np.ndarray, ar_params: np.ndarray, ma_params: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute innovations and CSS for ARMA model.

    Computes one-step-ahead prediction errors (innovations) using
    the innovations algorithm for ARMA models.

    Args:
        x: 1D time series, shape (n,).
        ar_params: AR parameters [φ_1, ..., φ_p], shape (p,).
        ma_params: MA parameters [θ_1, ..., θ_q], shape (q,).

    Returns:
        Tuple of (innovations, css) where:
        - innovations: Prediction errors, shape (n,).
        - css: Conditional sum of squares.
    """
    n = len(x)
    p = len(ar_params)
    q = len(ma_params)

    # Initialize innovations (set to zero for first max(p,q) observations)
    innovations = np.zeros(n)
    max_lag = max(p, q)

    # Forward recursion to compute innovations
    for t in range(max_lag, n):
        # AR part: x_t - φ_1 x_{t-1} - ... - φ_p x_{t-p}
        pred_ar = 0.0
        if p > 0:
            for i in range(p):
                if t - 1 - i >= 0:
                    pred_ar += ar_params[i] * x[t - 1 - i]

        # MA part: θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
        pred_ma = 0.0
        if q > 0:
            for i in range(q):
                if t - 1 - i >= 0:
                    pred_ma += ma_params[i] * innovations[t - 1 - i]

        # Innovation: x_t - (AR prediction + MA correction)
        innovations[t] = x[t] - pred_ar + pred_ma

    # CSS = sum of squared innovations
    css = np.sum(innovations[max_lag:] ** 2)

    return innovations, css


def conditional_sum_squares_arma(
    x: np.ndarray,
    ar_order: int,
    ma_order: int,
    start_params: Optional[np.ndarray] = None,
    method: str = "scipy",
    options: Optional[dict] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Fit ARMA model via conditional sum-of-squares (CSS).

    Minimizes the sum of squared one-step-ahead prediction errors
    to estimate ARMA(p, q) parameters.

    Args:
        x: 1D time series array, shape (n,).
        ar_order: AR order p. Must be >= 0.
        ma_order: MA order q. Must be >= 0.
        start_params: Initial parameter guess, shape (p+q,).
            If None, uses zeros for MA and Yule-Walker for AR.
        method: Optimization method. Options:
            - "scipy": Use scipy.optimize.minimize if available (default)
            - "css": Use simple deterministic coordinate descent
        options: Additional options for optimizer (method="scipy" only).

    Returns:
        Tuple of (params, success, final_obj) where:
        - params: Estimated parameters [φ_1, ..., φ_p, θ_1, ..., θ_q], shape (p+q,).
        - success: Whether optimization converged.
        - final_obj: Final CSS value.

    Raises:
        ValueError: If orders are invalid, x is not 1D, or method is unknown.
        RuntimeError: If scipy is requested but not available.

    Example:
        >>> np.random.seed(0)
        >>> n = 500
        >>> eps = np.random.normal(size=n)
        >>> x = np.zeros(n)
        >>> for t in range(1, n):
        ...     x[t] = 0.7 * x[t-1] + eps[t] + 0.3 * eps[t-1]
        >>> params, success, _ = conditional_sum_squares_arma(x, 1, 1)
        >>> success
        True
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if ar_order < 0:
        raise ValueError(f"ar_order must be >= 0, got {ar_order}")
    if ma_order < 0:
        raise ValueError(f"ma_order must be >= 0, got {ma_order}")
    if ar_order == 0 and ma_order == 0:
        raise ValueError("At least one of ar_order or ma_order must be > 0")

    n = len(x)
    n_params = ar_order + ma_order
    if n < n_params + 10:
        raise ValueError(f"Need at least {n_params + 10} observations, got {n}")

    if options is None:
        options = {}

    # Initialize parameters
    if start_params is None:
        params = np.zeros(n_params)
        # Use Yule-Walker for AR initial guess if ar_order > 0
        if ar_order > 0:
            try:
                phi_init, _ = yule_walker(x, p=ar_order)
                params[:ar_order] = phi_init
            except Exception:
                pass  # Use zeros if Yule-Walker fails
    else:
        params = np.asarray(start_params)
        if len(params) != n_params:
            raise ValueError(
                f"start_params must have length {n_params}, got {len(params)}"
            )

    def objective(p: np.ndarray) -> float:
        """Objective function: CSS."""
        ar_params = p[:ar_order] if ar_order > 0 else np.array([])
        ma_params = p[ar_order:] if ma_order > 0 else np.array([])
        _, css = _innovations_css(x, ar_params, ma_params)
        return css

    if method == "scipy":
        if not HAS_SCIPY:
            raise RuntimeError(
                "scipy.optimize is required for method='scipy'. "
                "Install scipy or use method='css'."
            )

        # Default optimization options
        opt_options = {
            "maxiter": 1000,
            **options,
        }
        # Remove deprecated options if present
        opt_options.pop("disp", None)
        opt_options.pop("iprint", None)

        # Optimize
        try:
            result = optimize.minimize(
                objective,
                params,
                method="L-BFGS-B",
                options=opt_options,
            )
            return result.x, result.success, result.fun
        except Exception:
            return params, False, objective(params)

    elif method == "css":
        # Simple deterministic coordinate descent
        # This is a basic implementation for when scipy is not available
        max_iter = options.get("maxiter", 100)

        best_params = params.copy()
        best_obj = objective(best_params)

        for _ in range(max_iter):
            improved = False
            for i in range(n_params):
                # Try small perturbations in each direction
                step = 0.01
                for direction in [-1, 1]:
                    trial_params = best_params.copy()
                    trial_params[i] += direction * step
                    trial_obj = objective(trial_params)
                    if trial_obj < best_obj:
                        best_params = trial_params
                        best_obj = trial_obj
                        improved = True

            if not improved:
                break

        return best_params, True, best_obj

    else:
        raise ValueError(f"Unknown method: {method}. Use 'scipy' or 'css'.")


def mle_arma(
    x: np.ndarray,
    ar_order: int,
    ma_order: int,
    start_params: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Fit ARMA model via maximum likelihood estimation (MLE).

    Maximizes the Gaussian log-likelihood. Requires scipy.optimize.

    Args:
        x: 1D time series array, shape (n,).
        ar_order: AR order p.
        ma_order: MA order q.
        start_params: Initial parameter guess.
        options: Options for scipy optimizer.

    Returns:
        Tuple of (params, success, neg_loglike).

    Raises:
        RuntimeError: If scipy is not available.
    """
    if not HAS_SCIPY:
        raise RuntimeError(
            "scipy.optimize is required for MLE. "
            "Use conditional_sum_squares_arma instead."
        )

    x = np.asarray(x)
    n = len(x)

    if start_params is None:
        # Use CSS for initial guess
        start_params, _, _ = conditional_sum_squares_arma(
            x, ar_order, ma_order, method="css" if not HAS_SCIPY else "scipy"
        )

    def neg_loglike(p: np.ndarray) -> float:
        """Negative log-likelihood (to minimize)."""
        ar_params = p[:ar_order] if ar_order > 0 else np.array([])
        ma_params = p[ar_order:] if ma_order > 0 else np.array([])
        innovations, css = _innovations_css(x, ar_params, ma_params)
        sigma2 = css / n
        # Gaussian log-likelihood approximation
        loglike = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        return -loglike

    if options is None:
        options = {"maxiter": 1000, "disp": False}

    try:
        result = optimize.minimize(
            neg_loglike,
            start_params,
            method="L-BFGS-B",
            options=options,
        )
        return result.x, result.success, -result.fun
    except Exception:
        return start_params, False, neg_loglike(start_params)

