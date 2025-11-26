"""Numerical utilities for probabilistic inference.

Provides stable implementations of log-sum-exp, multivariate normal PDFs,
resampling methods, and related helpers used across HMM, GMM, and particle filters.
"""

from typing import Optional, Tuple

import numpy as np


def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute log-sum-exp in a numerically stable way.

    Computes log(sum(exp(a))) avoiding overflow/underflow by subtracting
    the maximum before exponentiating.

    Args:
        a: Input array of log-values.
        axis: Axis along which to compute. If None, flattens array.

    Returns:
        Log-sum-exp result, same shape as input (with axis removed if specified).

    Examples:
        >>> logsumexp(np.array([-10, -11, -12]))
        -9.40760596444438...
        >>> logsumexp(np.array([[1, 2], [3, 4]]), axis=0)
        array([3.126928..., 4.126928...])
    """
    a = np.asarray(a)
    if axis is None:
        a_flat = a.flatten()
        if len(a_flat) == 0:
            return np.array(-np.inf)
        a_max = np.max(a_flat)
        return a_max + np.log(np.sum(np.exp(a_flat - a_max)))
    else:
        # Compute max along axis, keeping dimensions for broadcasting
        a_max = np.max(a, axis=axis, keepdims=True)
        # Compute sum along axis, keeping dimensions
        exp_diff = np.exp(a - a_max)
        sum_exp = np.sum(exp_diff, axis=axis, keepdims=True)
        result = a_max + np.log(sum_exp)
        # Squeeze to remove the keepdims dimension
        return np.squeeze(result, axis=axis)


def normal_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute multivariate normal PDF value (not log).

    Args:
        x: Observation vector, shape (d,).
        mean: Mean vector, shape (d,).
        cov: Covariance matrix, shape (d, d). Must be positive definite.

    Returns:
        PDF value (scalar).

    Raises:
        ValueError: If shapes are incompatible or cov is not positive definite.

    Examples:
        >>> x = np.array([0.0])
        >>> mean = np.array([0.0])
        >>> cov = np.array([[1.0]])
        >>> normal_pdf(x, mean, cov)
        0.3989422804014327
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if x.shape != mean.shape:
        raise ValueError(f"x shape {x.shape} != mean shape {mean.shape}")
    if cov.shape != (len(mean), len(mean)):
        raise ValueError(f"cov shape {cov.shape} incompatible with mean shape {mean.shape}")

    d = len(mean)
    diff = x - mean

    # Use Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive definite")

    # Solve L * y = diff for y
    y = np.linalg.solve(L, diff)
    log_det = np.sum(np.log(np.diag(L)))

    log_pdf = -0.5 * d * np.log(2 * np.pi) - log_det - 0.5 * np.dot(y, y)
    return np.exp(log_pdf)


def log_normal_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute log multivariate normal PDF.

    Uses Cholesky decomposition for numerical stability.

    Args:
        x: Observation vector, shape (d,).
        mean: Mean vector, shape (d,).
        cov: Covariance matrix, shape (d, d). Must be positive definite.

    Returns:
        Log PDF value (scalar).

    Raises:
        ValueError: If shapes are incompatible or cov is not positive definite.

    Examples:
        >>> x = np.array([0.0])
        >>> mean = np.array([0.0])
        >>> cov = np.array([[1.0]])
        >>> log_normal_pdf(x, mean, cov)
        -0.9189385332046727
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if x.shape != mean.shape:
        raise ValueError(f"x shape {x.shape} != mean shape {mean.shape}")
    if cov.shape != (len(mean), len(mean)):
        raise ValueError(f"cov shape {cov.shape} incompatible with mean shape {mean.shape}")

    d = len(mean)
    diff = x - mean

    # Use Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive definite")

    # Solve L * y = diff for y
    y = np.linalg.solve(L, diff)
    log_det = np.sum(np.log(np.diag(L)))

    log_pdf = -0.5 * d * np.log(2 * np.pi) - log_det - 0.5 * np.dot(y, y)
    return log_pdf


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling of indices from normalized weights.

    Deterministic given the RNG state. Returns indices into the weights array.

    Args:
        weights: Normalized weights (should sum to 1), shape (N,).
        rng: Random number generator.

    Returns:
        Array of indices, shape (N,), where each index appears approximately
        weights[i] * N times.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> indices = systematic_resample(weights, rng)
        >>> len(indices) == len(weights)
        True
    """
    weights = np.asarray(weights)
    n = len(weights)

    # Normalize to ensure sum is 1
    weights = weights / np.sum(weights)

    # Cumulative sum
    cumsum = np.cumsum(weights)

    # Systematic resampling: single random offset, then regular spacing
    u = rng.random()
    indices = np.zeros(n, dtype=int)

    u_j = (u + np.arange(n)) / n
    for i in range(n):
        # Find first cumsum >= u_j[i]
        indices[i] = np.searchsorted(cumsum, u_j[i], side="left")

    # Clamp to valid range (shouldn't be needed, but safety)
    indices = np.clip(indices, 0, n - 1)
    return indices


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size (ESS) from normalized weights.

    ESS = 1 / sum(w^2) when weights are normalized.
    Lower ESS indicates higher variance (more resampling needed).

    Args:
        weights: Normalized weights (should sum to 1), shape (N,).

    Returns:
        Effective sample size (scalar).

    Examples:
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> ess = effective_sample_size(weights)
        >>> 1.0 <= ess <= len(weights)
        True
    """
    weights = np.asarray(weights)
    # Normalize
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights**2)


def normalize_log_weights(log_w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize log-weights and return normalized weights + log-evidence.

    Computes: w = exp(log_w - logsumexp(log_w))
    Returns: (normalized_weights, log_evidence) where log_evidence = logsumexp(log_w).

    Args:
        log_w: Log-weights, shape (N,).

    Returns:
        Tuple of (normalized_weights, log_evidence).

    Examples:
        >>> log_w = np.array([-1.0, -2.0, -3.0])
        >>> w, log_z = normalize_log_weights(log_w)
        >>> np.allclose(np.sum(w), 1.0)
        True
        >>> log_z > -1.0
        True
    """
    log_w = np.asarray(log_w)
    log_z = logsumexp(log_w)
    w = np.exp(log_w - log_z)
    return w, log_z


def ensure_1d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 1D, raising error if not.

    Args:
        x: Input array.

    Returns:
        1D array.

    Raises:
        ValueError: If array cannot be flattened to 1D or has wrong shape.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    if x.ndim > 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 2D, adding dimension if needed.

    Args:
        x: Input array.

    Returns:
        2D array (at least 2D).
    """
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

