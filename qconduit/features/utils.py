"""Utility helpers for feature transformers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _to_2d(X: np.ndarray) -> np.ndarray:
    """Ensure array is strictly 2D with shape (n_samples, n_features)."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}.")
    return X


def check_array(
    X: Any,
    *,
    ensure_2d: bool = True,
    dtype: type | None = np.float64,
    allow_object: bool = False,
) -> np.ndarray:
    """Validate input array."""
    try:
        array = np.asarray(X, dtype=dtype if dtype is not None and not allow_object else None)
    except TypeError as exc:
        raise ValueError("Array cannot be converted to the requested dtype.") from exc
    if np.iscomplexobj(array):
        raise ValueError("Complex data is not supported.")
    if ensure_2d:
        array = _to_2d(array)
    if not allow_object:
        if not np.all(np.isfinite(array)):
            raise ValueError("Array contains NaN or infinite values.")
    return array


def rng_default(seed: int | None = None) -> np.random.Generator:
    """Return a deterministic RNG helper."""
    return np.random.default_rng(seed)


def ensure_same_shape(X: np.ndarray, expected_features: int) -> None:
    """Validate input feature dimensionality."""
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {X.shape[1]}."
        )


def validate_quantile_range(quantile_range: Iterable[float]) -> tuple[float, float]:
    """Validate quantile range tuple."""
    q_min, q_max = tuple(quantile_range)
    if not (0.0 <= q_min < q_max <= 100.0):
        raise ValueError("quantile_range must satisfy 0 <= min < max <= 100.")
    return float(q_min), float(q_max)

