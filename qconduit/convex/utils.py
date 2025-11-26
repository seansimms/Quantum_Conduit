"""
Numerical helper routines for convex optimization algorithms.

These helpers emphasize determinism and graceful degradation when matrices are
nearly singular. They purposely avoid relying on SciPy so that the convex
module only depends on NumPy by default.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """
    Return the symmetric part of ``matrix``.

    Many textbook algorithms assume exactly symmetric Hessians. This helper
    mitigates small asymmetries due to floating-point error by returning
    ``0.5 * (matrix + matrix.T)``.
    """

    return 0.5 * (matrix + matrix.T)


def stable_solve(matrix: np.ndarray, rhs: np.ndarray, reg: float = 1e-12) -> np.ndarray:
    """
    Solve ``A x = b`` with simple regularization fallbacks.

    The function first attempts ``np.linalg.solve``. Upon encountering a
    ``LinAlgError`` it retries with Tikhonov regularization by adding ``reg``
    to the diagonal. If the system remains singular it falls back to a
    least-squares solve via ``np.linalg.lstsq``.
    """

    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        if reg > 0.0:
            augmented = matrix + reg * np.eye(matrix.shape[0], dtype=matrix.dtype)
            try:
                return np.linalg.solve(augmented, rhs)
            except np.linalg.LinAlgError:
                pass
    # Final fallback: least squares
    sol, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return sol


def cholesky_safe(hessian: np.ndarray, ridge: float = 1e-10, max_tries: int = 5) -> np.ndarray:
    """
    Compute a Cholesky factor with automatic diagonal regularization.

    Args:
        H: Symmetric positive semidefinite matrix.
        ridge: Initial ridge added to the diagonal.
        max_tries: Maximum geometric increases of the ridge term.

    Returns:
        Lower-triangular Cholesky factor.

    Raises:
        np.linalg.LinAlgError: If the matrix cannot be regularized within the
            allowed number of attempts.
    """

    attempt = 0
    current = float(ridge)
    while attempt <= max_tries:
        try:
            return np.linalg.cholesky(
                hessian + current * np.eye(hessian.shape[0], dtype=hessian.dtype)
            )
        except np.linalg.LinAlgError:
            attempt += 1
            current *= 10.0
    raise np.linalg.LinAlgError("Failed to compute Cholesky factor with regularization")


def project_box(x: np.ndarray, lb: Optional[np.ndarray], ub: Optional[np.ndarray]) -> np.ndarray:
    """
    Project ``x`` onto the box defined by ``lb`` and ``ub``.

    Parameters may be ``None`` (interpreted as ``-inf``/``+inf``), in which
    case the projection leaves the corresponding coordinates unchanged.
    """

    projected = np.array(x, dtype=float, copy=True)
    if lb is not None:
        projected = np.maximum(projected, lb)
    if ub is not None:
        projected = np.minimum(projected, ub)
    return projected


__all__ = ["symmetrize", "stable_solve", "cholesky_safe", "project_box"]

