"""Utility helpers for finite differences and linear algebra routines.

These utilities avoid any dependency on SciPy and provide deterministic,
pure NumPy implementations suitable for small to medium scale problems.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray
Objective = Callable[[Array], float]


def approx_grad(
    fun: Objective, x: Array, eps: float = 1e-6, return_evals: bool = False
) -> Array | tuple[Array, int]:
    """Compute a central-difference gradient approximation.

    Parameters
    ----------
    fun:
        Objective function returning a scalar given x.
    x:
        Point where the gradient is approximated.
    eps:
        Perturbation size for finite differences.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = np.asarray(x, dtype=float).copy()
    grad = np.zeros_like(x, dtype=float)
    evals = 0
    for i in range(x.size):
        ei = np.zeros_like(x)
        ei[i] = eps
        fx_plus = fun(x + ei)
        fx_minus = fun(x - ei)
        evals += 2
        grad[i] = (fx_plus - fx_minus) / (2.0 * eps)
    if return_evals:
        return grad, evals
    return grad


def approx_hessian(
    fun: Objective, x: Array, eps: float = 1e-4, return_evals: bool = False
) -> Array | tuple[Array, int]:
    """Approximate the Hessian using second-order central differences."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    fx = fun(x)
    evals = 1
    for i in range(n):
        ei = np.zeros_like(x)
        ei[i] = eps
        f_ip = fun(x + ei)
        f_im = fun(x - ei)
        evals += 2
        hess[i, i] = (f_ip - 2 * fx + f_im) / (eps**2)
        for j in range(i + 1, n):
            ej = np.zeros_like(x)
            ej[j] = eps
            f_pp = fun(x + ei + ej)
            f_pm = fun(x + ei - ej)
            f_mp = fun(x - ei + ej)
            f_mm = fun(x - ei - ej)
            evals += 4
            value = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            hess[i, j] = value
            hess[j, i] = value
    if return_evals:
        return hess, evals
    return hess


def is_pos_def(mat: Array, tol: float = 1e-12) -> bool:
    """Check if a matrix is positive definite via eigenvalues."""
    sym = 0.5 * (mat + mat.T)
    eigvals = np.linalg.eigvalsh(sym)
    return np.all(eigvals > tol)


def safe_solve(mat: Array, vec: Array, reg: float = 1e-12) -> Array:
    """Solve linear system with ridge fallback for singular matrices."""
    try:
        return np.linalg.solve(mat, vec)
    except np.linalg.LinAlgError:
        eye = np.eye(mat.shape[0], dtype=mat.dtype)
        return np.linalg.solve(mat + reg * eye, vec)


__all__ = [
    "Array",
    "Objective",
    "approx_grad",
    "approx_hessian",
    "is_pos_def",
    "safe_solve",
]

