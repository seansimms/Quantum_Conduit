"""Gradient-based optimization algorithms."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .core import RTOL, OptimizeResult, Problem, check_convergence
from .utils import approx_grad


def _compute_gradient(problem: Problem, x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return gradient along with (nfev_increment, njev_increment)."""
    if problem.grad is not None:
        return np.asarray(problem.grad(x), dtype=float), 0, 1
    grad, evals = approx_grad(problem.fun, x, return_evals=True)
    return grad, int(evals), 0


def gradient_descent(
    problem: Problem,
    x0: np.ndarray,
    lr: float = 1e-2,
    maxiter: int = 10_000,
    tol: float = RTOL,
    momentum: bool = False,
    beta: float = 0.9,
    nesterov: bool = False,
    callback: Optional[Callable[[np.ndarray, float, np.ndarray], None]] = None,
    history: bool = False,
) -> OptimizeResult:
    """Classic gradient descent with optional momentum and Nesterov update."""
    if nesterov and not momentum:
        momentum = True
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    hist: list[np.ndarray] = []
    if history:
        hist.append(x.copy())
    nfev = 0
    njev = 0
    nit = 0
    fx = problem.fun(x)
    nfev += 1
    success = False
    message = "Maximum iterations reached."
    grad = np.zeros_like(x)
    while nit < maxiter:
        point = x + beta * v if nesterov else x
        grad, grad_fev, grad_jev = _compute_gradient(problem, point)
        nfev += grad_fev
        njev += grad_jev
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
            x = point if nesterov else x
            fx = problem.fun(x)
            nfev += 1
            if history and (len(hist) == 0 or not np.array_equal(hist[-1], x)):
                hist.append(x.copy())
            nit += 1
            break
        if momentum:
            v = beta * v - lr * grad
            if nesterov:
                x = point + v
            else:
                x = x + v
        else:
            x = x - lr * grad
        fx = problem.fun(x)
        nfev += 1
        if callback is not None:
            callback(x.copy(), fx, grad.copy())
        if history:
            hist.append(x.copy())
        nit += 1
    else:
        grad_norm = float(np.linalg.norm(grad))
        if history and (len(hist) == 0 or not np.array_equal(hist[-1], x)):
            hist.append(x.copy())
    result = OptimizeResult(
        x=x,
        fun=float(fx),
        nit=nit,
        success=success,
        message=message,
        grad_norm=float(np.linalg.norm(grad)),
        nfev=nfev,
        njev=njev,
        nhev=0,
        history=hist,
    )
    return result


__all__ = ["gradient_descent"]

