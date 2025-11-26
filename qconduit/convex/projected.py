"""
Projected gradient and Newton methods for simple convex constraints.

The functions here require user-provided objective/gradient (and Hessian for
Newton) as well as a projection operator. They implement Armijo backtracking to
ensure deterministic convergence on well-behaved convex objectives.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .core import OptimizeResult, Status
from .utils import stable_solve

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]
Hessian = Callable[[np.ndarray], np.ndarray]
Projection = Callable[[np.ndarray], np.ndarray]


def _armijo(
    x: np.ndarray,
    direction: np.ndarray,
    obj: Objective,
    grad: np.ndarray,
    proj: Projection,
    beta: float = 0.5,
    sigma: float = 1e-4,
) -> tuple[np.ndarray, float]:
    value = obj(x)
    step = 1.0
    for _ in range(30):
        candidate = proj(x + step * direction)
        new_value = obj(candidate)
        actual_direction = candidate - x
        if new_value <= value + sigma * grad.dot(actual_direction):
            return candidate, new_value
        step *= beta
    return x, value


def projected_gradient(
    obj: Objective,
    grad_fun: Gradient,
    proj: Projection,
    x0: np.ndarray,
    lr: float = 1e-2,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> OptimizeResult:
    """
    Projected gradient descent with Armijo backtracking.
    """

    x = np.asarray(x0, dtype=float).reshape(-1)
    value = obj(x)
    converged = False
    nit = 0
    for nit in range(1, maxiter + 1):
        grad = grad_fun(x)
        direction = -grad
        candidate, new_value = _armijo(x, lr * direction, obj, grad, proj)
        x = candidate
        value = new_value
        proj_grad = proj(x - grad_fun(x)) - x
        if np.linalg.norm(proj_grad) <= tol:
            converged = True
            break
    status = Status.OPTIMAL if converged else Status.MAX_ITER
    return OptimizeResult(
        x=x,
        fun=float(value),
        status=status,
        message=(
            "Projected gradient terminated"
            if converged
            else "Projected gradient hit iteration limit"
        ),
        nit=nit,
        primal_residual=float(np.linalg.norm(proj(x - grad_fun(x)) - x)),
    )


def projected_newton(
    obj: Objective,
    grad_fun: Gradient,
    hess_fun: Hessian,
    proj: Projection,
    x0: np.ndarray,
    maxiter: int = 200,
    tol: float = 1e-8,
) -> OptimizeResult:
    """
    Projected Newton method with backtracking.
    """

    x = np.asarray(x0, dtype=float).reshape(-1)
    value = obj(x)
    converged = False
    nit = 0
    for nit in range(1, maxiter + 1):
        grad = grad_fun(x)
        hessian = hess_fun(x)
        try:
            direction = -stable_solve(hessian, grad)
        except np.linalg.LinAlgError:
            return OptimizeResult(
                x=x,
                fun=float(value),
                status=Status.NUMERICAL_ERROR,
                message="Hessian solve failed",
                nit=nit,
            )
        candidate, new_value = _armijo(x, direction, obj, grad, proj)
        x = candidate
        value = new_value
        proj_grad = proj(x - grad_fun(x)) - x
        if np.linalg.norm(proj_grad) <= tol:
            converged = True
            break
    status = Status.OPTIMAL if converged else Status.MAX_ITER
    return OptimizeResult(
        x=x,
        fun=float(value),
        status=status,
        message=(
            "Projected Newton terminated"
            if converged
            else "Projected Newton hit iteration limit"
        ),
        nit=nit,
        primal_residual=float(np.linalg.norm(proj(x - grad_fun(x)) - x)),
    )


__all__ = ["projected_gradient", "projected_newton"]

