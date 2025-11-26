"""Trust-region methods with dogleg and Cauchy point strategies."""

from __future__ import annotations

import numpy as np

from .core import RTOL, OptimizeResult, Problem, check_convergence
from .utils import approx_grad, approx_hessian, safe_solve


def _compute_gradient(problem: Problem, x: np.ndarray) -> tuple[np.ndarray, int, int]:
    if problem.grad is not None:
        return np.asarray(problem.grad(x), dtype=float), 0, 1
    grad, evals = approx_grad(problem.fun, x, return_evals=True)
    return grad, int(evals), 0


def _compute_hessian(problem: Problem, x: np.ndarray) -> tuple[np.ndarray, int, int]:
    if problem.hess is not None:
        return np.asarray(problem.hess(x), dtype=float), 0, 1
    hess, evals = approx_hessian(problem.fun, x, return_evals=True)
    return hess, int(evals), 0


def _cauchy_point(grad: np.ndarray, hess: np.ndarray, delta: float) -> np.ndarray:
    grad_norm = np.linalg.norm(grad)
    if grad_norm == 0:
        return np.zeros_like(grad)
    gbg = float(grad @ (hess @ grad))
    if gbg <= 0:
        tau = 1.0
    else:
        tau = min((grad_norm**3) / (delta * gbg), 1.0)
    return -(tau * delta / grad_norm) * grad


def _dogleg_step(
    grad: np.ndarray, hess: np.ndarray, delta: float
) -> np.ndarray:
    """Compute dogleg step combining Cauchy point and Newton step."""
    p_u = _cauchy_point(grad, hess, delta)
    try:
        p_b = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        p_b = safe_solve(hess, -grad)
    if np.linalg.norm(p_b) <= delta:
        return p_b
    if np.linalg.norm(p_u) >= delta:
        return (delta / np.linalg.norm(p_u)) * p_u
    diff = p_b - p_u
    a = float(np.dot(diff, diff))
    if a <= 0:
        return (delta / np.linalg.norm(p_u)) * p_u
    b = 2.0 * float(np.dot(p_u, diff))
    c = float(np.dot(p_u, p_u)) - delta**2
    disc = max(b * b - 4 * a * c, 0.0)
    tau = (-b + np.sqrt(disc)) / (2 * a)
    return p_u + tau * diff


def trust_region(
    problem: Problem,
    x0: np.ndarray,
    delta0: float = 1.0,
    max_delta: float = 100.0,
    eta: float = 0.15,
    maxiter: int = 200,
    tol: float = RTOL,
    history: bool = False,
) -> OptimizeResult:
    """Dogleg trust-region solver."""
    x = np.asarray(x0, dtype=float).copy()
    delta = delta0
    hist: list[np.ndarray] = []
    if history:
        hist.append(x.copy())
    nfev = 0
    njev = 0
    nhev = 0
    nit = 0
    fx = problem.fun(x)
    nfev += 1
    success = False
    message = "Maximum iterations reached."
    grad = np.zeros_like(x)

    while nit < maxiter:
        grad, grad_fev, grad_jev = _compute_gradient(problem, x)
        nfev += grad_fev
        njev += grad_jev
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
            if history and (len(hist) == 0 or not np.array_equal(hist[-1], x)):
                hist.append(x.copy())
            nit += 1
            break
        hess, hess_fev, hess_jev = _compute_hessian(problem, x)
        nfev += hess_fev
        nhev += hess_jev
        step = _dogleg_step(grad, hess, delta)
        x_candidate = x + step
        f_candidate = problem.fun(x_candidate)
        nfev += 1
        actual_red = fx - f_candidate
        model = fx + np.dot(grad, step) + 0.5 * step @ (hess @ step)
        predicted_red = fx - model
        if predicted_red <= 0:
            rho = 0.0
        else:
            rho = actual_red / predicted_red
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and np.linalg.norm(step) >= 0.9 * delta:
            delta = min(2.0 * delta, max_delta)
        if rho > eta and np.isfinite(f_candidate):
            x = x_candidate
            fx = f_candidate
            if history:
                hist.append(x.copy())
        nit += 1
    else:
        grad_norm = float(np.linalg.norm(grad))
    result = OptimizeResult(
        x=x,
        fun=float(fx),
        nit=nit,
        success=success,
        message=message,
        grad_norm=float(np.linalg.norm(grad)),
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        history=hist,
    )
    return result


__all__ = ["trust_region"]

