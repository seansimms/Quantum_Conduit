"""Newton and damped Newton optimization routines."""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from .core import RTOL, OptimizeResult, Problem, check_convergence
from .line_search import backtracking_armijo
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


def _get_line_search_requires_grad(func: Callable) -> bool:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) < 2:
        return False
    return params[1].name == "grad"


def newton_method(
    problem: Problem,
    x0: np.ndarray,
    maxiter: int = 100,
    tol: float = RTOL,
    use_line_search: bool = False,
    line_search: Callable = backtracking_armijo,
    lambda_reg: float = 0.0,
    history: bool = False,
) -> OptimizeResult:
    """Newton's method with optional line search and damping."""
    x = np.asarray(x0, dtype=float).copy()
    hist: list[np.ndarray] = []
    if history:
        hist.append(x.copy())
    nfev = 0
    njev = 0
    nhev = 0
    nit = 0
    fx = problem.fun(x)
    nfev += 1
    grad = np.zeros_like(x)
    grad_norm = float("inf")
    hess = np.eye(x.size)
    requires_grad = _get_line_search_requires_grad(line_search)
    success = False
    message = "Maximum iterations reached."

    def grad_for_line_search(point: np.ndarray) -> np.ndarray:
        nonlocal nfev, njev
        g, fe, je = _compute_gradient(problem, point)
        nfev += fe
        njev += je
        return g

    while nit < maxiter:
        grad, grad_fev, grad_jev = _compute_gradient(problem, x)
        nfev += grad_fev
        njev += grad_jev
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
            if history:
                hist.append(x.copy())
            nit += 1
            break
        hess, hess_fev, hess_jev = _compute_hessian(problem, x)
        nfev += hess_fev
        nhev += hess_jev
        reg = max(lambda_reg, 0.0)
        eye = np.eye(len(x))
        solved = False
        for _ in range(5):
            try:
                step = np.linalg.solve(hess + reg * eye, -grad)
                solved = True
                break
            except np.linalg.LinAlgError:
                reg = reg * 10 + 1e-8
        if not solved:
            step = safe_solve(hess + reg * eye, -grad)
        alpha = 1.0
        if use_line_search:
            if requires_grad:
                gradient_callable = problem.grad or grad_for_line_search
                alpha, ls_evals = line_search(problem.fun, gradient_callable, x, step)
            else:
                alpha, ls_evals = line_search(problem.fun, x, step, grad)
            nfev += int(ls_evals)
        x = x + alpha * step
        fx = problem.fun(x)
        nfev += 1
        if history:
            hist.append(x.copy())
        nit += 1
    if not success:
        grad, grad_fev, grad_jev = _compute_gradient(problem, x)
        nfev += grad_fev
        njev += grad_jev
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
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


__all__ = ["newton_method"]

