"""Quasi-Newton optimization algorithms (BFGS and L-BFGS)."""

from __future__ import annotations

import inspect
from collections import deque
from typing import Callable, Deque

import numpy as np

from .core import RTOL, OptimizeResult, Problem, check_convergence
from .line_search import wolfe_line_search
from .utils import approx_grad


def _compute_gradient(problem: Problem, x: np.ndarray) -> tuple[np.ndarray, int, int]:
    if problem.grad is not None:
        return np.asarray(problem.grad(x), dtype=float), 0, 1
    grad, evals = approx_grad(problem.fun, x, return_evals=True)
    return grad, int(evals), 0


def _line_search_requires_grad(func: Callable) -> bool:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    return len(params) >= 2 and params[1].name == "grad"


def bfgs(
    problem: Problem,
    x0: np.ndarray,
    maxiter: int = 1000,
    tol: float = RTOL,
    line_search: Callable = wolfe_line_search,
    history: bool = False,
) -> OptimizeResult:
    """Full-memory BFGS with strong Wolfe line search."""
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    inv_hessian = np.eye(n)
    hist: list[np.ndarray] = []
    if history:
        hist.append(x.copy())
    nfev = 0
    njev = 0
    nit = 0
    fx = problem.fun(x)
    nfev += 1
    grad, grad_fev, grad_jev = _compute_gradient(problem, x)
    nfev += grad_fev
    njev += grad_jev
    requires_grad = _line_search_requires_grad(line_search)
    success = False
    message = "Maximum iterations reached."

    def grad_for_line_search(point: np.ndarray) -> np.ndarray:
        nonlocal nfev, njev
        g, fe, je = _compute_gradient(problem, point)
        nfev += fe
        njev += je
        return g

    while nit < maxiter:
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
            if history and (len(hist) == 0 or not np.array_equal(hist[-1], x)):
                hist.append(x.copy())
            nit += 1
            break
        direction = -inv_hessian @ grad
        if requires_grad:
            gradient_callable = problem.grad or grad_for_line_search
            alpha, ls_evals = line_search(problem.fun, gradient_callable, x, direction)
        else:
            alpha, ls_evals = line_search(problem.fun, x, direction, grad)
        nfev += int(ls_evals)
        s = alpha * direction
        x_new = x + s
        fx = problem.fun(x_new)
        nfev += 1
        grad_new, grad_fev, grad_jev = _compute_gradient(problem, x_new)
        nfev += grad_fev
        njev += grad_jev
        y = grad_new - grad
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            inv_hessian = np.eye(n)
        else:
            rho = 1.0 / ys
            identity = np.eye(n)
            outer_ss = np.outer(s, s)
            outer_sy = np.outer(s, y)
            outer_ys = outer_sy.T
            inv_hessian = (
                (identity - rho * outer_sy)
                @ inv_hessian
                @ (identity - rho * outer_ys)
                + rho * outer_ss
            )
        x = x_new
        grad = grad_new
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
        nhev=0,
        history=hist,
    )
    return result


def lbfgs(
    problem: Problem,
    x0: np.ndarray,
    m: int = 10,
    maxiter: int = 1000,
    tol: float = RTOL,
    line_search: Callable = wolfe_line_search,
    history: bool = False,
) -> OptimizeResult:
    """Limited-memory BFGS using two-loop recursion."""
    if m <= 0:
        raise ValueError("Memory parameter m must be positive.")
    x = np.asarray(x0, dtype=float).copy()
    hist: list[np.ndarray] = []
    if history:
        hist.append(x.copy())
    nfev = 0
    njev = 0
    nit = 0
    fx = problem.fun(x)
    nfev += 1
    grad, grad_fev, grad_jev = _compute_gradient(problem, x)
    nfev += grad_fev
    njev += grad_jev
    s_history: Deque[np.ndarray] = deque(maxlen=m)
    y_history: Deque[np.ndarray] = deque(maxlen=m)
    requires_grad = _line_search_requires_grad(line_search)
    success = False
    message = "Maximum iterations reached."

    def grad_for_line_search(point: np.ndarray) -> np.ndarray:
        nonlocal nfev, njev
        g, fe, je = _compute_gradient(problem, point)
        nfev += fe
        njev += je
        return g

    def two_loop(g: np.ndarray) -> np.ndarray:
        q = g.copy()
        alpha_vals = []
        for s, y in reversed(list(zip(s_history, y_history))):
            rho = 1.0 / float(np.dot(y, s))
            alpha_i = rho * float(np.dot(s, q))
            q = q - alpha_i * y
            alpha_vals.append((rho, alpha_i, s, y))
        if len(s_history) > 0:
            last_s = s_history[-1]
            last_y = y_history[-1]
            gamma = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
        else:
            gamma = 1.0
        r = gamma * q
        for rho, alpha_i, s, y in reversed(alpha_vals):
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha_i - beta)
        return -r

    while nit < maxiter:
        grad_norm = float(np.linalg.norm(grad))
        if check_convergence(grad_norm, tol):
            success = True
            message = "Gradient tolerance satisfied."
            if history and (len(hist) == 0 or not np.array_equal(hist[-1], x)):
                hist.append(x.copy())
            nit += 1
            break
        direction = two_loop(grad)
        if requires_grad:
            gradient_callable = problem.grad or grad_for_line_search
            alpha, ls_evals = line_search(problem.fun, gradient_callable, x, direction)
        else:
            alpha, ls_evals = line_search(problem.fun, x, direction, grad)
        nfev += int(ls_evals)
        s = alpha * direction
        x_new = x + s
        fx = problem.fun(x_new)
        nfev += 1
        grad_new, grad_fev, grad_jev = _compute_gradient(problem, x_new)
        nfev += grad_fev
        njev += grad_jev
        y = grad_new - grad
        ys = float(np.dot(y, s))
        if ys > 1e-12:
            s_history.append(s)
            y_history.append(y)
        x = x_new
        grad = grad_new
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
        nhev=0,
        history=hist,
    )
    return result


__all__ = ["bfgs", "lbfgs"]

