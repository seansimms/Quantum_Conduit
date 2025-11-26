"""
Interior-point (log-barrier) methods for linear programs.

Implements a basic primal log-barrier algorithm with Newton steps and
backtracking line search as described in Boyd & Vandenberghe (Chapter 11).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .core import OptimizeResult, Status
from .lp import simplex
from .utils import stable_solve


def _ensure_array(arr: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1)
    if out.ndim == 2:
        return out
    raise ValueError(f"{name} must be 1D or 2D array")


def _project_onto_inequalities(
    g_mat: np.ndarray,
    h_vec: np.ndarray,
    x_vec: np.ndarray,
    iterations: int = 50,
) -> np.ndarray:
    """Heuristically project onto Gx <= h using successive projections."""

    for _ in range(iterations):
        violation = g_mat @ x_vec - h_vec
        max_violation = float(np.max(violation)) if violation.size else 0.0
        if max_violation <= 0.0:
            break
        idx = int(np.argmax(violation))
        g_row = g_mat[idx]
        denom = np.dot(g_row, g_row) + 1e-12
        x_vec -= (max_violation / denom) * g_row
    return x_vec


def _find_feasible_start(
    c: np.ndarray,
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: np.ndarray,
    h_vec: np.ndarray,
) -> np.ndarray:
    n = c.shape[0]
    zero_obj = np.zeros(n)
    res = simplex(zero_obj, a_mat, b_vec, g_mat, h_vec)
    if res.status == Status.OPTIMAL and res.x is not None:
        return res.x
    return np.zeros(n)


def _solve_newton(
    hessian: np.ndarray,
    grad: np.ndarray,
    a_mat: Optional[np.ndarray],
    rhs: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    n = hessian.shape[0]
    if a_mat is None or a_mat.size == 0:
        direction = -stable_solve(hessian, grad)
        return direction, np.zeros(0)
    m = a_mat.shape[0]
    kkt_matrix = np.block([[hessian, a_mat.T], [a_mat, np.zeros((m, m))]])
    vec = np.concatenate([-grad, rhs])
    sol = stable_solve(kkt_matrix, vec)
    return sol[:n], sol[n:]


def log_barrier_method(
    c: np.ndarray,
    g_mat: np.ndarray,
    h_vec: np.ndarray,
    a_mat: Optional[np.ndarray] = None,
    b_vec: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    mu0: float = 1.0,
    tol: float = 1e-8,
    maxiter: int = 50,
) -> OptimizeResult:
    """
    Solve ``min c^T x`` subject to ``G x <= h`` and optional equalities via
    a primal log-barrier method.
    """

    c = np.asarray(c, dtype=float).reshape(-1)
    g_mat = np.asarray(g_mat, dtype=float)
    h_vec = np.asarray(h_vec, dtype=float).reshape(-1)
    if g_mat.shape[0] != h_vec.shape[0]:
        raise ValueError("G and h dimension mismatch")
    n = c.shape[0]
    if g_mat.shape[1] != n:
        raise ValueError("G must have the same number of columns as len(c)")

    a_mat = _ensure_array(a_mat, "A")
    b_vec = _ensure_array(b_vec, "b")
    if (a_mat is None) ^ (b_vec is None):
        raise ValueError("A and b must be provided together")
    if a_mat is not None and b_vec is not None and a_mat.shape[0] != b_vec.shape[0]:
        raise ValueError("A and b row mismatch")

    if x0 is None:
        x = _find_feasible_start(c, a_mat, b_vec, g_mat, h_vec)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)
    if x.shape[0] != n:
        raise ValueError("Initial point has wrong dimension")

    x = _project_onto_inequalities(g_mat, h_vec, x)
    slack = h_vec - g_mat @ x
    slack = np.maximum(slack, 1e-6)
    t = 1.0
    mu = mu0 if mu0 > 1.0 else 10.0
    failed_linesearch = False
    for outer in range(1, maxiter + 1):
        for _ in range(30):
            inv_slack = 1.0 / slack
            grad = t * c + g_mat.T @ inv_slack
            hessian = np.zeros((n, n))
            for i in range(g_mat.shape[0]):
                gi = g_mat[i]
                hessian += np.outer(gi, gi) / (slack[i] ** 2)
            rhs_eq = None if a_mat is None else (np.asarray(b_vec) - a_mat @ x)
            try:
                direction, _ = _solve_newton(hessian, grad, a_mat, rhs_eq)
            except np.linalg.LinAlgError:
                return OptimizeResult(
                    x=x,
                    fun=float(c @ x),
                    status=Status.NUMERICAL_ERROR,
                    message="Newton system solve failed",
                    nit=outer,
                )

            decrement = grad @ direction
            if abs(decrement) < tol:
                break

            alpha = 1.0
            g_dot = g_mat @ direction
            positive = g_dot > 0
            if np.any(positive):
                alpha = min(alpha, 0.99 * np.min(slack[positive] / g_dot[positive]))

            phi = t * c @ x - np.sum(np.log(slack))
            for _ in range(25):
                x_trial = x + alpha * direction
                slack_trial = h_vec - g_mat @ x_trial
                if np.any(slack_trial <= 0):
                    alpha *= 0.5
                    continue
                phi_trial = t * c @ x_trial - np.sum(np.log(slack_trial))
                if phi_trial <= phi + 0.25 * alpha * decrement:
                    x = x_trial
                    slack = slack_trial
                    break
                alpha *= 0.5
            else:
                failed_linesearch = True
                break

        if failed_linesearch:
            break

        duality_gap = g_mat.shape[0] / t
        if duality_gap < tol:
            break
        t *= mu

    fun_val = float(c @ x)
    primal_eq = float(np.linalg.norm(a_mat @ x - b_vec, ord=np.inf)) if a_mat is not None else 0.0
    primal_ineq = float(np.linalg.norm(np.maximum(g_mat @ x - h_vec, 0.0), ord=np.inf))
    is_optimal = max(primal_eq, primal_ineq) <= tol and not failed_linesearch
    status = Status.OPTIMAL if is_optimal else Status.MAX_ITER
    return OptimizeResult(
        x=x,
        fun=fun_val,
        status=status,
        message="Barrier iterations completed",
        nit=outer,
        primal_residual=max(primal_eq, primal_ineq),
        slack=h_vec - g_mat @ x,
    )


__all__ = ["log_barrier_method"]

