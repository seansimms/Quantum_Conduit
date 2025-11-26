"""
Quadratic programming solvers.

Implements a deterministic active-set method for small convex quadratic
programs as described in Nocedal & Wright (2006). The solver supports equality
constraints, general linear inequalities, and bound constraints by treating the
bounds as inequalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .core import OptimizeResult, Status
from .utils import project_box, stable_solve, symmetrize


@dataclass
class _ConstraintSystem:
    a_eq: np.ndarray
    b_eq: np.ndarray
    g_mat: np.ndarray
    h_vec: np.ndarray
    lb: Optional[np.ndarray]
    ub: Optional[np.ndarray]


def _assemble_constraints(
    n: int,
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray],
    h_vec: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> _ConstraintSystem:
    def _matrix(mat: Optional[np.ndarray], rows: int | None = None) -> np.ndarray:
        if mat is None:
            return np.zeros((rows or 0, n))
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != n:
            raise ValueError("Constraint matrix dimension mismatch")  # pragma: no cover
        return arr

    def _vector(vec: Optional[np.ndarray], rows: int) -> np.ndarray:
        if vec is None:
            return np.zeros(rows)
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.shape[0] != rows:
            raise ValueError("Constraint vector dimension mismatch")  # pragma: no cover
        return arr

    a_eq = _matrix(a_mat)
    b_eq = _vector(b_vec, a_eq.shape[0])
    g_in = _matrix(g_mat)
    h_in = _vector(h_vec, g_in.shape[0])

    lb_vec = None if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    ub_vec = None if ub is None else np.asarray(ub, dtype=float).reshape(-1)
    if lb_vec is not None and lb_vec.shape[0] != n:
        raise ValueError("Lower bound dimension mismatch")  # pragma: no cover
    if ub_vec is not None and ub_vec.shape[0] != n:
        raise ValueError("Upper bound dimension mismatch")  # pragma: no cover

    bound_rows = []
    bound_rhs = []
    if lb_vec is not None:
        mask = np.isfinite(lb_vec)
        if np.any(mask):
            rows = -np.eye(n)[mask]
            rhs = -lb_vec[mask]
            bound_rows.append(rows)
            bound_rhs.append(rhs)
    if ub_vec is not None:
        mask = np.isfinite(ub_vec)
        if np.any(mask):
            rows = np.eye(n)[mask]
            rhs = ub_vec[mask]
            bound_rows.append(rows)
            bound_rhs.append(rhs)

    if bound_rows:
        g_bounds = np.vstack(bound_rows)
        h_bounds = np.concatenate(bound_rhs)
        g_full = np.vstack([g_in, g_bounds]) if g_in.size else g_bounds
        h_full = np.concatenate([h_in, h_bounds]) if h_in.size else h_bounds
    else:
        g_full = g_in
        h_full = h_in

    return _ConstraintSystem(a_eq=a_eq, b_eq=b_eq, g_mat=g_full, h_vec=h_full, lb=lb_vec, ub=ub_vec)


def _initial_feasible_point(system: _ConstraintSystem, n: int, tol: float) -> np.ndarray:
    if system.a_eq.size:
        # Minimal-norm solution to A x = b
        x, *_ = np.linalg.lstsq(system.a_eq, system.b_eq, rcond=None)
    else:
        x = np.zeros(n)

    x = project_box(x, system.lb, system.ub)

    if system.a_eq.size:
        eq_matrix = system.a_eq
        gram = eq_matrix @ eq_matrix.T
        for _ in range(5):
            residual = system.b_eq - eq_matrix @ x
            if np.linalg.norm(residual, ord=np.inf) <= tol:
                break
            try:
                delta_dual = stable_solve(gram, residual)
            except np.linalg.LinAlgError:
                break
            delta = eq_matrix.T @ delta_dual
            x += delta
            x = project_box(x, system.lb, system.ub)

    if system.g_mat.size:
        for _ in range(20):
            violation = system.g_mat @ x - system.h_vec
            if violation.size == 0:
                break
            max_violation = float(np.max(violation))
            if max_violation <= tol:
                break
            idx = int(np.argmax(violation))
            g_row = system.g_mat[idx]
            denom = np.dot(g_row, g_row) + 1e-12
            x -= (max_violation / denom) * g_row
            x = project_box(x, system.lb, system.ub)
    return x


def _kkt_solve(
    hessian: np.ndarray,
    grad: np.ndarray,
    a_mat: np.ndarray,
    rhs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = hessian.shape[0]
    m = a_mat.shape[0]
    if m == 0:
        try:
            p = -stable_solve(hessian, grad)
        except np.linalg.LinAlgError:
            raise  # pragma: no cover
        return p, np.zeros(0)
    kkt_matrix = np.block([[hessian, a_mat.T], [a_mat, np.zeros((m, m))]])
    vec = np.concatenate([-grad, rhs])
    sol = stable_solve(kkt_matrix, vec)
    return sol[:n], sol[n:]


def active_set_qp(
    hessian: np.ndarray,
    g_vec: np.ndarray,
    a_mat: Optional[np.ndarray] = None,
    b_vec: Optional[np.ndarray] = None,
    g_mat: Optional[np.ndarray] = None,
    h_vec: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
) -> OptimizeResult:
    """
    Solve a convex quadratic program via an active-set method.
    """

    hessian = symmetrize(np.asarray(hessian, dtype=float))
    g_vec = np.asarray(g_vec, dtype=float).reshape(-1)
    n = g_vec.shape[0]
    if hessian.shape != (n, n):
        raise ValueError("H must be square and match the dimension of g")

    system = _assemble_constraints(n, a_mat, b_vec, g_mat, h_vec, lb, ub)
    x = _initial_feasible_point(system, n, tol)

    active_ineq: List[int] = []
    nit = 0

    def objective(vector: np.ndarray) -> float:
        return 0.5 * vector @ (hessian @ vector) + g_vec @ vector

    for nit in range(1, maxiter + 1):
        grad = hessian @ x + g_vec
        if active_ineq:
            a_active = system.g_mat[active_ineq]
            b_active = system.h_vec[active_ineq]
        else:
            a_active = np.zeros((0, n))
            b_active = np.zeros(0)

        if system.a_eq.size:
            a_work = np.vstack([system.a_eq, a_active])
            b_work = np.concatenate([system.b_eq, b_active])
        else:
            a_work = a_active
            b_work = b_active
        rhs = b_work - a_work @ x

        try:
            p, multipliers = _kkt_solve(hessian, grad, a_work, rhs)
        except np.linalg.LinAlgError:
            return OptimizeResult(
                x=x,
                fun=float(objective(x)),
                status=Status.NUMERICAL_ERROR,
                message="KKT solve failed",
                nit=nit,
            )

        if np.linalg.norm(p) <= tol:
            if active_ineq:
                lam = multipliers[system.a_eq.shape[0] :]
                if lam.size and lam.min() < -tol:
                    idx = int(np.argmin(lam))
                    del active_ineq[idx]
                    continue
            slack = system.g_mat @ x - system.h_vec
            if slack.size and slack.max() > tol:
                idx = int(np.argmax(slack))
                if idx not in active_ineq:
                    active_ineq.append(idx)
                    continue
            eq_res = (
                float(np.linalg.norm(system.a_eq @ x - system.b_eq, ord=np.inf))
                if system.a_eq.size
                else 0.0
            )
            ineq_res = (
                float(np.linalg.norm((system.g_mat @ x - system.h_vec).clip(min=0.0), ord=np.inf))
                if system.g_mat.size
                else 0.0
            )
            return OptimizeResult(
                x=x,
                fun=float(objective(x)),
                status=Status.OPTIMAL,
                message="KKT conditions satisfied",
                nit=nit,
                primal_residual=max(eq_res, ineq_res),
            )

        alpha = 1.0
        blocker = -1
        inactive = [i for i in range(system.g_mat.shape[0]) if i not in active_ineq]
        for idx in inactive:
            g_row = system.g_mat[idx]
            denom = g_row @ p
            if denom > tol:
                step = (system.h_vec[idx] - g_row @ x) / denom
                if step < alpha:
                    alpha = step
                    blocker = idx

        x = x + alpha * p
        x = project_box(x, system.lb, system.ub)

        if blocker >= 0 and blocker not in active_ineq:
            active_ineq.append(blocker)

    return OptimizeResult(
        x=x,
        fun=float(objective(x)),
        status=Status.MAX_ITER,
        message="Maximum iterations reached",
        nit=nit,
    )


__all__ = ["active_set_qp"]

