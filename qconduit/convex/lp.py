"""
Linear programming algorithms: revised simplex and SciPy wrappers.

The revised simplex implementation follows the textbook algorithm outlined in
Nocedal & Wright (Chapter 13) and assumes problems are provided in the generic
form

```
    minimize    c^T x
    subject to  A x = b
                G x <= h
                lb <= x <= ub
```

All constraints are converted to the standard equality form with nonnegative
variables before running a two-phase revised simplex method. Lower bounds are
handled by shifting variables, free variables are represented as the difference
of two nonnegative variables, and upper bounds become explicit inequalities.

Example:
    >>> import numpy as np
    >>> from qconduit.convex.lp import simplex
    >>> c = np.array([-3.0, -5.0])  # maximize 3x + 5y -> minimize negative
    >>> G = np.array([[1.0, 2.0], [3.0, 2.0]])
    >>> h = np.array([4.0, 6.0])
    >>> result = simplex(c, a_mat=None, b_vec=None, g_mat=G, h_vec=h, lb=np.zeros(2))
    >>> result.status
    <Status.OPTIMAL: 'optimal'>
    >>> result.x  # Optimal point (x1, x2) = (2, 1)
    array([2., 1.])

References:
    - Nocedal & Wright, *Numerical Optimization*, 2nd edition, 2006.
    - Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, 1997.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .core import OptimizeResult, Status

try:
    from scipy.optimize import linprog as _scipy_linprog

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - SciPy is optional
    SCIPY_AVAILABLE = False
    _scipy_linprog = None


@dataclass
class _StandardFormLP:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    transform: np.ndarray
    shift: np.ndarray
    const_offset: float
    base_var_count: int
    n_real: int
    n_constraints: int


def _coerce_vector(vec: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise ValueError("Vector dimension mismatch")  # pragma: no cover
    arr = arr.copy()
    return arr


@dataclass
class _SimplexState:
    x: np.ndarray
    objective: float
    basis: List[int]
    iterations: int
    status: Status
    message: str


def _convert_to_standard(
    c: np.ndarray,
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray],
    h_vec: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
    tol: float,
) -> _StandardFormLP:
    c = np.asarray(c, dtype=float).reshape(-1)
    n = c.shape[0]
    if n == 0:
        raise ValueError("Linear program must contain at least one variable")  # pragma: no cover

    def _coerce_matrix(
        mat: Optional[np.ndarray],
        rows: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if mat is None:
            return None
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != n:
            raise ValueError("Matrix dimension mismatch")  # pragma: no cover
        if rows is not None and arr.shape[0] != rows:
            raise ValueError("Matrix row mismatch")  # pragma: no cover
        return arr

    a_matrix = _coerce_matrix(a_mat)
    b_vec_in = None if b_vec is None else np.asarray(b_vec, dtype=float).reshape(-1)
    if (a_matrix is None) ^ (b_vec_in is None):
        raise ValueError("A and b must be provided together")  # pragma: no cover
    if b_vec_in is not None and b_vec_in.shape[0] != a_matrix.shape[0]:
        raise ValueError("A and b dimension mismatch")  # pragma: no cover

    g_matrix = _coerce_matrix(g_mat)
    h_vec_in = None if h_vec is None else np.asarray(h_vec, dtype=float).reshape(-1)
    if (g_matrix is None) ^ (h_vec_in is None):
        raise ValueError("G and h must be provided together")  # pragma: no cover
    if h_vec_in is not None and h_vec_in.shape[0] != g_matrix.shape[0]:
        raise ValueError("G and h dimension mismatch")  # pragma: no cover

    lb_vec = _coerce_vector(lb, n)
    if lb_vec is None:
        lb_vec = np.zeros(n)
    else:
        lb_vec = lb_vec.copy()
    ub_vec = _coerce_vector(ub, n)
    shift = np.zeros(n)
    columns: List[np.ndarray] = []
    const_offset = 0.0

    for i in range(n):
        lb_i = lb_vec[i]
        if np.isfinite(lb_i):
            shift[i] = lb_i
            col = np.zeros(n)
            col[i] = 1.0
            columns.append(col)
            const_offset += c[i] * lb_i
        else:
            col_pos = np.zeros(n)
            col_pos[i] = 1.0
            columns.append(col_pos)
            col_neg = np.zeros(n)
            col_neg[i] = -1.0
            columns.append(col_neg)

    transform = np.column_stack(columns)
    base_var_count = transform.shape[1]
    c_base = transform.T @ c

    def _apply_shift(mat: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return mat @ transform, rhs - mat @ shift

    eq_blocks = []
    rhs_list: List[np.ndarray] = []
    if a_matrix is not None:
        a_eq, b_eq = _apply_shift(a_matrix, b_vec_in)
        eq_blocks.append(a_eq)
        rhs_list.append(b_eq)

    ineq_blocks = []
    ineq_rhs = []
    if g_matrix is not None:
        g_eq, h_eq = _apply_shift(g_matrix, h_vec_in)
        ineq_blocks.append(g_eq)
        ineq_rhs.append(h_eq)

    if ub_vec is not None:
        for idx, ub_i in enumerate(ub_vec):
            if np.isfinite(ub_i):
                row = transform[idx, :]
                rhs = ub_i - shift[idx]
                ineq_blocks.append(row[np.newaxis, :])
                ineq_rhs.append(np.array([rhs]))

    if ineq_blocks:
        g_stack = np.vstack(ineq_blocks)
        h_stack = np.concatenate(ineq_rhs)
    else:
        g_stack = np.zeros((0, base_var_count))
        h_stack = np.zeros(0)

    m_ineq = g_stack.shape[0]
    n_real = base_var_count + m_ineq
    slack_eye = np.eye(m_ineq)
    g_aug = np.hstack([g_stack, slack_eye]) if m_ineq else np.zeros((0, n_real))

    if eq_blocks:
        a_stack = np.vstack(eq_blocks)
        a_aug = np.hstack([a_stack, np.zeros((a_stack.shape[0], m_ineq))])
        rhs_stack = np.concatenate(rhs_list)
    else:
        a_aug = np.zeros((0, n_real))
        rhs_stack = np.zeros(0)

    a_final = np.vstack([a_aug, g_aug]) if a_aug.size or g_aug.size else np.zeros((0, n_real))
    b_final = (
        np.concatenate([rhs_stack, h_stack])
        if rhs_stack.size or h_stack.size
        else np.zeros(0)
    )

    for i in range(b_final.shape[0]):
        if b_final[i] < 0:
            a_final[i, :] *= -1
            b_final[i] *= -1
        if b_final[i] < -tol:
            raise ValueError("Infeasible: negative RHS after normalization")  # pragma: no cover

    c_real = np.concatenate([c_base, np.zeros(m_ineq)])
    return _StandardFormLP(
        A=a_final,
        b=b_final,
        c=c_real,
        transform=transform,
        shift=shift,
        const_offset=float(const_offset),
        base_var_count=base_var_count,
        n_real=n_real,
        n_constraints=a_final.shape[0],
    )


def _revised_simplex(
    a_mat: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: List[int],
    maxiter: int,
    tol: float,
    eligible: Optional[Iterable[int]] = None,
) -> _SimplexState:
    m, n = a_mat.shape
    if m == 0:  # pragma: no cover - unreachable due to earlier check
        return _SimplexState(
            x=np.zeros(n),
            objective=0.0,
            basis=basis,
            iterations=0,
            status=Status.OPTIMAL,
            message="No constraints",
        )
    eligible_set = set(range(n)) if eligible is None else set(eligible)
    nit = 0
    basis = basis.copy()
    while nit < maxiter:
        nit += 1
        try:
            basis_matrix = a_mat[:, basis]
            x_basic = np.linalg.solve(basis_matrix, b)
        except np.linalg.LinAlgError:
            return _SimplexState(
                x=np.zeros(n),
                objective=np.inf,
                basis=basis,
                iterations=nit,
                status=Status.NUMERICAL_ERROR,
                message="Basis matrix singular",
            )
        if np.any(x_basic < -tol):
            # Numerical safeguard
            idx = np.argmin(x_basic)
            return _SimplexState(
                x=np.zeros(n),
                objective=np.inf,
                basis=basis,
                iterations=nit,
                status=Status.NUMERICAL_ERROR,
                message=f"Infeasible basic solution at row {idx}",
            )
        try:
            y = np.linalg.solve(basis_matrix.T, c[basis])
        except np.linalg.LinAlgError:
            return _SimplexState(
                x=np.zeros(n),
                objective=np.inf,
                basis=basis,
                iterations=nit,
                status=Status.NUMERICAL_ERROR,
                message="Dual solve failed",
            )
        reduced = c - a_mat.T @ y
        reduced[basis] = 0.0
        entering = None
        min_value = -tol
        for j in eligible_set:
            if j in basis:
                continue
            if reduced[j] < min_value:
                min_value = reduced[j]
                entering = j
        if entering is None:
            x_full = np.zeros(n)
            x_full[basis] = x_basic
            objective = c @ x_full
            return _SimplexState(
                x=x_full,
                objective=float(objective),
                basis=basis,
                iterations=nit,
                status=Status.OPTIMAL,
                message="Optimal solution reached",
            )
        try:
            direction = np.linalg.solve(basis_matrix, a_mat[:, entering])
        except np.linalg.LinAlgError:
            return _SimplexState(
                x=np.zeros(n),
                objective=np.inf,
                basis=basis,
                iterations=nit,
                status=Status.NUMERICAL_ERROR,
                message="Direction solve failed",
            )
        positive = direction > tol
        if not np.any(positive):
            # Degenerate column; treat as zero reduced cost and search again.
            reduced[entering] = 0.0
            continue
        ratios = np.full_like(x_basic, np.inf)
        ratios[positive] = x_basic[positive] / direction[positive]
        leave_pos = int(np.argmin(ratios))
        basis[leave_pos] = entering
    return _SimplexState(
        x=np.zeros(n),
        objective=np.inf,
        basis=basis,
        iterations=nit,
        status=Status.MAX_ITER,
        message="Maximum iterations exceeded",
    )


def simplex(
    c: np.ndarray,
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray] = None,
    h_vec: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    maxiter: int = 1000,
    tol: float = 1e-9,
) -> OptimizeResult:
    """
    Solve a linear program via a two-phase revised simplex method.
    """

    try:
        standard = _convert_to_standard(c, a_mat, b_vec, g_mat, h_vec, lb, ub, tol)
    except ValueError as exc:
        return OptimizeResult(
            x=None,
            fun=None,
            status=Status.NUMERICAL_ERROR,
            message=str(exc),
            nit=0,
        )

    if standard.n_constraints == 0:
        if np.any(standard.c[: standard.base_var_count] < -tol):
            return OptimizeResult(
                x=None,
                fun=None,
                status=Status.UNBOUNDED,
                message="Objective decreases without constraints",
                nit=0,
            )
        z_base = np.zeros(standard.base_var_count)
        x = standard.shift + standard.transform @ z_base
        fun = float(np.asarray(c, dtype=float) @ x)
        return OptimizeResult(
            x=x,
            fun=fun,
            status=Status.OPTIMAL,
            message="Trivial solution (no constraints)",
            nit=0,
            primal_residual=0.0,
            dual_residual=None,
        )

    a_phase = np.hstack([standard.A, np.eye(standard.n_constraints)])
    c_phase1 = np.concatenate([np.zeros(standard.n_real), np.ones(standard.n_constraints)])
    basis = list(range(standard.n_real, standard.n_real + standard.n_constraints))

    phase1 = _revised_simplex(a_phase, standard.b, c_phase1, basis, maxiter, tol)
    if phase1.status != Status.OPTIMAL:
        return OptimizeResult(
            x=None,
            fun=None,
            status=phase1.status,
            message=f"Phase I failed: {phase1.message}",
            nit=phase1.iterations,
        )
    if phase1.objective > tol:
        return OptimizeResult(
            x=None,
            fun=None,
            status=Status.INFEASIBLE,
            message="Problem infeasible (Phase I objective > 0)",
            nit=phase1.iterations,
        )

    c_phase2 = np.concatenate([standard.c, np.zeros(standard.n_constraints)])
    phase2 = _revised_simplex(
        a_phase,
        standard.b,
        c_phase2,
        basis=phase1.basis,
        maxiter=maxiter,
        tol=tol,
        eligible=range(standard.n_real),
    )
    nit_total = phase1.iterations + phase2.iterations
    if phase2.status != Status.OPTIMAL:
        return OptimizeResult(
            x=None,
            fun=None,
            status=phase2.status,
            message=f"Phase II failed: {phase2.message}",
            nit=nit_total,
        )

    solution = phase2.x[: standard.n_real]
    z_base = solution[: standard.base_var_count]
    x = standard.shift + standard.transform @ z_base
    fun_val = float(np.asarray(c, dtype=float) @ x)

    slack = None
    if g_mat is not None and h_vec is not None:
        slack = np.asarray(h_vec, dtype=float) - np.asarray(g_mat, dtype=float) @ x
    primal_residual = None
    if a_mat is not None and b_vec is not None:
        eq_residual = np.asarray(a_mat, dtype=float) @ x - np.asarray(b_vec, dtype=float)
        r_eq = np.linalg.norm(eq_residual, ord=np.inf)
        primal_residual = r_eq
    if g_mat is not None and h_vec is not None:
        slack_violation = np.minimum(slack, 0.0)
        norm_slack = float(np.linalg.norm(slack_violation, ord=np.inf))
        primal_residual = max(primal_residual or 0.0, norm_slack)

    return OptimizeResult(
        x=x,
        fun=fun_val,
        status=Status.OPTIMAL,
        message="Optimal solution found",
        nit=nit_total,
        primal_residual=primal_residual,
        slack=slack,
    )


def linprog_wrapper(
    c: np.ndarray,
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray] = None,
    h_vec: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    maxiter: int = 1000,
    tol: float = 1e-9,
) -> OptimizeResult:
    """
    Solve an LP via SciPy's ``linprog`` if SciPy is installed.
    """

    if not SCIPY_AVAILABLE:  # pragma: no cover - depends on SciPy
        return OptimizeResult(
            x=None,
            fun=None,
            status=Status.NUMERICAL_ERROR,
            message="SciPy is not available",
            nit=0,
        )
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    bounds = None
    if lb is not None or ub is not None:
        lb_vec = (
            np.asarray(lb, dtype=float).reshape(-1)
            if lb is not None
            else -np.inf * np.ones_like(c_arr, dtype=float)
        )
        ub_vec = (
            np.asarray(ub, dtype=float).reshape(-1)
            if ub is not None
            else np.inf * np.ones_like(c_arr, dtype=float)
        )
        if lb_vec.shape[0] != c_arr.shape[0] or ub_vec.shape[0] != c_arr.shape[0]:
            raise ValueError("Bounds must match variable dimension")
        bounds = list(zip(lb_vec, ub_vec))

    res = _scipy_linprog(
        c=c_arr,
        A_eq=a_mat,
        b_eq=b_vec,
        A_ub=g_mat,
        b_ub=h_vec,
        bounds=bounds,
        options={"maxiter": maxiter, "tol": tol},
        method="highs",
    )
    status = Status.OPTIMAL if res.success else Status.NUMERICAL_ERROR
    message = res.message
    return OptimizeResult(
        x=res.x if res.success else None,
        fun=res.fun if res.success else None,
        status=status,
        message=message,
        nit=res.nit,
        primal_residual=None,
        dual_residual=None,
    )


__all__ = ["simplex", "linprog_wrapper"]

