"""
Karush-Kuhn-Tucker diagnostics for convex programs.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .utils import symmetrize


def kkt_residuals(
    hessian: Optional[np.ndarray],
    g_vec: Optional[np.ndarray],
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray],
    h_vec: Optional[np.ndarray],
    x: np.ndarray,
    lam: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute norms of KKT residuals for a QP or LP.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    hess = (
        symmetrize(np.zeros((n, n)))
        if hessian is None
        else symmetrize(np.asarray(hessian, dtype=float))
    )
    g_lin = np.zeros(n) if g_vec is None else np.asarray(g_vec, dtype=float).reshape(-1)

    stationarity = hess @ x + g_lin
    if a_mat is not None:
        mat_eq = np.asarray(a_mat, dtype=float)
        lam_vec = (
            np.zeros(mat_eq.shape[0])
            if lam is None
            else np.asarray(lam, dtype=float).reshape(-1)
        )
        stationarity += mat_eq.T @ lam_vec
        primal_eq = float(np.linalg.norm(mat_eq @ x - np.asarray(b_vec, dtype=float), ord=np.inf))
    else:
        primal_eq = 0.0

    complementary = 0.0
    primal_ineq = 0.0
    if g_mat is not None:
        mat_ineq = np.asarray(g_mat, dtype=float)
        h_rhs = (
            np.zeros(mat_ineq.shape[0])
            if h_vec is None
            else np.asarray(h_vec, dtype=float).reshape(-1)
        )
        slack = h_rhs - mat_ineq @ x
        mu_vec = (
            np.zeros(mat_ineq.shape[0])
            if mu is None
            else np.asarray(mu, dtype=float).reshape(-1)
        )
        stationarity += mat_ineq.T @ mu_vec
        primal_ineq = float(np.linalg.norm(np.minimum(slack, 0.0), ord=np.inf))
        complementary = float(np.linalg.norm(slack * mu_vec, ord=np.inf))

    dual_residual = float(np.linalg.norm(stationarity, ord=np.inf))
    return {
        "primal_eq": primal_eq,
        "primal_ineq": primal_ineq,
        "dual": dual_residual,
        "complementary": complementary,
    }


def is_kkt_optimal(
    hessian: Optional[np.ndarray],
    g_vec: Optional[np.ndarray],
    a_mat: Optional[np.ndarray],
    b_vec: Optional[np.ndarray],
    g_mat: Optional[np.ndarray],
    h_vec: Optional[np.ndarray],
    x: np.ndarray,
    lam: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> bool:
    """
    Return True if all KKT residuals are below ``tol``.
    """

    residuals = kkt_residuals(hessian, g_vec, a_mat, b_vec, g_mat, h_vec, x, lam, mu)
    return all(value <= tol for value in residuals.values())


__all__ = ["kkt_residuals", "is_kkt_optimal"]

