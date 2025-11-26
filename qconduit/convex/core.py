"""
Core problem and result dataclasses for the convex optimization module.

The module standardizes problem descriptions for LPs and QPs so that the
algorithm implementations operate on the same data containers. Equality
constraints use the pair ``(A, b)`` to represent ``A x = b`` while inequality
constraints use ``(G, h)`` to represent ``G x <= h``. Bounds ``lb`` and ``ub``
are optional element-wise vectors; ``None`` denotes a free bound whereas
``np.inf`` or ``-np.inf`` can be used to represent one-sided bounds.

References:
    - Boyd & Vandenberghe, *Convex Optimization* (2004)
    - Nocedal & Wright, *Numerical Optimization* (2006)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Status(Enum):
    """Solution status for optimization routines."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MAX_ITER = "max_iter"
    NUMERICAL_ERROR = "numerical_error"


@dataclass
class LPProblem:
    """
    Linear program in canonical form.

    The standard convention is to minimize ``c^T x`` subject to equality
    constraints ``A x = b`` and inequality constraints ``G x <= h`` with
    optional box bounds. The simplex implementation converts the supplied data
    into the equality-only standard form that the algorithm requires.
    """

    c: np.ndarray
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None


@dataclass
class QPProblem:
    """
    Quadratic program with convex quadratic and optional linear constraints.

    The quadratic term ``H`` should be symmetric positive semidefinite for
    deterministic convergence guarantees. Equality and inequality constraints
    follow the same convention as :class:`LPProblem`.
    """

    H: np.ndarray
    g: np.ndarray
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None


@dataclass
class OptimizeResult:
    """
    Solution container shared across all solvers.

    Attributes:
        x: Primal solution vector (or ``None`` if unavailable).
        fun: Objective value at ``x`` (or ``None`` when not computed).
        status: Enumeration describing solver exit.
        message: Human-readable string explaining the status.
        nit: Number of outer iterations performed.
        primal_residual: Norm of primal feasibility residual, if computed.
        dual_residual: Norm of dual feasibility residual, if computed.
        slack: Slack variables for inequality constraints when available.
    """

    x: Optional[np.ndarray]
    fun: Optional[float]
    status: Status
    message: str
    nit: int
    primal_residual: Optional[float] = None
    dual_residual: Optional[float] = None
    slack: Optional[np.ndarray] = None


__all__ = ["Status", "LPProblem", "QPProblem", "OptimizeResult"]

