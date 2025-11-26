"""
Convex optimization textbook algorithms and utilities.

This subpackage provides deterministic reference implementations of the
algorithms requested in G26, including simplex, interior-point, active-set QP,
projected methods, KKT diagnostics, and supporting linear algebra helpers.

Modules are intentionally NumPy-first so that they can run without SciPy,
although several functions expose optional SciPy-powered fallbacks when the
package is available.
"""

from . import core, ipm, kkt, lp, projected, qp, utils
from .core import LPProblem, OptimizeResult, QPProblem, Status
from .ipm import log_barrier_method
from .kkt import is_kkt_optimal, kkt_residuals
from .lp import linprog_wrapper, simplex
from .projected import projected_gradient, projected_newton
from .qp import active_set_qp

__all__ = [
    "core",
    "ipm",
    "kkt",
    "lp",
    "projected",
    "qp",
    "utils",
    # Core types
    "Status",
    "OptimizeResult",
    "LPProblem",
    "QPProblem",
    # Algorithms
    "simplex",
    "linprog_wrapper",
    "active_set_qp",
    "log_barrier_method",
    "projected_gradient",
    "projected_newton",
    "kkt_residuals",
    "is_kkt_optimal",
]

