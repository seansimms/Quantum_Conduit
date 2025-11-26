"""Core interfaces shared across classical optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient = Callable[[Array], Array]
Hessian = Callable[[Array], Array]

RTOL = 1e-8
ATOL = 1e-10


@dataclass(frozen=True)
class Problem:
    """Container describing an optimization problem."""

    fun: Objective
    grad: Optional[Gradient] = None
    hess: Optional[Hessian] = None
    dim: Optional[int] = None


@dataclass
class OptimizeResult:
    """Standard result object returned by all optimizers in this module."""

    x: Array
    fun: float
    nit: int
    success: bool
    message: str
    grad_norm: float
    nfev: int
    njev: int
    nhev: int
    history: List[Array] = field(default_factory=list)


def check_convergence(grad_norm: float, tol: float) -> bool:
    """Return True if gradient norm satisfies tolerance."""
    return grad_norm <= max(tol, ATOL)


__all__ = [
    "Array",
    "Objective",
    "Gradient",
    "Hessian",
    "Problem",
    "OptimizeResult",
    "check_convergence",
    "RTOL",
    "ATOL",
]

