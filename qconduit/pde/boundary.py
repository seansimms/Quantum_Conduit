"""
Boundary-condition abstractions for 1D PDE solvers.

Each `BoundaryCondition` stores a type (Dirichlet or Neumann) and a callable
``fun(x, t)`` that returns either the boundary value (Dirichlet) or the
derivative ``∂u/∂n`` (Neumann) at spatial position ``x`` and time ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

BoundaryType = Literal["dirichlet", "neumann"]


@dataclass(frozen=True)
class BoundaryCondition:
    """Container describing a boundary type and its time-dependent value."""

    type: BoundaryType
    fun: Callable[[float, float], float]

    def value(self, x: float, t: float) -> float:
        """Return the boundary value (Dirichlet) or derivative (Neumann)."""
        return float(self.fun(x, t))

