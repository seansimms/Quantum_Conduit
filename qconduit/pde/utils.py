"""
Utility helpers for PDE solvers: CFL computations, grids, and boundaries.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .boundary import BoundaryCondition


def compute_cfl(dx: float, dt: float, wave_speed: float) -> float:
    """Return the Courant–Friedrichs–Lewy number |c| dt / dx."""
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive.")
    return abs(wave_speed) * dt / dx


def is_stable_explicit(cfl: float, limit: float = 1.0) -> bool:
    """Return True if an explicit scheme is stable under the provided limit."""
    return cfl <= limit + 1e-12


def linspace_grid(x0: float, x1: float, nx: int) -> np.ndarray:
    """Create a uniform grid with `nx` points between `x0` and `x1`."""
    if nx < 2:
        raise ValueError("nx must be at least 2 to form a grid.")
    return np.linspace(float(x0), float(x1), int(nx))


def apply_boundary_values(
    u: np.ndarray,
    dx: float,
    t: float,
    x_coords: Tuple[float, float],
    boundary_left: BoundaryCondition,
    boundary_right: BoundaryCondition,
) -> None:
    """
    Mutate `u` in-place to satisfy Dirichlet/Neumann boundary conditions.

    Parameters
    ----------
    u:
        Solution array including boundary nodes.
    dx:
        Uniform grid spacing.
    t:
        Current simulation time.
    x_coords:
        Tuple containing (x_left, x_right).
    boundary_left / boundary_right:
        BoundaryCondition objects describing each domain edge.
    """

    if u.size < 2:
        raise ValueError("Boundary application requires at least two grid points.")

    x_left, x_right = x_coords

    # Left boundary
    if boundary_left.type == "dirichlet":
        u[0] = boundary_left.value(x_left, t)
    else:
        # Neumann derivative: (u_1 - u_0) / dx = g -> u_0 = u_1 - g * dx
        derivative = boundary_left.value(x_left, t)
        u[0] = u[1] - derivative * dx

    # Right boundary
    if boundary_right.type == "dirichlet":
        u[-1] = boundary_right.value(x_right, t)
    else:
        derivative = boundary_right.value(x_right, t)
        u[-1] = u[-2] + derivative * dx

