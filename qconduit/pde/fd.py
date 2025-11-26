"""
Finite-difference solvers for canonical 1D PDEs (heat, wave, advection).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .boundary import BoundaryCondition
from .utils import apply_boundary_values, compute_cfl, linspace_grid


class FDSolver1D:
    """
    Textbook finite-difference solver for 1D PDEs on a uniform grid.

    Parameters mirror standard discretizations described in PDE textbooks.
    The solver accepts Dirichlet or Neumann boundary conditions per domain
    edge and returns the full time-history of the solution.
    """

    _HEAT_SCHEMES = {"explicit", "implicit", "crank-nicolson"}
    _ADVECTION_SCHEMES = {"lax", "upwind"}

    def __init__(
        self,
        equation: Literal["heat", "wave", "advection"],
        length: float,
        nx: int,
        dt: float,
        total_time: float,
        alpha: Optional[float] = None,
        c: Optional[float] = None,
        boundary_left: BoundaryCondition = None,
        boundary_right: BoundaryCondition = None,
        scheme: Literal["explicit", "implicit", "crank-nicolson", "lax", "upwind"] = "explicit",
    ) -> None:
        if boundary_left is None or boundary_right is None:
            raise ValueError("Both left and right boundary conditions must be provided.")

        self.equation = equation
        self.scheme = scheme
        self.length = float(length)
        self.nx = int(nx)
        self.dt = float(dt)
        self.total_time = float(total_time)
        self.alpha = alpha
        self.c = c
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.dx = self.length / (self.nx - 1)
        self.grid = linspace_grid(0.0, self.length, self.nx)
        self.nt = self._compute_num_steps()

        if self.nx < 3:
            raise ValueError("nx must be at least 3 to support second derivatives.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive.")

        self._validate_equation_and_scheme()
        self._precompute_matrices()

    # ------------------------------------------------------------------
    def solve(self, initial_condition: np.ndarray) -> np.ndarray:
        """Run the solver and return an array of shape (nt+1, nx)."""
        u0 = np.asarray(initial_condition, dtype=float)
        if u0.shape != (self.nx,):
            raise ValueError(f"initial_condition must have shape ({self.nx},).")

        history = np.zeros((self.nt + 1, self.nx), dtype=float)
        u_current = u0.copy()
        apply_boundary_values(
            u_current,
            dx=self.dx,
            t=0.0,
            x_coords=(0.0, self.length),
            boundary_left=self.boundary_left,
            boundary_right=self.boundary_right,
        )
        history[0] = u_current

        # For wave equation we need previous state for the leapfrog update.
        u_previous = None
        if self.equation == "wave":
            u_previous = u_current.copy()

        for step in range(1, self.nt + 1):
            t_next = step * self.dt

            if self.equation == "heat":
                u_next = self._advance_heat(u_current, t_next)
            elif self.equation == "wave":
                u_next, u_previous = self._advance_wave(u_current, u_previous)
            else:  # advection
                u_next = self._advance_advection(u_current)

            apply_boundary_values(
                u_next,
                dx=self.dx,
                t=t_next,
                x_coords=(0.0, self.length),
                boundary_left=self.boundary_left,
                boundary_right=self.boundary_right,
            )

            history[step] = u_next
            u_current = u_next

        return history

    # ------------------------------------------------------------------
    def _compute_num_steps(self) -> int:
        nt_float = self.total_time / self.dt
        nt = round(nt_float)
        if abs(nt - nt_float) > 1e-9:
            raise ValueError("total_time must be an integer multiple of dt.")
        return int(nt)

    def _validate_equation_and_scheme(self) -> None:
        if self.equation not in {"heat", "wave", "advection"}:
            raise ValueError(f"Unsupported equation '{self.equation}'.")

        if self.equation == "heat":
            if self.scheme not in self._HEAT_SCHEMES:
                raise ValueError("Heat equation supports explicit/implicit/crank-nicolson.")
            if self.alpha is None or self.alpha <= 0.0:
                raise ValueError("Heat equation requires positive alpha.")
        elif self.equation == "wave":
            if self.scheme != "explicit":
                raise ValueError("Wave equation currently supports the explicit scheme.")
            if self.c is None or self.c == 0.0:
                raise ValueError("Wave equation requires a non-zero wave speed c.")
        else:  # advection
            if self.scheme not in self._ADVECTION_SCHEMES:
                raise ValueError("Advection supports 'lax' or 'upwind' schemes.")
            if self.c is None:
                raise ValueError("Advection requires a transport speed c.")

        if self.scheme == "crank-nicolson" and self.equation != "heat":
            raise ValueError("Crank–Nicolson is implemented for the heat equation only.")

        self._check_explicit_stability()

    def _check_explicit_stability(self) -> None:
        if self.scheme == "explicit" and self.equation == "heat":
            r = self.alpha * self.dt / (self.dx**2)
            if r > 0.5 + 1e-12:
                raise ValueError("Explicit heat scheme unstable: alpha * dt / dx^2 must be <= 0.5.")
        elif self.scheme == "explicit" and self.equation == "wave":
            cfl = compute_cfl(self.dx, self.dt, self.c)
            if cfl > 1.0 + 1e-12:
                raise ValueError("Wave equation violates CFL <= 1 for explicit scheme.")
        elif self.equation == "advection":
            cfl = compute_cfl(self.dx, self.dt, self.c)
            if cfl > 1.0 + 1e-12:
                raise ValueError("Advection explicit schemes require CFL <= 1.")

    # ------------------------------------------------------------------
    def _precompute_matrices(self) -> None:
        self._lhs_matrix = None
        self._rhs_matrix = None

        if self.equation != "heat":
            return

        r = self.alpha * self.dt / (self.dx**2)
        identity = np.eye(self.nx)
        laplacian = np.zeros((self.nx, self.nx))
        for i in range(1, self.nx - 1):
            laplacian[i, i - 1] = 1.0
            laplacian[i, i] = -2.0
            laplacian[i, i + 1] = 1.0

        if self.scheme == "implicit":
            lhs = identity - r * laplacian
            self._apply_boundary_structure(lhs)
            self._lhs_matrix = lhs
        elif self.scheme == "crank-nicolson":
            lhs = identity - 0.5 * r * laplacian
            rhs = identity + 0.5 * r * laplacian
            self._apply_boundary_structure(lhs)
            self._apply_boundary_structure(rhs)
            self._lhs_matrix = lhs
            self._rhs_matrix = rhs

    def _apply_boundary_structure(self, matrix: np.ndarray) -> None:
        # Left boundary row
        matrix[0, :] = 0.0
        if self.boundary_left.type == "dirichlet":
            matrix[0, 0] = 1.0
        else:  # Neumann
            matrix[0, 0] = -1.0
            matrix[0, 1] = 1.0

        # Right boundary row
        matrix[-1, :] = 0.0
        if self.boundary_right.type == "dirichlet":
            matrix[-1, -1] = 1.0
        else:
            matrix[-1, -2] = -1.0
            matrix[-1, -1] = 1.0

    def _apply_boundary_rhs(self, rhs: np.ndarray, t: float) -> None:
        if self.boundary_left.type == "dirichlet":
            rhs[0] = self.boundary_left.value(0.0, t)
        else:
            rhs[0] = self.boundary_left.value(0.0, t) * self.dx

        if self.boundary_right.type == "dirichlet":
            rhs[-1] = self.boundary_right.value(self.length, t)
        else:
            rhs[-1] = self.boundary_right.value(self.length, t) * self.dx

    # ------------------------------------------------------------------
    def _advance_heat(self, u_current: np.ndarray, t_next: float) -> np.ndarray:
        if self.scheme == "explicit":
            return self._advance_heat_explicit(u_current)
        if self.scheme == "implicit":
            rhs = u_current.copy()
            self._apply_boundary_rhs(rhs, t_next)
            return np.linalg.solve(self._lhs_matrix, rhs)
        # Crank–Nicolson
        rhs = self._rhs_matrix @ u_current
        self._apply_boundary_rhs(rhs, t_next)
        return np.linalg.solve(self._lhs_matrix, rhs)

    def _advance_heat_explicit(self, u_current: np.ndarray) -> np.ndarray:
        u_next = u_current.copy()
        coeff = self.alpha * self.dt / (self.dx**2)
        for i in range(1, self.nx - 1):
            u_next[i] = (
                u_current[i]
                + coeff * (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1])
            )
        return u_next

    def _advance_wave(
        self, u_current: np.ndarray, u_previous: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u_next = np.zeros_like(u_current)
        cfl_sq = (self.c * self.dt / self.dx) ** 2

        for i in range(1, self.nx - 1):
            u_next[i] = (
                2.0 * u_current[i]
                - u_previous[i]
                + cfl_sq * (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1])
            )

        # First step bootstrap assumes zero initial velocity.
        if np.allclose(u_previous, u_current):
            for i in range(1, self.nx - 1):
                u_next[i] = (
                    u_current[i]
                    + 0.5 * cfl_sq * (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1])
                )

        return u_next, u_current

    def _advance_advection(self, u_current: np.ndarray) -> np.ndarray:
        if self.scheme == "lax":
            return self._advance_advection_lax(u_current)
        return self._advance_advection_upwind(u_current)

    def _advance_advection_lax(self, u_current: np.ndarray) -> np.ndarray:
        u_next = u_current.copy()
        lam = self.c * self.dt / self.dx
        for i in range(1, self.nx - 1):
            u_next[i] = 0.5 * (u_current[i + 1] + u_current[i - 1]) - 0.5 * lam * (
                u_current[i + 1] - u_current[i - 1]
            )
        return u_next

    def _advance_advection_upwind(self, u_current: np.ndarray) -> np.ndarray:
        u_next = u_current.copy()
        cfl = abs(self.c) * self.dt / self.dx
        direction = 1.0 if self.c >= 0 else -1.0

        for i in range(1, self.nx - 1):
            if direction >= 0:
                u_next[i] = u_current[i] - cfl * (u_current[i] - u_current[i - 1])
            else:
                u_next[i] = u_current[i] - cfl * (u_current[i + 1] - u_current[i])
        return u_next

