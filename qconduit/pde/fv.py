"""
Finite-volume solvers for 1D conservation laws using textbook schemes.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .boundary import BoundaryCondition
from .utils import compute_cfl


class FVSolver1D:
    """First-order finite-volume solver on a uniform grid."""

    def __init__(
        self,
        flux: Callable[[float], float],
        length: float,
        nx: int,
        dt: float,
        total_time: float,
        boundary_left: BoundaryCondition,
        boundary_right: BoundaryCondition,
        scheme: Literal["upwind", "lax-friedrichs"] = "upwind",
    ) -> None:
        if nx < 2:
            raise ValueError("nx must be at least 2 for finite-volume grids.")
        if dt <= 0.0 or total_time <= 0.0:
            raise ValueError("dt and total_time must be positive.")
        if total_time / dt - round(total_time / dt) > 1e-9:
            raise ValueError("total_time must be an integer multiple of dt.")

        self.flux = flux
        self.length = float(length)
        self.nx = int(nx)
        self.dt = float(dt)
        self.total_time = float(total_time)
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.scheme = scheme
        self.dx = self.length / self.nx
        self.nt = int(round(self.total_time / self.dt))

        if scheme not in {"upwind", "lax-friedrichs"}:
            raise ValueError("scheme must be 'upwind' or 'lax-friedrichs'.")

    # ------------------------------------------------------------------
    def solve(self, initial_condition: np.ndarray) -> np.ndarray:
        """Return an array of shape (nt+1, nx) with cell averages over time."""
        u = np.asarray(initial_condition, dtype=float)
        if u.shape != (self.nx,):
            raise ValueError(f"initial_condition must have shape ({self.nx},).")

        self._check_cfl(u)

        history = np.zeros((self.nt + 1, self.nx))
        history[0] = u

        for step in range(1, self.nt + 1):
            t = (step - 1) * self.dt
            u = self._advance(u, t)
            history[step] = u

        return history

    # ------------------------------------------------------------------
    def _check_cfl(self, u: np.ndarray) -> None:
        speeds = []
        for i in range(self.nx - 1):
            speeds.append(abs(self._estimate_speed(u[i], u[i + 1])))
        max_speed = max(speeds) if speeds else abs(self._estimate_speed(u[0], u[0] + 1e-12))
        cfl = compute_cfl(self.dx, self.dt, max_speed)
        if cfl > 1.0 + 1e-12:
            raise ValueError("Finite-volume solver requires CFL <= 1 based on flux derivative.")

    def _advance(self, u: np.ndarray, t: float) -> np.ndarray:
        extended = self._build_extended_state(u, t)
        fluxes = np.zeros(self.nx + 1)
        for i in range(self.nx + 1):
            left_state = extended[i]
            right_state = extended[i + 1]
            fluxes[i] = self._numerical_flux(left_state, right_state)

        u_next = u.copy()
        scale = self.dt / self.dx
        for i in range(self.nx):
            u_next[i] = u[i] - scale * (fluxes[i + 1] - fluxes[i])
        return u_next

    def _build_extended_state(self, u: np.ndarray, t: float) -> np.ndarray:
        extended = np.zeros(self.nx + 2)
        extended[1:-1] = u

        extended[0] = self._ghost_value(
            neighbor=u[0],
            boundary=self.boundary_left,
            derivative_sign=-1.0,
            x=0.0,
            t=t,
            dx=self.dx,
        )
        extended[-1] = self._ghost_value(
            neighbor=u[-1],
            boundary=self.boundary_right,
            derivative_sign=1.0,
            x=self.length,
            t=t,
            dx=self.dx,
        )
        return extended

    @staticmethod
    def _ghost_value(
        neighbor: float,
        boundary: BoundaryCondition,
        derivative_sign: float,
        x: float,
        t: float,
        dx: float,
    ) -> float:
        if boundary.type == "dirichlet":
            return boundary.value(x, t)
        derivative = boundary.value(x, t)
        return neighbor + derivative_sign * derivative * dx

    def _numerical_flux(self, left: float, right: float) -> float:
        if self.scheme == "upwind":
            speed = self._estimate_speed(left, right)
            if speed >= 0:
                return self.flux(left)
            return self.flux(right)

        # Lax–Friedrichs with alpha = dx / dt for additional diffusion.
        alpha = self.dx / self.dt
        return 0.5 * (self.flux(left) + self.flux(right)) - 0.5 * alpha * (right - left)

    def _estimate_speed(self, left: float, right: float) -> float:
        if abs(right - left) < 1e-12:
            eps = 1e-8
            return (self.flux(left + eps) - self.flux(left - eps)) / (2.0 * eps)
        return (self.flux(right) - self.flux(left)) / (right - left)

