from __future__ import annotations

import numpy as np
import pytest

from qconduit.pde.boundary import BoundaryCondition
from qconduit.pde.fd import FDSolver1D


def _sine_initial_condition(length: float, nx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length, nx)
    return x, np.sin(np.pi * x / length)


def _gaussian_initial_condition(
    length: float, nx: int, center: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length, nx)
    profile = np.exp(-((x - center) ** 2) / (2 * width**2))
    return x, profile


def test_heat_explicit_matches_analytic_solution() -> None:
    length = 1.0
    nx = 51
    alpha = 0.1
    dt = 1e-4
    total_time = 5e-3
    x, u0 = _sine_initial_condition(length, nx)

    bc = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)
    solver = FDSolver1D(
        equation="heat",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        boundary_left=bc,
        boundary_right=bc,
        scheme="explicit",
    )

    u_hist = solver.solve(u0)
    t = total_time
    analytic = np.sin(np.pi * x / length) * np.exp(-alpha * (np.pi**2) * t / (length**2))
    error = np.max(np.abs(u_hist[-1] - analytic))
    assert error < 5e-3


def test_heat_crank_nicolson_handles_larger_time_steps() -> None:
    length = 1.0
    nx = 51
    alpha = 0.1
    dt = 5e-3  # Much larger than explicit would allow
    total_time = 5e-2
    x, u0 = _sine_initial_condition(length, nx)
    bc = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)

    solver = FDSolver1D(
        equation="heat",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        boundary_left=bc,
        boundary_right=bc,
        scheme="crank-nicolson",
    )

    u_hist = solver.solve(u0)
    analytic = np.sin(np.pi * x / length) * np.exp(-alpha * (np.pi**2) * total_time / (length**2))
    error = np.max(np.abs(u_hist[-1] - analytic))
    assert error < 5e-2  # looser tolerance due to large dt


def test_wave_explicit_standing_mode() -> None:
    length = 1.0
    nx = 101
    c = 1.0
    dt = 5e-4
    total_time = 5e-3
    x, u0 = _sine_initial_condition(length, nx)
    bc = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)

    solver = FDSolver1D(
        equation="wave",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        c=c,
        boundary_left=bc,
        boundary_right=bc,
        scheme="explicit",
    )

    u_hist = solver.solve(u0)
    t = total_time
    analytic = np.sin(np.pi * x / length) * np.cos(np.pi * c * t / length)
    assert np.max(np.abs(u_hist[-1] - analytic)) < 1e-2


def test_fd_solver_rejects_incorrect_initial_shape() -> None:
    length = 1.0
    nx = 51
    alpha = 0.1
    dt = 1e-4
    total_time = 1e-3
    bc = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)
    solver = FDSolver1D(
        equation="heat",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        boundary_left=bc,
        boundary_right=bc,
        scheme="explicit",
    )
    with pytest.raises(ValueError):
        solver.solve(np.zeros(nx + 1))


def test_advection_upwind_transports_gaussian_pulse() -> None:
    length = 1.0
    nx = 101
    c = 1.0
    dt = 5e-4
    total_time = 5e-3
    x, u0 = _gaussian_initial_condition(length, nx, center=0.3, width=0.05)
    zeros = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)

    solver = FDSolver1D(
        equation="advection",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        c=c,
        boundary_left=zeros,
        boundary_right=zeros,
        scheme="upwind",
    )

    u_hist = solver.solve(u0)
    centroid_initial = np.dot(x, u0) / np.sum(u0)
    centroid_final = np.dot(x, u_hist[-1]) / np.sum(u_hist[-1])
    assert abs((centroid_final - centroid_initial) - c * total_time) < 5e-3


def test_advection_lax_is_more_diffusive_than_upwind() -> None:
    length = 1.0
    nx = 101
    c = 1.0
    dt = 5e-4
    total_time = 5e-3
    _, u0 = _gaussian_initial_condition(length, nx, center=0.4, width=0.04)
    zeros = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)

    upwind_solver = FDSolver1D(
        equation="advection",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        c=c,
        boundary_left=zeros,
        boundary_right=zeros,
        scheme="upwind",
    )
    lax_solver = FDSolver1D(
        equation="advection",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        c=c,
        boundary_left=zeros,
        boundary_right=zeros,
        scheme="lax",
    )

    upwind_final = upwind_solver.solve(u0)[-1]
    lax_final = lax_solver.solve(u0)[-1]
    assert np.max(lax_final) < np.max(upwind_final)
    assert np.linalg.norm(lax_final - u0) > np.linalg.norm(upwind_final - u0)

