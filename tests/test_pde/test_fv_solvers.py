from __future__ import annotations

import numpy as np

from qconduit.pde.boundary import BoundaryCondition
from qconduit.pde.fv import FVSolver1D


def _linear_flux(c: float):
    return lambda u: c * u


def test_fv_upwind_translates_square_wave_without_distortion() -> None:
    length = 1.0
    nx = 100
    c = 1.0
    dt = 5e-3
    total_time = 3 * dt * 2  # 0.03 seconds, shift of 3 cells
    shift_cells = int(c * total_time / (length / nx))
    assert shift_cells == 3

    u0 = np.zeros(nx)
    start, end = 20, 40
    u0[start:end] = 1.0

    zero_bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    solver = FVSolver1D(
        flux=_linear_flux(c),
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        boundary_left=zero_bc,
        boundary_right=zero_bc,
        scheme="upwind",
    )

    u_hist = solver.solve(u0)
    final = u_hist[-1]
    mass = final.sum()
    assert mass > 0
    indices = np.arange(nx)
    centroid = np.dot(indices, final) / mass
    expected_centroid = ((start + end - 1) / 2) + shift_cells
    assert abs(centroid - expected_centroid) < 0.5
    peak_idx = start + shift_cells + (end - start) // 2
    assert final[peak_idx] > 0.8


def test_lax_friedrichs_is_more_diffusive_than_upwind() -> None:
    length = 1.0
    nx = 80
    c = 1.0
    dt = 4e-3
    total_time = 2e-2
    zero_bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    u0 = np.zeros(nx)
    u0[25:40] = 1.0

    solver_upwind = FVSolver1D(
        flux=_linear_flux(c),
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        boundary_left=zero_bc,
        boundary_right=zero_bc,
        scheme="upwind",
    )
    solver_lax = FVSolver1D(
        flux=_linear_flux(c),
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        boundary_left=zero_bc,
        boundary_right=zero_bc,
        scheme="lax-friedrichs",
    )

    upwind_final = solver_upwind.solve(u0)[-1]
    lax_final = solver_lax.solve(u0)[-1]
    assert np.linalg.norm(lax_final - upwind_final) > 1e-3
    assert np.sum(lax_final > 0.9) < np.sum(upwind_final > 0.9)


def test_fv_upwind_negative_speed_moves_left() -> None:
    length = 1.0
    nx = 80
    c = -1.0
    dt = 4e-3
    total_time = 2e-2
    zero_bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    u0 = np.zeros(nx)
    u0[40:55] = 1.0
    solver = FVSolver1D(
        flux=_linear_flux(c),
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        boundary_left=zero_bc,
        boundary_right=zero_bc,
        scheme="upwind",
    )
    centers = np.linspace(0.5 * (length / nx), length - 0.5 * (length / nx), nx)
    history = solver.solve(u0)
    centroid_initial = np.dot(centers, u0) / np.sum(u0)
    centroid_final = np.dot(centers, history[-1]) / np.sum(history[-1])
    assert abs((centroid_final - centroid_initial) - c * total_time) < 5e-3

