from __future__ import annotations

import numpy as np

from qconduit.pde.boundary import BoundaryCondition
from qconduit.pde.fd import FDSolver1D
from qconduit.pde.utils import apply_boundary_values


def test_dirichlet_boundary_enforces_linear_ramp() -> None:
    length = 1.0
    nx = 60
    alpha = 0.2
    dt = 1e-3
    total_time = 2.0

    left = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    right = BoundaryCondition("dirichlet", lambda x, t: 1.0)
    solver = FDSolver1D(
        equation="heat",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        boundary_left=left,
        boundary_right=right,
        scheme="implicit",
    )

    u0 = np.zeros(nx)
    final_state = solver.solve(u0)[-1]
    expected = np.linspace(0.0, 1.0, nx)
    assert np.max(np.abs(final_state - expected)) < 5e-2


def test_neumann_boundary_preserves_zero_flux() -> None:
    length = 1.0
    nx = 80
    alpha = 0.05
    dt = 5e-4
    total_time = 5e-2
    x = np.linspace(0.0, length, nx)

    left = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)
    right = BoundaryCondition("neumann", lambda x_val, t: 0.0)
    solver = FDSolver1D(
        equation="heat",
        length=length,
        nx=nx,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        boundary_left=left,
        boundary_right=right,
        scheme="explicit",
    )

    # Localized bump near the insulated boundary.
    u0 = np.exp(-200 * (x - 0.8) ** 2)
    u_hist = solver.solve(u0)
    final_state = u_hist[-1]
    dx = length / (nx - 1)
    right_flux = (final_state[-1] - final_state[-2]) / dx
    assert abs(right_flux) < 1e-2


def test_apply_boundary_values_handles_left_neumann() -> None:
    dx = 0.1
    arr = np.array([0.0, 0.5, 1.0])
    left = BoundaryCondition("neumann", lambda x, t: 2.0)
    right = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    apply_boundary_values(
        arr,
        dx=dx,
        t=0.0,
        x_coords=(0.0, 1.0),
        boundary_left=left,
        boundary_right=right,
    )
    assert np.isclose(arr[0], arr[1] - 2.0 * dx)

