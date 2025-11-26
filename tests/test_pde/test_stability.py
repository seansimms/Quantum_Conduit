from __future__ import annotations

import numpy as np
import pytest

from qconduit.pde.boundary import BoundaryCondition
from qconduit.pde.fd import FDSolver1D
from qconduit.pde.fv import FVSolver1D
from qconduit.pde.utils import (
    apply_boundary_values,
    compute_cfl,
    is_stable_explicit,
    linspace_grid,
)


def test_compute_cfl_and_stability_helpers() -> None:
    cfl = compute_cfl(dx=0.1, dt=0.02, wave_speed=2.0)
    assert cfl == pytest.approx(0.4)
    assert is_stable_explicit(cfl)
    assert not is_stable_explicit(1.1)


def test_heat_solver_rejects_unstable_explicit_parameters() -> None:
    bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    with pytest.raises(ValueError):
        FDSolver1D(
            equation="heat",
            length=1.0,
            nx=21,
            dt=5e-3,
            total_time=1e-2,
            alpha=1.0,
            boundary_left=bc,
            boundary_right=bc,
            scheme="explicit",
        )


def test_advection_solver_rejects_large_cfl() -> None:
    bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    with pytest.raises(ValueError):
        FDSolver1D(
            equation="advection",
            length=1.0,
            nx=21,
            dt=5e-2,
            total_time=1e-1,
            c=2.0,
            boundary_left=bc,
            boundary_right=bc,
            scheme="upwind",
        )


def test_utils_validation_helpers() -> None:
    with pytest.raises(ValueError):
        compute_cfl(dx=0.0, dt=0.1, wave_speed=1.0)
    with pytest.raises(ValueError):
        linspace_grid(0.0, 1.0, 1)

    arr = np.array([1.0])
    zero_bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    with pytest.raises(ValueError):
        apply_boundary_values(
            arr,
            dx=0.1,
            t=0.0,
            x_coords=(0.0, 1.0),
            boundary_left=zero_bc,
            boundary_right=zero_bc,
        )


def test_fd_solver_requires_valid_configuration() -> None:
    bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    with pytest.raises(ValueError):
        FDSolver1D(
            equation="heat",
            length=1.0,
            nx=51,
            dt=1e-3,
            total_time=1e-2,
            alpha=0.1,
            boundary_left=None,
            boundary_right=bc,
            scheme="explicit",
        )

    with pytest.raises(ValueError):
        FDSolver1D(
            equation="heat",
            length=1.0,
            nx=51,
            dt=1e-3,
            total_time=1.2345e-2,
            alpha=0.1,
            boundary_left=bc,
            boundary_right=bc,
            scheme="explicit",
        )


def test_fv_solver_validates_inputs() -> None:
    zero_bc = BoundaryCondition("dirichlet", lambda x, t: 0.0)
    with pytest.raises(ValueError):
        FVSolver1D(
            flux=lambda u: u,
            length=1.0,
            nx=1,
            dt=1e-3,
            total_time=1e-2,
            boundary_left=zero_bc,
            boundary_right=zero_bc,
        )

    with pytest.raises(ValueError):
        FVSolver1D(
            flux=lambda u: u,
            length=1.0,
            nx=10,
            dt=0.0,
            total_time=1e-2,
            boundary_left=zero_bc,
            boundary_right=zero_bc,
        )

    with pytest.raises(ValueError):
        FVSolver1D(
            flux=lambda u: u,
            length=1.0,
            nx=10,
            dt=1e-3,
            total_time=1.234e-2,
            boundary_left=zero_bc,
            boundary_right=zero_bc,
        )
    with pytest.raises(ValueError):
        FVSolver1D(
            flux=lambda u: u,
            length=1.0,
            nx=10,
            dt=1e-3,
            total_time=1e-2,
            boundary_left=zero_bc,
            boundary_right=zero_bc,
            scheme="godunov",
        )

    solver = FVSolver1D(
        flux=lambda u: u,
        length=1.0,
        nx=10,
        dt=1e-3,
        total_time=1e-2,
        boundary_left=zero_bc,
        boundary_right=zero_bc,
    )
    with pytest.raises(ValueError):
        solver.solve(np.zeros(11))

