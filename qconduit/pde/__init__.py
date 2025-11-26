"""
Deterministic textbook PDE solvers built on explicit finite-difference (FD)
and finite-volume (FV) discretizations.

The module currently provides:

* `FDSolver1D` – heat, wave, and advection equations on a uniform 1D grid
  with explicit, implicit, Crank–Nicolson, Lax–Friedrichs, and upwind schemes.
* `FVSolver1D` – first-order FV solvers for 1D conservation laws with
  upwind or Lax–Friedrichs numerical fluxes.
* Boundary-condition abstractions for Dirichlet and Neumann data.
* CFL utilities to reason about stability of explicit schemes.

Limitations: grids are uniform, domains are 1D, and computations run on CPU
with NumPy only. Boundary conditions are limited to textbook Dirichlet or
Neumann data. All solvers are deterministic and reproducible.

Example
-------
>>> import numpy as np
>>> from qconduit.pde.boundary import BoundaryCondition
>>> from qconduit.pde.fd import FDSolver1D
>>>
>>> L = 1.0
>>> nx = 50
>>> dt = 1e-4
>>> total_time = 1e-2
>>> alpha = 0.01
>>> x = np.linspace(0.0, L, nx)
>>>
>>> bc_left = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)
>>> bc_right = BoundaryCondition("dirichlet", lambda x_val, t: 0.0)
>>> solver = FDSolver1D(
...     equation="heat",
...     length=L,
...     nx=nx,
...     dt=dt,
...     total_time=total_time,
...     alpha=alpha,
...     boundary_left=bc_left,
...     boundary_right=bc_right,
...     scheme="crank-nicolson",
... )
>>> u0 = np.sin(np.pi * x / L)
>>> u_history = solver.solve(u0)
>>> u_history.shape
(101, 50)
"""

from .boundary import BoundaryCondition, BoundaryType
from .fd import FDSolver1D
from .fv import FVSolver1D
from .utils import compute_cfl, is_stable_explicit, linspace_grid

__all__ = [
    "BoundaryCondition",
    "BoundaryType",
    "FDSolver1D",
    "FVSolver1D",
    "compute_cfl",
    "is_stable_explicit",
    "linspace_grid",
]

