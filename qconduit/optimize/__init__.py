"""Deterministic classical optimization algorithms for qconduit.

Example
-------
>>> import numpy as np
>>> from qconduit.optimize import Problem, bfgs
>>> def rosen(x):
...     return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
>>> def rosen_grad(x):
...     return np.array([
...         -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
...         200 * (x[1] - x[0] ** 2),
...     ])
>>> problem = Problem(fun=rosen, grad=rosen_grad, dim=2)
>>> res = bfgs(problem, np.array([-1.2, 1.0]))
>>> round(res.fun, 6)
0.0
"""

from .core import ATOL, RTOL, OptimizeResult, Problem, check_convergence
from .gradient import gradient_descent
from .line_search import backtracking_armijo, wolfe_line_search
from .newton import newton_method
from .quasi_newton import bfgs, lbfgs
from .trust_region import trust_region
from .utils import approx_grad, approx_hessian, is_pos_def, safe_solve

__all__ = [
    "ATOL",
    "Problem",
    "OptimizeResult",
    "RTOL",
    "approx_grad",
    "approx_hessian",
    "backtracking_armijo",
    "bfgs",
    "check_convergence",
    "gradient_descent",
    "is_pos_def",
    "lbfgs",
    "newton_method",
    "safe_solve",
    "trust_region",
    "wolfe_line_search",
]

