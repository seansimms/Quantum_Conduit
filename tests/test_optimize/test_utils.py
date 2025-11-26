import numpy as np
import pytest

from qconduit.optimize.utils import approx_grad, approx_hessian, safe_solve


def test_approx_grad_matches_linear_function():
    def fun(x: np.ndarray) -> float:
        return float(3 * x[0] - 2 * x[1])

    grad = approx_grad(fun, np.array([0.2, -0.1]))
    assert np.allclose(grad, np.array([3.0, -2.0]), atol=1e-6)


def test_approx_hessian_matches_quadratic():
    def fun(x: np.ndarray) -> float:
        return float(x[0] ** 2 + 3 * x[1] ** 2)

    hess = approx_hessian(fun, np.array([0.5, -1.5]))
    assert np.allclose(hess, np.diag([2.0, 6.0]), atol=1e-3)


def test_approx_grad_invalid_eps():
    with pytest.raises(ValueError):
        approx_grad(lambda x: float(x[0]), np.array([0.0]), eps=0.0)


def test_safe_solve_regularizes_singular_matrix():
    mat = np.array([[1.0, 1.0], [1.0, 1.0]])
    vec = np.array([1.0, 1.0])
    solution = safe_solve(mat, vec)
    assert np.allclose(mat @ solution, vec, atol=1e-6)

