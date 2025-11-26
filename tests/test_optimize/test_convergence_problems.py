import numpy as np

from qconduit.optimize import (
    Problem,
    bfgs,
    gradient_descent,
    lbfgs,
    newton_method,
)


def rosenbrock(x: np.ndarray) -> float:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ]
    )


def rosenbrock_hess(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200],
        ]
    )


def himmelblau(x: np.ndarray) -> float:
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
            2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7),
        ]
    )


def test_bfgs_and_lbfgs_rosenbrock():
    problem = Problem(fun=rosenbrock, grad=rosenbrock_grad, dim=2)
    x0 = np.array([-1.2, 1.0])
    res_bfgs = bfgs(problem, x0, maxiter=300)
    res_lbfgs = lbfgs(problem, x0, m=7, maxiter=300)
    for res in (res_bfgs, res_lbfgs):
        assert res.success
        assert np.allclose(res.x, np.ones(2), atol=1e-6)
        assert res.fun < 1e-10


def test_newton_with_hessian_converges_fast():
    problem = Problem(
        fun=rosenbrock,
        grad=rosenbrock_grad,
        hess=rosenbrock_hess,
        dim=2,
    )
    res = newton_method(problem, np.array([-1.1, 0.9]), maxiter=20, lambda_reg=1e-4)
    assert res.success
    assert res.nit < 15
    assert res.fun < 1e-12


def test_gradient_descent_quadratic_matches_solution():
    A = np.array([[2.0, 0.2], [0.2, 3.0]])
    b = np.array([1.0, 1.0])

    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ (A @ x) - b @ x

    def grad(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    problem = Problem(fun=fun, grad=grad, dim=2)
    expected = np.linalg.solve(A, b)
    res = gradient_descent(problem, np.array([0.0, 0.0]), lr=0.2, maxiter=300)
    assert res.success
    assert np.allclose(res.x, expected, atol=1e-6)


def test_himmelblau_minimum_found():
    problem = Problem(fun=himmelblau, grad=himmelblau_grad, dim=2)
    x0 = np.array([3.0, 1.5])
    res = lbfgs(problem, x0, maxiter=400)
    expected = np.array([3.0, 2.0])
    assert res.success
    assert np.allclose(res.x, expected, atol=1e-5)

