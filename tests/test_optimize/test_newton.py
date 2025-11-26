import numpy as np
import pytest

from qconduit.optimize import (
    Problem,
    backtracking_armijo,
    newton_method,
    wolfe_line_search,
)
from qconduit.optimize import newton as newton_module


def test_newton_solves_quadratic_in_one_step():
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    b = np.array([1.0, -1.0])

    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ (A @ x) - b @ x

    def grad(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    def hess(_: np.ndarray) -> np.ndarray:
        return A

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    x0 = np.array([2.0, 2.0])
    res = newton_method(problem, x0, maxiter=5, use_line_search=False)
    expected = np.linalg.solve(A, b)
    assert res.success
    assert res.nit <= 2
    assert np.allclose(res.x, expected, atol=1e-10)


def test_damped_newton_rosenbrock():
    def rosen(x: np.ndarray) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def rosen_grad(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        )

    def rosen_hess(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                [-400 * x[0], 200],
            ]
        )

    problem = Problem(fun=rosen, grad=rosen_grad, hess=rosen_hess, dim=2)
    res = newton_method(
        problem,
        np.array([-1.2, 1.0]),
        maxiter=50,
        lambda_reg=1e-3,
    )
    assert res.success
    assert res.fun < 1e-8


def test_newton_fallback_gradient_and_hessian():
    def fun(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    problem = Problem(fun=fun, dim=2)
    res = newton_method(problem, np.array([2.5, -3.0]), maxiter=20)
    assert res.success
    assert res.grad_norm < 1e-6


def test_newton_with_line_search_branch():
    def fun(x: np.ndarray) -> float:
        return float((x[0] - 2) ** 2 + 0.5 * (x[1] + 1) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([2 * (x[0] - 2), x[1] + 1])

    def hess(_: np.ndarray) -> np.ndarray:
        return np.array([[2.0, 0.0], [0.0, 0.5]])

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    res = newton_method(
        problem,
        np.array([0.0, 0.0]),
        use_line_search=True,
        line_search=backtracking_armijo,
    )
    assert res.success
    assert res.nit > 1


def test_newton_history_tracking():
    def fun(x: np.ndarray) -> float:
        return float(np.sum((x - 0.5) ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2 * (x - 0.5)

    def hess(_: np.ndarray) -> np.ndarray:
        return 2 * np.eye(2)

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    res = newton_method(problem, np.array([2.0, -1.0]), history=True, maxiter=5)
    assert res.success
    assert len(res.history) >= 2


def test_newton_wolfe_with_fd_gradient():
    def fun(x: np.ndarray) -> float:
        return float((x[0] - 1) ** 2 + (x[1] + 2) ** 2)

    problem = Problem(fun=fun, dim=2)
    res = newton_method(
        problem,
        np.array([3.0, -4.0]),
        use_line_search=True,
        line_search=wolfe_line_search,
        maxiter=10,
    )
    assert res.success


def test_newton_safe_solve_fallback(monkeypatch: pytest.MonkeyPatch):
    original_solve = newton_module.np.linalg.solve

    call_counter = {"count": 0}

    def flaky_solve(*args, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] <= 5:
            raise np.linalg.LinAlgError
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(newton_module.np.linalg, "solve", flaky_solve)

    def fun(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2 * (x - 1.0)

    def hess(_: np.ndarray) -> np.ndarray:
        return 2 * np.eye(2)

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    res = newton_method(problem, np.array([3.0, -2.0]), maxiter=20)
    assert res.success


def test_newton_zero_iterations_checks_gradient():
    def fun(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2 * x

    def hess(_: np.ndarray) -> np.ndarray:
        return 2 * np.eye(2)

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    res = newton_method(problem, np.zeros(2), maxiter=0)
    assert res.success


def test_get_line_search_requires_grad_helper():
    def needs_grad(f, grad, x, p):
        return 1.0, 0

    def no_grad(f, x, p):
        return 1.0, 0

    def invalid(f):
        return 1.0, 0

    assert newton_module._get_line_search_requires_grad(needs_grad)
    assert not newton_module._get_line_search_requires_grad(no_grad)
    assert not newton_module._get_line_search_requires_grad(invalid)

