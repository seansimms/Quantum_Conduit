import numpy as np

from qconduit.optimize import Problem, gradient_descent


def test_gradient_descent_quadratic_converges():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ (A @ x) - b @ x

    def grad(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    problem = Problem(fun=fun, grad=grad, dim=2)
    x0 = np.array([3.0, -1.0])
    res = gradient_descent(problem, x0, lr=0.1, maxiter=500)
    expected = np.linalg.solve(A, b)
    assert res.success
    assert np.allclose(res.x, expected, atol=1e-5)
    assert res.grad_norm < 1e-6


def test_momentum_and_nesterov_reduce_rosenbrock():
    def rosen(x: np.ndarray) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def rosen_grad(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        )

    problem = Problem(fun=rosen, grad=rosen_grad, dim=2)
    x0 = np.array([-1.2, 1.0])
    base_value = rosen(x0)

    res_momentum = gradient_descent(
        problem, x0, lr=1e-3, maxiter=2000, momentum=True, beta=0.9
    )
    res_nesterov = gradient_descent(
        problem, x0, lr=1e-3, maxiter=2000, momentum=True, nesterov=True
    )
    assert res_momentum.fun < base_value
    assert res_nesterov.fun < base_value
    assert res_nesterov.fun <= res_momentum.fun


def test_gradient_descent_deterministic():
    rng_start = np.array([0.5, -0.25])

    def f(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2 * x

    problem = Problem(fun=f, grad=grad, dim=2)
    res1 = gradient_descent(problem, rng_start, lr=0.2, maxiter=50, history=True)
    res2 = gradient_descent(problem, rng_start, lr=0.2, maxiter=50, history=True)
    assert np.allclose(res1.x, res2.x)
    assert np.allclose(res1.history[-1], res2.history[-1])

