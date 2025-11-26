import numpy as np

from qconduit.optimize import Problem, trust_region


def test_trust_region_quadratic_exact_step():
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    b = np.array([1.0, -2.0])

    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ (A @ x) - b @ x

    def grad(x: np.ndarray) -> np.ndarray:
        return A @ x - b

    def hess(_: np.ndarray) -> np.ndarray:
        return A

    problem = Problem(fun=fun, grad=grad, hess=hess, dim=2)
    res = trust_region(problem, np.array([0.0, 0.0]), delta0=10.0, maxiter=5)
    expected = np.linalg.solve(A, b)
    assert res.success
    assert np.allclose(res.x, expected, atol=1e-8)


def test_trust_region_rosenbrock_progress():
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
    x0 = np.array([-1.2, 1.0])
    initial = rosen(x0)
    res = trust_region(problem, x0, maxiter=80)
    assert res.fun < initial
    assert res.success


def test_trust_region_finite_difference_gradients():
    def fun(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    problem = Problem(fun=fun, dim=2)
    res = trust_region(problem, np.array([3.0, -2.0]), maxiter=50, history=True)
    assert res.success
    assert len(res.history) >= 2


def test_trust_region_rejects_unhelpful_step():
    def concave_fun(x: np.ndarray) -> float:
        return float(-0.5 * x @ x)

    def concave_grad(x: np.ndarray) -> np.ndarray:
        return -x

    def concave_hess(_: np.ndarray) -> np.ndarray:
        return -np.eye(2)

    problem = Problem(fun=concave_fun, grad=concave_grad, hess=concave_hess, dim=2)
    res = trust_region(problem, np.array([0.5, 0.0]), maxiter=5, eta=0.9)
    assert not res.success

