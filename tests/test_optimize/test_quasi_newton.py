import numpy as np

from qconduit.optimize import Problem, bfgs, lbfgs


def rosenbrock(x: np.ndarray) -> float:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
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


def test_bfgs_reaches_rosenbrock_minimum():
    problem = Problem(fun=rosenbrock, grad=rosenbrock_grad, dim=2)
    res = bfgs(problem, np.array([-1.2, 1.0]), maxiter=200)
    assert res.success
    assert np.allclose(res.x, np.ones(2), atol=1e-5)
    assert res.fun < 1e-9


def test_lbfgs_handles_higher_dimension():
    def rosen_nd(x: np.ndarray) -> float:
        return sum(
            (1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
            for i in range(0, len(x) - 1, 2)
        )

    def rosen_grad_nd(x: np.ndarray) -> np.ndarray:
        g = np.zeros_like(x)
        for i in range(0, len(x) - 1, 2):
            g[i] = -2 * (1 - x[i]) - 400 * x[i] * (x[i + 1] - x[i] ** 2)
            g[i + 1] = 200 * (x[i + 1] - x[i] ** 2)
        return g

    problem = Problem(fun=rosen_nd, grad=rosen_grad_nd, dim=4)
    x0 = np.array([-1.2, 1.0, -1.0, 1.0])
    res = lbfgs(problem, x0, m=5, maxiter=400)
    assert res.success
    assert np.allclose(res.x, np.ones(4), atol=1e-5)
    assert res.fun < 1e-8


def test_bfgs_vs_lbfgs_on_himmelblau():
    problem = Problem(fun=himmelblau, grad=himmelblau_grad, dim=2)
    x0 = np.array([3.0, 1.5])
    res_bfgs = bfgs(problem, x0, maxiter=200)
    res_lbfgs = lbfgs(problem, x0, m=6, maxiter=200)
    assert res_bfgs.success and res_lbfgs.success
    assert res_bfgs.fun < 1e-10
    assert res_lbfgs.fun < 1e-10
    assert np.allclose(res_bfgs.x, res_lbfgs.x, atol=1e-6)


def test_bfgs_without_gradient():
    problem = Problem(fun=rosenbrock, dim=2)
    res = bfgs(problem, np.array([-1.2, 1.0]), maxiter=100)
    assert res.success
    assert res.fun < 1e-6

