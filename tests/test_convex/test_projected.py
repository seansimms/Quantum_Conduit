import numpy as np

from qconduit.convex.projected import projected_gradient, projected_newton


def test_projected_gradient_box_quadratic():
    target = np.array([0.8, -0.2, 1.2])
    def obj(x: np.ndarray) -> float:
        return 0.5 * np.sum((x - target) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        return x - target

    def proj(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)
    res = projected_gradient(obj, grad, proj, x0=np.zeros(3), lr=0.5, maxiter=200)
    expected = proj(target)
    assert np.allclose(res.x, expected, atol=1e-4)


def test_projected_newton_affine_projection():
    Q = np.eye(2)
    b = np.array([-1.0, -2.0])
    def obj(x: np.ndarray) -> float:
        return 0.5 * x @ (Q @ x) + b @ x

    def grad(x: np.ndarray) -> np.ndarray:
        return Q @ x + b

    def hess(_: np.ndarray) -> np.ndarray:
        return Q

    def proj(x: np.ndarray) -> np.ndarray:
        shift = (np.sum(x) - 1.0) / 2.0
        return x - shift

    res = projected_newton(obj, grad, hess, proj, x0=np.array([0.0, 0.0]))
    assert abs(np.sum(res.x) - 1.0) <= 1e-8

