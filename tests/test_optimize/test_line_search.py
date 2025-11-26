import numpy as np
import pytest

from qconduit.optimize.line_search import backtracking_armijo, wolfe_line_search


def quadratic_fun(x: np.ndarray) -> float:
    return float(x.T @ x)


def quadratic_grad(x: np.ndarray) -> np.ndarray:
    return 2 * x


def test_backtracking_armijo_monotone():
    x = np.array([1.0, -2.0])
    grad = quadratic_grad(x)
    direction = -grad
    alpha, nevals = backtracking_armijo(quadratic_fun, x, direction, grad)
    assert 0 < alpha <= 1.0
    new_val = quadratic_fun(x + alpha * direction)
    assert new_val <= quadratic_fun(x)
    assert nevals > 0


def rosen(x: np.ndarray) -> float:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ]
    )


def test_wolfe_conditions_rosenbrock():
    x = np.array([-1.2, 1.0])
    grad = rosen_grad(x)
    direction = -grad
    alpha, _ = wolfe_line_search(rosen, rosen_grad, x, direction)
    phi0 = rosen(x)
    phi_alpha = rosen(x + alpha * direction)
    directional_derivative = rosen_grad(x + alpha * direction) @ direction
    assert phi_alpha <= phi0 + 1e-4 * alpha * (grad @ direction)
    assert abs(directional_derivative) <= 0.9 * abs(grad @ direction)


def test_backtracking_armijo_raises_on_invalid_params():
    x = np.array([1.0])
    grad = quadratic_grad(x)
    with pytest.raises(ValueError):
        backtracking_armijo(quadratic_fun, x, -grad, grad, c=1.5)
    with pytest.raises(ValueError):
        backtracking_armijo(quadratic_fun, x, -grad, grad, rho=1.1)


def test_wolfe_zoom_phase_triggered():
    x = np.array([-1.2, 1.0])
    grad = rosen_grad(x)
    direction = -grad
    alpha, _ = wolfe_line_search(
        rosen,
        rosen_grad,
        x,
        direction,
        alpha0=5.0,
    )
    assert alpha < 1.0

