"""
Integration tests for convex optimization module.

Tests that the convex module integrates correctly with the main package
and can be used in realistic scenarios.
"""

import numpy as np

from qconduit import (
    OptimizeResult,
    Status,
    active_set_qp,
    is_kkt_optimal,
    kkt_residuals,
    log_barrier_method,
    projected_gradient,
    projected_newton,
    simplex,
)


def test_main_package_imports():
    """Test that all convex optimization APIs are accessible from main package."""
    assert simplex is not None
    assert active_set_qp is not None
    assert log_barrier_method is not None
    assert projected_gradient is not None
    assert projected_newton is not None
    assert kkt_residuals is not None
    assert is_kkt_optimal is not None
    assert OptimizeResult is not None
    assert Status is not None


def test_simplex_integration():
    """Test simplex solver through main package import."""
    c = np.array([-3.0, -5.0])
    G = np.array([[1.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])
    result = simplex(c, a_mat=None, b_vec=None, g_mat=G, h_vec=h)
    assert result.status == Status.OPTIMAL
    assert result.x is not None
    assert result.fun is not None
    assert np.allclose(result.x, [1.0, 1.5], atol=1e-6)


def test_qp_integration():
    """Test QP solver through main package import."""
    H = np.eye(2)
    g = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    result = active_set_qp(H, g, a_mat=A, b_vec=b)
    assert result.status == Status.OPTIMAL
    assert result.x is not None
    assert np.allclose(result.x, [0.5, 0.5], atol=1e-6)


def test_ipm_integration():
    """Test interior-point method through main package import."""
    c = np.array([-3.0, -5.0])
    G = np.array([[1.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])
    result = log_barrier_method(c, G, h, mu0=5.0, maxiter=50)
    assert result.status in {Status.OPTIMAL, Status.MAX_ITER}
    assert result.x is not None
    assert result.fun is not None


def test_projected_methods_integration():
    """Test projected methods through main package import."""
    target = np.array([0.8, -0.2, 1.2])

    def obj(x: np.ndarray) -> float:
        return 0.5 * np.sum((x - target) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        return x - target

    def proj(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    result = projected_gradient(obj, grad, proj, x0=np.zeros(3), lr=0.5, maxiter=200)
    assert result.status == Status.OPTIMAL
    assert result.x is not None
    expected = proj(target)
    assert np.allclose(result.x, expected, atol=1e-3)


def test_kkt_diagnostics_integration():
    """Test KKT diagnostics through main package import."""
    H = np.eye(2)
    g = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    x = np.array([0.5, 0.5])
    # Correct multiplier: Hx + g + A^T lambda = 0
    # [0.5, 0.5] + [-1, -1] + [1, 1] * lambda = 0
    # [-0.5, -0.5] + [lambda, lambda] = 0
    # lambda = 0.5
    lam = np.array([0.5])
    residuals = kkt_residuals(H, g, A, b, None, None, x, lam=lam)
    assert residuals["primal_eq"] <= 1e-9
    assert residuals["dual"] <= 1e-9
    assert is_kkt_optimal(H, g, A, b, None, None, x, lam=lam, tol=1e-8)


def test_workflow_integration():
    """Test a realistic workflow combining multiple convex optimization tools."""
    # Step 1: Solve an LP
    c = np.array([-1.0, -2.0])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([3.0, 0.0, 0.0])
    lp_result = simplex(c, None, None, G, h)
    assert lp_result.status == Status.OPTIMAL

    # Step 2: Use LP solution as starting point for QP
    if lp_result.x is not None:
        H = np.eye(2)
        g = np.array([0.0, 0.0])
        qp_result = active_set_qp(H, g, lb=np.zeros(2), ub=lp_result.x + 0.1)
        assert qp_result.status == Status.OPTIMAL

        # Step 3: Validate QP solution with KKT
        if qp_result.x is not None:
            residuals = kkt_residuals(H, g, None, None, None, None, qp_result.x)
            assert residuals["dual"] <= 1e-6


def test_result_type_consistency():
    """Test that all solvers return consistent OptimizeResult types."""
    c = np.array([-1.0, -1.0])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([2.0, 0.0, 0.0])

    lp_result = simplex(c, None, None, G, h)
    ipm_result = log_barrier_method(c, G, h, maxiter=30)

    assert isinstance(lp_result, OptimizeResult)
    assert isinstance(ipm_result, OptimizeResult)
    assert hasattr(lp_result, "x")
    assert hasattr(lp_result, "fun")
    assert hasattr(lp_result, "status")
    assert hasattr(lp_result, "message")
    assert hasattr(lp_result, "nit")


def test_error_handling_integration():
    """Test that error handling works correctly through main package."""
    # Infeasible problem
    c = np.array([1.0, 1.0])
    G = np.array([[1.0, 1.0], [-1.0, -1.0]])
    h = np.array([1.0, -2.0])  # Infeasible: x1 + x2 <= 1 and x1 + x2 >= 2
    result = simplex(c, None, None, G, h)
    assert result.status == Status.INFEASIBLE
    assert result.x is None

