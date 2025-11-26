import numpy as np

from qconduit.convex.kkt import is_kkt_optimal, kkt_residuals


def test_kkt_residuals_at_optimum():
    H = np.eye(2)
    g = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    x = np.array([0.5, 0.5])
    lam = np.array([0.5])
    residuals = kkt_residuals(H, g, A, b, None, None, x, lam=lam)
    assert residuals["primal_eq"] <= 1e-9
    assert residuals["dual"] <= 1e-9
    assert is_kkt_optimal(H, g, A, b, None, None, x, lam=lam)


def test_kkt_detects_infeasibility():
    H = np.eye(1)
    g = np.array([0.0])
    G = np.array([[1.0]])
    h = np.array([0.0])
    x = np.array([1.0])
    residuals = kkt_residuals(H, g, None, None, G, h, x)
    assert residuals["primal_ineq"] > 0.5
    assert not is_kkt_optimal(H, g, None, None, G, h, x)

