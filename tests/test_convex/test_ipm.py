import numpy as np
import pytest

from qconduit.convex.core import Status
from qconduit.convex.ipm import log_barrier_method
from qconduit.convex.lp import simplex


def test_log_barrier_matches_simplex():
    c = np.array([-3.0, -5.0])
    G = np.array([[1.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])
    simplex_res = simplex(c, None, None, g_mat=G, h_vec=h)
    ipm_res = log_barrier_method(c, G, h, mu0=5.0)
    assert simplex_res.status is Status.OPTIMAL
    assert ipm_res.status in {Status.OPTIMAL, Status.MAX_ITER}
    assert np.allclose(ipm_res.x, simplex_res.x, atol=1e-3)
    assert pytest.approx(simplex_res.fun, rel=1e-3) == ipm_res.fun


def test_log_barrier_custom_start():
    c = np.array([1.0, 2.0])
    G = np.array([[-1.0, 0.0], [0.0, -1.0]])
    h = np.array([0.0, 0.0])
    x0 = np.array([0.5, 0.5])
    res = log_barrier_method(c, G, h, x0=x0)
    assert res.status in {Status.OPTIMAL, Status.MAX_ITER}
    assert np.all(res.x >= -1e-8)


def test_log_barrier_with_equalities():
    c = np.array([1.0])
    G = np.array([[-1.0], [1.0]])
    h = np.array([0.0, 1.0])
    A = np.array([[1.0]])
    b = np.array([0.5])
    res = log_barrier_method(c, G, h, a_mat=A, b_vec=b)
    assert res.status in {Status.OPTIMAL, Status.MAX_ITER}
    assert pytest.approx(0.5, rel=1e-3) == res.x[0]


def test_log_barrier_rejects_bad_dimensions():
    c = np.array([1.0, 2.0])
    G = np.array([[1.0, 0.0]])
    h = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        log_barrier_method(c, G, h)


def test_log_barrier_bad_initial_point_dimension():
    c = np.array([1.0])
    G = np.array([[1.0]])
    h = np.array([1.0])
    with pytest.raises(ValueError):
        log_barrier_method(c, G, h, x0=np.array([0.0, 0.0]))


def test_log_barrier_projects_infeasible_start():
    c = np.array([1.0, 1.0])
    G = np.array([[1.0, 0.0], [0.0, 1.0]])
    h = np.array([1.0, 1.0])
    res = log_barrier_method(c, G, h, x0=np.array([5.0, 5.0]), mu0=5.0)
    assert res.status in {Status.OPTIMAL, Status.MAX_ITER}
    assert np.all(res.x <= 1.0 + 1e-6)

