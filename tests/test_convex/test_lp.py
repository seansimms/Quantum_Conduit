import numpy as np
import pytest

from qconduit.convex.core import Status
from qconduit.convex.lp import linprog_wrapper, simplex


def test_simplex_canonical_example():
    c = np.array([-3.0, -5.0])
    G = np.array([[1.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])
    result = simplex(c, a_mat=None, b_vec=None, g_mat=G, h_vec=h)
    assert result.status is Status.OPTIMAL
    assert np.allclose(result.x, np.array([1.0, 1.5]), atol=1e-6)
    assert pytest.approx(-10.5, rel=1e-8) == result.fun


def test_simplex_infeasible():
    c = np.array([1.0])
    G = np.array([[-1.0], [1.0]])
    h = np.array([-1.0, 0.0])
    result = simplex(c, None, None, g_mat=G, h_vec=h)
    assert result.status is Status.INFEASIBLE


def test_linprog_wrapper_matches_simplex():
    scipy = pytest.importorskip("scipy")
    del scipy  # unused
    rng = np.random.default_rng(42)
    c = rng.standard_normal(3)
    G = rng.standard_normal((4, 3))
    h = np.ones(4)
    lb = np.zeros(3)
    simplex_res = simplex(c, None, None, g_mat=G, h_vec=h, lb=lb)
    scipy_res = linprog_wrapper(c, None, None, g_mat=G, h_vec=h, lb=lb)
    assert simplex_res.status is Status.OPTIMAL
    assert scipy_res.status is Status.OPTIMAL
    assert np.allclose(simplex_res.x, scipy_res.x, atol=1e-6)
    assert pytest.approx(simplex_res.fun, rel=1e-6) == scipy_res.fun


def test_simplex_handles_equalities():
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    res = simplex(c, a_mat=A, b_vec=b, lb=np.zeros(2))
    assert res.status is Status.OPTIMAL
    assert pytest.approx(1.0, rel=1e-8) == res.fun
    assert pytest.approx(1.0, rel=1e-8) == np.sum(res.x)


def test_simplex_trivial_solution_no_constraints():
    c = np.array([2.0, 3.0])
    res = simplex(c, None, None)
    assert res.status is Status.OPTIMAL
    assert np.allclose(res.x, np.zeros_like(c))
    assert res.fun == 0.0


def test_simplex_unbounded_problem():
    c = np.array([-1.0])
    res = simplex(c, None, None)
    assert res.status is Status.UNBOUNDED


def test_simplex_invalid_dimension_error():
    c = np.array([1.0, 2.0])
    A = np.ones((1, 2))
    res = simplex(c, a_mat=A, b_vec=None)
    assert res.status is Status.NUMERICAL_ERROR


def test_simplex_upper_bounds():
    c = np.array([-1.0])
    ub = np.array([1.0])
    res = simplex(c, None, None, ub=ub)
    assert res.status is Status.OPTIMAL
    assert pytest.approx(-1.0, rel=1e-8) == res.fun
    assert pytest.approx(1.0, rel=1e-8) == res.x[0]


def test_linprog_wrapper_invalid_bounds_raises():
    pytest.importorskip("scipy")
    c = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        linprog_wrapper(c, None, None, lb=np.zeros(3), ub=np.ones(2))


def test_simplex_scalar_lower_bound_broadcasts():
    c = np.array([1.0, 2.0])
    res = simplex(c, None, None, lb=0.0)
    assert res.status is Status.OPTIMAL
    assert np.allclose(res.x, np.zeros_like(c))


def test_simplex_free_variable_split_path():
    c = np.array([0.0])
    G = np.array([[1.0], [-1.0]])
    h = np.array([1.0, 1.0])
    res = simplex(c, None, None, g_mat=G, h_vec=h, lb=np.array([-np.inf]))
    assert res.status is Status.OPTIMAL
    assert abs(res.x[0]) <= 1.0


def test_simplex_invalid_lower_bound_length():
    c = np.array([1.0, 2.0])
    res = simplex(c, None, None, lb=np.zeros(3))
    assert res.status is Status.NUMERICAL_ERROR

