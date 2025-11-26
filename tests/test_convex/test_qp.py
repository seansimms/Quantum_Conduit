import numpy as np
import pytest

import qconduit.convex.qp as qp_module
from qconduit.convex.core import Status
from qconduit.convex.qp import active_set_qp


def test_active_set_qp_equality_constraint():
    H = np.eye(2)
    g = np.zeros(2)
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    result = active_set_qp(H, g, a_mat=A, b_vec=b)
    assert result.status is Status.OPTIMAL
    assert np.allclose(result.x, np.array([0.5, 0.5]), atol=1e-6)
    assert pytest.approx(0.25, rel=1e-6) == result.fun


def test_active_set_qp_box_projection():
    target = np.array([1.5, -0.5, 0.2])
    H = np.eye(3)
    g = -target
    lb = np.zeros(3)
    ub = np.array([1.0, 1.0, 0.5])
    result = active_set_qp(H, g, lb=lb, ub=ub)
    assert result.status is Status.OPTIMAL
    expected = np.clip(target, lb, ub)
    assert np.allclose(result.x, expected, atol=1e-4)


def test_active_set_qp_linear_inequality():
    H = np.array([[1.0]])
    g = np.array([-2.0])
    G = np.array([[1.0]])
    h = np.array([1.0])
    res = active_set_qp(H, g, g_mat=G, h_vec=h)
    assert res.status is Status.OPTIMAL
    assert pytest.approx(1.0, rel=1e-6) == res.x[0]


def test_qp_initial_feasible_point_adjusts_equalities():
    system = qp_module._ConstraintSystem(
        a_eq=np.array([[1.0, 1.0], [1.0, 1.0]]),
        b_eq=np.array([1.0, 2.0]),
        g_mat=np.array([[1.0, 0.0]]),
        h_vec=np.array([0.5]),
        lb=None,
        ub=None,
    )
    x = qp_module._initial_feasible_point(system, 2, tol=1e-12)
    assert x.shape == (2,)


def test_qp_assemble_constraints_mixed_bounds():
    system = qp_module._assemble_constraints(
        n=2,
        a_mat=None,
        b_vec=None,
        g_mat=np.zeros((0, 2)),
        h_vec=np.zeros(0),
        lb=np.array([-np.inf, 0.0]),
        ub=np.array([1.0, np.inf]),
    )
    assert system.g_mat.shape[0] == 2

