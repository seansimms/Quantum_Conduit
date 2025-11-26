import numpy as np
import pytest

from qconduit.quantum import gates, simulation


def test_hadamard_on_zero_state():
    state = simulation.initial_state(1)
    result = simulation.apply_gate(state, gates.H(), [0])
    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(result, expected)


def test_cnot_on_10_state():
    state = simulation.basis_state(2, 2)
    result = simulation.apply_gate(state, gates.CNOT(), [0, 1])
    assert np.allclose(result, simulation.basis_state(3, 2))


def test_kron_n_matches_numpy():
    kron_gate = simulation.kron_n(gates.H(), gates.I())
    np_gate = np.kron(gates.H(), gates.I())
    assert np.allclose(kron_gate, np_gate)


def test_measure_deterministic():
    state = np.array([0, 0, 0, 1], dtype=complex)
    measured = simulation.measure(state)
    assert measured == 3


def test_measure_probabilistic_branch():
    state = np.array([1, 0], dtype=complex)
    measured = simulation.measure(state, probabilistic=True, seed=0)
    assert measured == 0


def test_initial_and_basis_state_validation():
    with pytest.raises(ValueError):
        simulation.initial_state(-1)
    with pytest.raises(ValueError):
        simulation.basis_state(4, 2)


def test_kron_n_requires_matrices():
    with pytest.raises(ValueError):
        simulation.kron_n()


def test_apply_gate_validation_branches():
    state = simulation.initial_state(1)
    # Empty targets returns a copy
    result = simulation.apply_gate(state, gates.H(), [])
    assert np.allclose(result, state)
    with pytest.raises(ValueError):
        simulation.apply_gate(state, gates.H(), [2])
    with pytest.raises(ValueError):
        simulation.apply_gate(state, gates.H(), [0, 0])
    with pytest.raises(ValueError):
        simulation.apply_gate(state, gates.CNOT(), [0])


def test_apply_gate_state_validation():
    bad_state = np.zeros((2, 2), dtype=complex)
    with pytest.raises(ValueError):
        simulation.apply_gate(bad_state, gates.H(), [0])
    non_power_state = np.array([1, 0, 0], dtype=complex)
    with pytest.raises(ValueError):
        simulation.apply_gate(non_power_state, gates.H(), [0])


def test_measure_normalizes_state():
    unnormalized = np.array([2, 0], dtype=complex)
    assert simulation.measure(unnormalized) == 0


