import numpy as np

from qconduit.quantum import circuits, gates, simulation


def test_circuit_creates_bell_state():
    circuit = circuits.Circuit(2)
    circuit.add_gate(gates.H(), [0])
    circuit.add_gate(gates.CNOT(), [0, 1])
    result = circuit.run()
    expected = (simulation.basis_state(0, 2) + simulation.basis_state(3, 2)) / np.sqrt(2)
    assert np.allclose(result, expected)


def test_circuit_reset_and_initial_state():
    circuit = circuits.Circuit(2)
    circuit.add_gate(gates.X(), [1])
    circuit.reset()
    result = circuit.run()
    assert np.allclose(result, circuit.initial_state())


