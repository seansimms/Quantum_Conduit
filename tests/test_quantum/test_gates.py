import numpy as np
import pytest

from qconduit.quantum import gates


def _is_unitary(matrix: np.ndarray) -> bool:
    ident = np.eye(matrix.shape[0], dtype=complex)
    return np.allclose(matrix @ matrix.conj().T, ident)


@pytest.mark.parametrize(
    "gate_fn",
    [
        gates.I,
        gates.X,
        gates.Y,
        gates.Z,
        gates.H,
        gates.S,
        gates.T,
    ],
)
def test_single_qubit_unitarity(gate_fn):
    assert _is_unitary(gate_fn())


@pytest.mark.parametrize(
    "gate_fn",
    [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        lambda: gates.controlled_phase(np.pi / 3),
    ],
)
def test_two_qubit_unitarity(gate_fn):
    assert _is_unitary(gate_fn())


def test_cnot_basis_mapping():
    gate = gates.CNOT()
    basis_states = np.eye(4, dtype=complex)
    mapped = gate @ basis_states[:, 2]
    assert np.allclose(mapped, basis_states[:, 3])


