"""
Deterministic, NumPy-based state-vector simulation utilities.

States are represented as complex vectors of length ``2**n`` ordered such that
qubit ``0`` is the most-significant bit (MSB) in the computational basis
ordering.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Array = np.ndarray


def _validate_state(state: Array) -> Array:
    """Ensure ``state`` is a normalized complex vector."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("State vectors must be one-dimensional.")
    size = state.size
    if size == 0 or (size & (size - 1)) != 0:
        raise ValueError("State vector length must be a positive power of two.")
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        state = state / norm
    return state


def kron_n(*matrices: Array) -> Array:
    """Return the Kronecker product of ``matrices`` evaluated left-to-right."""

    if not matrices:
        raise ValueError("Provide at least one matrix to kron_n.")
    result = np.array([[1]], dtype=complex)
    for matrix in matrices:
        result = np.kron(result, np.asarray(matrix, dtype=complex))
    return result


def initial_state(n_qubits: int) -> Array:
    """Return the ``|0...0>`` computational basis state."""

    if n_qubits < 0:
        raise ValueError("Number of qubits must be non-negative.")
    state = np.zeros(1 << n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def basis_state(index: int, n_qubits: int) -> Array:
    """Return the computational basis state identified by ``index``."""

    dim = 1 << n_qubits
    if not (0 <= index < dim):
        raise ValueError("Index outside computational basis range.")
    state = np.zeros(dim, dtype=complex)
    state[index] = 1.0
    return state


def apply_gate(state: Array, gate: Array, qubits: Sequence[int]) -> Array:
    """
    Apply ``gate`` to ``state`` on the specified ``qubits``.

    ``qubits`` are indexed such that ``0`` corresponds to the MSB. Gates are
    supplied as dense matrices acting on ``len(qubits)`` qubits.
    """

    state = _validate_state(state)
    gate = np.asarray(gate, dtype=complex)

    n_qubits = int(np.log2(state.size))
    target_qubits = list(qubits)
    if not target_qubits:
        return state.copy()
    if any(q < 0 or q >= n_qubits for q in target_qubits):
        raise ValueError("Target qubits out of range.")
    if len(set(target_qubits)) != len(target_qubits):
        raise ValueError("Target qubits must be unique.")

    k = len(target_qubits)
    if gate.shape != (1 << k, 1 << k):
        raise ValueError("Gate shape incompatible with number of targets.")

    tensor_state = state.reshape([2] * n_qubits)
    perm = target_qubits + [q for q in range(n_qubits) if q not in target_qubits]
    inverse_perm = np.argsort(perm)
    reshaped = np.transpose(tensor_state, perm).reshape(1 << k, -1)
    updated = gate @ reshaped
    restored = np.transpose(
        updated.reshape([2] * n_qubits), inverse_perm  # type: ignore[arg-type]
    ).reshape(-1)
    return restored


def measure(state: Array, probabilistic: bool = False, seed: int | None = None) -> int:
    """
    Measure ``state`` in the computational basis.

    By default the procedure is deterministic: the most probable basis state is
    returned. Set ``probabilistic=True`` to draw a random sample instead.
    """

    state = _validate_state(state)
    probabilities = np.abs(state) ** 2
    if probabilistic:
        rng = np.random.default_rng(seed)
        return int(rng.choice(len(state), p=probabilities))
    return int(np.argmax(probabilities))


__all__ = [
    "Array",
    "apply_gate",
    "basis_state",
    "initial_state",
    "kron_n",
    "measure",
]


