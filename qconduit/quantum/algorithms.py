"""
Textbook quantum algorithms simulated with deterministic NumPy routines.

Reference: M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
Information*, Cambridge University Press.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence

import numpy as np

from . import gates, simulation

Array = np.ndarray

OracleBool = Callable[[int], int]
OracleSimon = Callable[[int], int]


def _apply_boolean_oracle(state: Array, oracle: OracleBool, n_inputs: int) -> Array:
    """Apply a Deutsch-Jozsa style oracle to ``state``."""
    new_state = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        x = index >> 1
        ancilla = index & 1
        if oracle(x):
            ancilla ^= 1
        new_index = (x << 1) | ancilla
        new_state[new_index] = amplitude
    return new_state


def _apply_simon_oracle(state: Array, oracle: OracleSimon, n_inputs: int) -> Array:
    """Apply a Simon oracle Uf(x, y) = (x, y xor f(x))."""
    mask = (1 << n_inputs) - 1
    new_state = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        x = index >> n_inputs
        y = index & mask
        new_y = y ^ oracle(x)
        new_index = (x << n_inputs) | new_y
        new_state[new_index] = amplitude
    return new_state


def _first_register_probabilities(
    state: Array, n_first: int, total_qubits: int
) -> Dict[int, float]:
    """Return probabilities for measuring the first ``n_first`` qubits."""

    probs: Dict[int, float] = {}
    remainder_bits = total_qubits - n_first
    for index, amplitude in enumerate(state):
        first_value = index >> remainder_bits
        probs[first_value] = probs.get(first_value, 0.0) + float(np.abs(amplitude) ** 2)
    return probs


def _parity(value: int) -> int:
    """Return the parity (mod 2) of ``value``."""

    return bin(value).count("1") % 2


def _solve_hidden_string(equations: Iterable[int], n_qubits: int) -> str:
    """
    Solve for the Simon hidden string using linear equations ``y·s = 0``.
    """

    eqs = list({eq for eq in equations if eq != 0})
    if not eqs:
        raise ValueError("Insufficient equations to determine Simon string.")
    for candidate in range(1, 1 << n_qubits):
        if all(_parity(candidate & eq) == 0 for eq in eqs):
            return format(candidate, f"0{n_qubits}b")
    raise ValueError("No non-zero solution satisfies all equations.")


def deutsch_jozsa(oracle: OracleBool, n_qubits: int) -> str:
    """
    Classical simulation of the Deutsch-Jozsa algorithm.

    Returns the bitstring observed on the first register: ``"0"*n`` signals a
    constant oracle; any other string signals a balanced oracle.
    """

    total_qubits = n_qubits + 1
    state = simulation.initial_state(total_qubits)
    state = simulation.apply_gate(state, gates.X(), [n_qubits])
    for qubit in range(total_qubits):
        state = simulation.apply_gate(state, gates.H(), [qubit])
    state = _apply_boolean_oracle(state, oracle, n_qubits)
    for qubit in range(n_qubits):
        state = simulation.apply_gate(state, gates.H(), [qubit])
    measurement = simulation.measure(state)
    result = measurement >> 1
    return format(result, f"0{n_qubits}b")


def simon(oracle: OracleSimon, n_qubits: int) -> str:
    """
    Classical simulation of Simon's algorithm.

    The simulator deterministically extracts the hidden string ``s`` such that
    ``f(x) = f(x xor s)`` while ensuring measurement statistics match the
    textbook expectation.
    """

    total_qubits = 2 * n_qubits
    state = simulation.initial_state(total_qubits)
    for qubit in range(n_qubits):
        state = simulation.apply_gate(state, gates.H(), [qubit])
    state = _apply_simon_oracle(state, oracle, n_qubits)
    for qubit in range(n_qubits):
        state = simulation.apply_gate(state, gates.H(), [qubit])
    probabilities = _first_register_probabilities(state, n_qubits, total_qubits)
    equations = [value for value, prob in probabilities.items() if prob > 1e-9]
    return _solve_hidden_string(equations, n_qubits)


def _apply_phase_oracle(state: Array, solutions: Sequence[int]) -> Array:
    """Flip the phase of amplitudes corresponding to solution indices."""

    new_state = state.copy()
    for index in solutions:
        new_state[index] *= -1
    return new_state


def _diffusion(state: Array) -> Array:
    """Apply the Grover diffusion operator."""

    mean = np.mean(state)
    return 2 * mean - state


def grover(oracle: OracleBool, n_qubits: int) -> int:
    """
    Grover's search algorithm for small search spaces.

    Returns the integer index of a marked item supplied by ``oracle``.
    """

    n_states = 1 << n_qubits
    solutions = [index for index in range(n_states) if oracle(index)]
    if not solutions:
        raise ValueError("Grover oracle must mark at least one solution.")
    state = simulation.initial_state(n_qubits)
    for qubit in range(n_qubits):
        state = simulation.apply_gate(state, gates.H(), [qubit])
    iterations = max(1, int(round(np.pi / 4 * np.sqrt(n_states / len(solutions)))))
    for _ in range(iterations):
        state = _apply_phase_oracle(state, solutions)
        state = _diffusion(state)
    probabilities = np.abs(state) ** 2
    best_solution = max(solutions, key=lambda idx: probabilities[idx])
    return int(best_solution)


def qft_matrix(n_qubits: int) -> Array:
    """Return the ``n``-qubit Quantum Fourier Transform matrix."""

    dim = 1 << n_qubits
    omega = np.exp(2j * np.pi / dim)
    indices = np.arange(dim)
    matrix = omega ** np.outer(indices, indices)
    return matrix / np.sqrt(dim)


def iqft_matrix(n_qubits: int) -> Array:
    """Return the inverse Quantum Fourier Transform matrix."""

    return np.conjugate(qft_matrix(n_qubits)).T


def _reverse_qubits(state: Array, n_qubits: int) -> Array:
    """Reverse the order of qubits using SWAP gates."""

    current = state
    for index in range(n_qubits // 2):
        current = simulation.apply_gate(
            current, gates.SWAP(), [index, n_qubits - index - 1]
        )
    return current


def _apply_qft_sequential(state: Array, n_qubits: int, inverse: bool = False) -> Array:
    """Apply the QFT (or inverse) using a gate sequence."""

    current = state.copy()
    if inverse:
        current = _reverse_qubits(current, n_qubits)
        for target in reversed(range(n_qubits)):
            for control in reversed(range(target + 1, n_qubits)):
                angle = -np.pi / (1 << (control - target))
                current = simulation.apply_gate(
                    current, gates.controlled_phase(angle), [control, target]
                )
            current = simulation.apply_gate(current, gates.H(), [target])
        return current

    for target in range(n_qubits):
        current = simulation.apply_gate(current, gates.H(), [target])
        for control in range(target + 1, n_qubits):
            angle = np.pi / (1 << (control - target))
            current = simulation.apply_gate(
                current, gates.controlled_phase(angle), [control, target]
            )
    current = _reverse_qubits(current, n_qubits)
    return current


def qft(state: Array, n_qubits: int | None = None) -> Array:
    """Apply the sequential QFT to ``state``."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("State must be a vector.")
    if n_qubits is None:
        n_qubits = int(np.log2(state.size))
    if state.size != 1 << n_qubits:
        raise ValueError("State size incompatible with n_qubits.")
    return _apply_qft_sequential(state, n_qubits, inverse=False)


def iqft(state: Array, n_qubits: int | None = None) -> Array:
    """Apply the sequential inverse QFT to ``state``."""

    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("State must be a vector.")
    if n_qubits is None:
        n_qubits = int(np.log2(state.size))
    if state.size != 1 << n_qubits:
        raise ValueError("State size incompatible with n_qubits.")
    return _apply_qft_sequential(state, n_qubits, inverse=True)


__all__ = [
    "Array",
    "deutsch_jozsa",
    "grover",
    "iqft",
    "iqft_matrix",
    "qft",
    "qft_matrix",
    "simon",
]


