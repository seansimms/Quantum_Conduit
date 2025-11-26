"""Standard quantum gates as NumPy arrays.

All gates follow the textbook definitions and are compatible with the
LSB-first qubit ordering convention used in qconduit.quantum.utils.
"""

from __future__ import annotations

import numpy as np

from .utils import Array


def I() -> Array:  # noqa: E743, N802
    """Identity gate."""
    return np.array([[1, 0], [0, 1]], dtype=complex)


def X() -> Array:  # noqa: N802
    """Pauli-X (NOT) gate."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def Y() -> Array:  # noqa: N802
    """Pauli-Y gate."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def Z() -> Array:  # noqa: N802
    """Pauli-Z gate."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def H() -> Array:  # noqa: N802
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def S() -> Array:  # noqa: N802
    """S (phase) gate."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def T() -> Array:  # noqa: N802
    """T (pi/8) gate."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def CNOT() -> Array:  # noqa: N802
    """CNOT gate with control as first qubit, target as second.

    Matrix ordering: |00⟩, |01⟩, |10⟩, |11⟩ where the first qubit is
    the control and the second is the target.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def CZ() -> Array:  # noqa: N802
    """Controlled-Z gate."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
        dtype=complex,
    )


def SWAP() -> Array:  # noqa: N802
    """SWAP gate."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


def controlled_phase(phi: float) -> Array:
    """Controlled phase gate with angle phi."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * phi)],
        ],
        dtype=complex,
    )


__all__ = [
    "CNOT",
    "CZ",
    "H",
    "I",
    "S",
    "SWAP",
    "T",
    "X",
    "Y",
    "Z",
    "controlled_phase",
]
