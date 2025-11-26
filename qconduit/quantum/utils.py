"""Canonical gate application utilities with deterministic LSB-first ordering.

This module provides reference implementations for applying single- and two-qubit
gates to statevectors using the LSB-first (qubit 0 = least significant bit)
convention throughout. These routines are fully deterministic and do not rely
on heuristics or gate-specific detection.
"""

from __future__ import annotations

from typing import List

import numpy as np

Array = np.ndarray


def kron_n(*matrices: Array) -> Array:
    """Compute Kronecker product of multiple matrices left-to-right."""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def apply_single_qubit_gate(
    state: Array,
    gate: Array,
    qubit: int,
    n_qubits: int | None = None,
) -> Array:
    """Apply a single-qubit gate to `qubit` in an n-qubit statevector.

    Convention: LSB-first, so qubit 0 is the least significant bit in the
    computational basis index.

    Parameters
    ----------
    state : Array
        Statevector of length 2**n_qubits.
    gate : Array
        2x2 unitary gate matrix.
    qubit : int
        Target qubit index (0-indexed, 0 = LSB).
    n_qubits : int, optional
        Number of qubits. Inferred from state length if not provided.

    Returns
    -------
    Array
        Transformed statevector.
    """
    if n_qubits is None:
        n_qubits = int(np.log2(len(state)))
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit {qubit} out of range [0, {n_qubits})")

    # Build full operator via Kronecker product (reversed for LSB-first)
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = np.asarray(gate, dtype=complex)
    full_gate = kron_n(*ops[::-1])
    return full_gate @ state


def apply_two_qubit_gate(
    state: Array,
    gate: Array,
    qubits: List[int],
    n_qubits: int | None = None,
) -> Array:
    """Apply a two-qubit gate to specified qubits in LSB-first convention.

    Parameters
    ----------
    state : Array
        Statevector of length 2**n_qubits.
    gate : Array
        4x4 unitary gate matrix. The gate acts on the two-qubit subspace
        indexed as |q0 q1⟩ where q0 is the first element of `qubits` and
        q1 is the second.
    qubits : List[int]
        Two-element list [control, target] or [q0, q1] specifying which
        qubits the gate acts on.
    n_qubits : int, optional
        Number of qubits. Inferred from state length if not provided.

    Returns
    -------
    Array
        Transformed statevector.
    """
    if len(qubits) != 2:
        raise ValueError("Exactly two qubits must be specified")
    q0, q1 = qubits
    if n_qubits is None:
        n_qubits = int(np.log2(len(state)))
    if q0 < 0 or q0 >= n_qubits:
        raise ValueError(f"qubit {q0} out of range [0, {n_qubits})")
    if q1 < 0 or q1 >= n_qubits:
        raise ValueError(f"qubit {q1} out of range [0, {n_qubits})")
    if q0 == q1:
        raise ValueError("qubits must be distinct")

    gate = np.asarray(gate, dtype=complex)
    dim = 2**n_qubits
    full_gate = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        bits_i = [(i >> k) & 1 for k in range(n_qubits)]
        for j in range(dim):
            bits_j = [(j >> k) & 1 for k in range(n_qubits)]
            # Non-target qubits must match
            if any(bits_i[k] != bits_j[k] for k in range(n_qubits) if k not in qubits):
                continue
            # Map to 2-qubit subsystem: index = 2*b_q0 + b_q1
            idx_i = (bits_i[q0] << 1) + bits_i[q1]
            idx_j = (bits_j[q0] << 1) + bits_j[q1]
            full_gate[i, j] = gate[idx_i, idx_j]

    return full_gate @ state


__all__ = [
    "Array",
    "apply_single_qubit_gate",
    "apply_two_qubit_gate",
    "kron_n",
]

