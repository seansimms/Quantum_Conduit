"""Circuit builders for Trotterized time-evolution."""

from __future__ import annotations

import math
from typing import List, Sequence

from qconduit.circuit import QuantumCircuit
from qconduit.operators import PauliSum, PauliTerm
from qconduit.time_evolution.core import OrderLiteral


def _append_pauli_term_evolution(
    circuit: QuantumCircuit,
    term: PauliTerm,
    dt: float,
) -> None:
    """
    Append gates to `circuit` implementing exp(-i * term.coeff * dt * P)
    using a standard decomposition into basis changes, CNOT ladder, and
    a single RZ rotation.

    Parameters
    ----------
    circuit:
        QuantumCircuit to which gates will be appended.
    term:
        PauliTerm with real coefficient.
    dt:
        Time step duration for this term.
    """
    coeff = term.coeff
    coeff_real = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
    if isinstance(coeff, complex) and abs(coeff.imag) > 1e-12:
        raise ValueError(
            "PauliTerm coefficient must be real for time evolution; "
            f"got {term.coeff}."
        )
    angle = 2.0 * coeff_real * dt

    paulis: Sequence[str] = term.paulis
    n_qubits = circuit.n_qubits
    if len(paulis) != n_qubits:
        raise ValueError(
            f"PauliTerm length {len(paulis)} does not match circuit.n_qubits={n_qubits}."
        )

    non_identity_qubits: List[int] = []
    for q, p in enumerate(paulis):
        label = p.upper()
        if label not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli label {label!r} in term.")
        if label != "I":
            non_identity_qubits.append(q)

    if not non_identity_qubits:
        # Global phase, no gates needed.
        return

    # Basis changes: record how to undo them.
    basis_change: List[tuple[int, str]] = []
    for q in non_identity_qubits:
        label = paulis[q].upper()
        if label == "X":
            circuit.add_gate("H", [q])
            basis_change.append((q, "X"))
        elif label == "Y":
            # Implement S† via RZ(-pi/2) up to global phase, then H.
            circuit.add_gate("RZ", [q], params=[-math.pi / 2.0])
            circuit.add_gate("H", [q])
            basis_change.append((q, "Y"))
        elif label == "Z":
            basis_change.append((q, "Z"))
        else:
            basis_change.append((q, "I"))

    # Parity CNOT ladder
    pivot = non_identity_qubits[0]
    ladder_qubits: List[int] = []
    for q in non_identity_qubits[1:]:
        # CNOT with q as control and pivot as target
        circuit.add_gate("CNOT", [q, pivot])
        ladder_qubits.append(q)

    # Single RZ rotation on pivot
    if abs(angle) > 0.0:
        circuit.add_gate("RZ", [pivot], params=[angle])

    # Undo ladder
    for q in reversed(ladder_qubits):
        circuit.add_gate("CNOT", [q, pivot])

    # Undo basis changes (reverse order)
    for q, label in reversed(basis_change):
        if label == "X":
            circuit.add_gate("H", [q])
        elif label == "Y":
            # Undo H then S† ≈ RZ(-pi/2)
            circuit.add_gate("H", [q])
            circuit.add_gate("RZ", [q], params=[math.pi / 2.0])
        # Z and I: no action


def build_trotter_step_circuit(
    hamiltonian: PauliSum,
    dt: float,
    n_qubits: int,
    order: OrderLiteral = 1,
) -> QuantumCircuit:
    """
    Build a QuantumCircuit implementing a single Trotter step for
    evolution under the given PauliSum Hamiltonian.

    Parameters
    ----------
    hamiltonian:
        PauliSum Hamiltonian H = sum_k H_k.
    dt:
        Time step duration.
    n_qubits:
        Number of qubits.
    order:
        1 for first-order, 2 for second-order symmetric Trotter.

    Returns
    -------
    QuantumCircuit
        Circuit implementing the approximate exp(-i H dt).
    """
    if order not in (1, 2):
        raise ValueError(f"Unsupported Trotter order {order}. Only 1 and 2 are supported.")

    circuit = QuantumCircuit(n_qubits=n_qubits)
    terms: Sequence[PauliTerm] = hamiltonian.terms

    if order == 1:
        for term in terms:
            _append_pauli_term_evolution(circuit, term, dt=dt)
    else:
        half_dt = dt / 2.0
        # Forward half step
        for term in terms:
            _append_pauli_term_evolution(circuit, term, dt=half_dt)
        # Reverse half step
        for term in reversed(terms):
            _append_pauli_term_evolution(circuit, term, dt=half_dt)

    return circuit


def build_trotter_circuit(
    hamiltonian: PauliSum,
    t: float,
    n_steps: int,
    n_qubits: int,
    order: OrderLiteral = 1,
) -> QuantumCircuit:
    """
    Build a QuantumCircuit implementing Trotterized time evolution
    under H = PauliSum for total time t using n_steps Trotter steps.

    Parameters
    ----------
    hamiltonian:
        PauliSum Hamiltonian.
    t:
        Total evolution time.
    n_steps:
        Number of Trotter steps (must be >= 1).
    n_qubits:
        Number of qubits.
    order:
        1 or 2 for first- or second-order Trotter.

    Returns
    -------
    QuantumCircuit
        Circuit corresponding to [U_dt]^n_steps, where U_dt is the
        single-step Trotter circuit for dt = t / n_steps.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    dt = t / float(n_steps)

    circuit = QuantumCircuit(n_qubits=n_qubits)
    # Build one step, then append its operations n_steps times.
    step_circuit = build_trotter_step_circuit(
        hamiltonian=hamiltonian,
        dt=dt,
        n_qubits=n_qubits,
        order=order,
    )
    step_ops = list(step_circuit.ops)
    for _ in range(n_steps):
        for op in step_ops:
            circuit.add_gate(
                op.name,
                list(op.qubits),
                params=list(op.params) if op.params is not None else None,
            )

    return circuit


