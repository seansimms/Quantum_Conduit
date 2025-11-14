"""Basis transpiler functions for converting circuits to target gate sets."""

from __future__ import annotations

from typing import Sequence

from qconduit.circuit import QuantumCircuit
from qconduit.transpile.decompose import decompose_gate_to_basis


def transpile_to_basis(
    circuit: QuantumCircuit,
    basis_gates: Sequence[str],
) -> QuantumCircuit:
    """
    Transpile a circuit into a target gate basis.

    Each gate that is already in `basis_gates` is copied unchanged. Gates that
    are not in the basis are replaced, when possible, by an equivalent sequence
    of gates generated via `decompose_gate_to_basis`.

    Parameters
    ----------
    circuit:
        Input QuantumCircuit.
    basis_gates:
        Iterable of gate names allowed in the target basis, e.g.
        ["RX", "RZ", "CNOT"].

    Returns
    -------
    QuantumCircuit
        New circuit with gates restricted to `basis_gates`.

    Raises
    ------
    ValueError
        If any gate cannot be represented using the specified basis.
    """
    allowed = {b.upper() for b in basis_gates}
    out = QuantumCircuit(circuit.n_qubits)

    for i, gate in enumerate(circuit.ops):
        gate_name_upper = gate.name.upper()

        # If gate is already in target basis, copy it
        if gate_name_upper in allowed:
            out.add_gate(gate.name, gate.qubits, gate.params)
        else:
            # Decompose the gate
            decomp = decompose_gate_to_basis(circuit, i, basis_gates)

            # Append all gates from decomposition to output circuit
            for decomp_gate in decomp.ops:
                out.add_gate(decomp_gate.name, decomp_gate.qubits, decomp_gate.params)

    return out


def transpile_to_rx_rz_cx_basis(
    circuit: QuantumCircuit,
) -> QuantumCircuit:
    """
    Convenience transpiler that maps a circuit to the {RX, RZ, CNOT} basis.

    This is a common target for hardware with native X/Z rotations and CNOT
    entangling gates.

    Parameters
    ----------
    circuit:
        Input QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        New circuit with gates restricted to RX, RZ, and CNOT.
    """
    basis = ["RX", "RZ", "CNOT"]
    return transpile_to_basis(circuit, basis)


def transpile_to_clifford_t(
    circuit: QuantumCircuit,
    allow_rz_fallback: bool = True,
) -> QuantumCircuit:
    """
    Transpile a circuit to a Clifford+T-style basis.

    The target basis includes:

        - H, S, T, CNOT (and optionally RZ if allow_rz_fallback=True).

    For RZ rotations whose angles are integer multiples of π/4, this function
    replaces them by sequences of S and T gates. For other RZ angles, if
    allow_rz_fallback is True, the RZ gates are kept as-is; otherwise a
    ValueError is raised.

    Parameters
    ----------
    circuit:
        Input QuantumCircuit.
    allow_rz_fallback:
        If True, include "RZ" in the target basis to allow arbitrary Z-rotations
        that are not exactly decomposable into S/T. If False, only multiples of
        π/4 are permitted.

    Returns
    -------
    QuantumCircuit
        New circuit with gates restricted to the chosen Clifford+T-style basis.

    Raises
    ------
    ValueError
        If allow_rz_fallback is False and an RZ gate cannot be expressed
        exactly using S/T.
    """
    basis = ["H", "S", "T", "CNOT"]
    if allow_rz_fallback:
        basis.append("RZ")

    return transpile_to_basis(circuit, basis)


__all__ = [
    "transpile_to_basis",
    "transpile_to_rx_rz_cx_basis",
    "transpile_to_clifford_t",
]


