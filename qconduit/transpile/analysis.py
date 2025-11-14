"""Circuit analysis utilities for gate counts and depth estimation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from qconduit.circuit import QuantumCircuit


@dataclass(frozen=True)
class GateCountSummary:
    """
    Summary of gate counts by type, with optional T-count and Clifford count.

    Attributes
    ----------
    counts:
        Mapping from gate name to its occurrence count in the circuit.
    t_count:
        Number of T gates (0 if not present).
    clifford_count:
        Number of Clifford gates (H, S, X, Y, Z, CNOT, CZ, etc.) observed.
    total_gates:
        Total number of gates in the circuit.
    """

    counts: Mapping[str, int]
    t_count: int
    clifford_count: int
    total_gates: int


def summarize_gate_counts(
    circuit: QuantumCircuit,
) -> GateCountSummary:
    """
    Count gates by type and report T-count and Clifford-count.

    Clifford gates are assumed to include:

        H, S, X, Y, Z, CNOT, CZ

    (case-insensitive, depending on how gate names are stored).

    Parameters
    ----------
    circuit:
        Circuit to analyze.

    Returns
    -------
    GateCountSummary
        Summary of gate counts, T-count, and Clifford-count.
    """
    counts = Counter()
    for op in circuit.ops:
        counts[op.name] += 1

    total_gates = sum(counts.values())
    t_count = counts.get("T", 0)

    # Clifford gate names (case-insensitive check)
    clifford_names = {"H", "S", "X", "Y", "Z", "CNOT", "CZ"}
    clifford_count = sum(
        count for name, count in counts.items() if name.upper() in clifford_names
    )

    return GateCountSummary(
        counts=dict(counts),
        t_count=t_count,
        clifford_count=clifford_count,
        total_gates=total_gates,
    )


def estimate_circuit_depth(
    circuit: QuantumCircuit,
) -> int:
    """
    Estimate the depth of a circuit under a simple parallelization model.

    Gates that act on disjoint sets of qubits are assumed to be executable in
    parallel in the same layer. Multi-qubit gates occupy all their qubits in
    a layer.

    Parameters
    ----------
    circuit:
        Circuit to analyze.

    Returns
    -------
    int
        Estimated circuit depth (number of layers). The empty circuit has
        depth 0.
    """
    if len(circuit.ops) == 0:
        return 0

    # Track layers: each layer is a set of qubit indices used in that layer
    layers: list[set[int]] = []

    for gate in circuit.ops:
        qubits = set(gate.qubits)

        # Find the earliest layer where all qubits are free
        placed = False
        for layer in layers:
            if layer.isdisjoint(qubits):
                # Can place gate in this layer
                layer.update(qubits)
                placed = True
                break

        # If no suitable layer found, create a new one
        if not placed:
            layers.append(qubits.copy())

    return len(layers)


__all__ = [
    "GateCountSummary",
    "summarize_gate_counts",
    "estimate_circuit_depth",
]


