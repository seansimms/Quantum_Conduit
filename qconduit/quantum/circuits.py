"""
Minimal circuit builder backed by the deterministic state-vector simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import simulation
from .gates import Array


@dataclass(frozen=True)
class GateApplication:
    """Store a gate and the qubits it targets."""

    gate: Array
    targets: Tuple[int, ...]


class Circuit:
    """Sequential circuit composed of textbook gates."""

    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("Circuit must operate on at least one qubit.")
        self.n_qubits = n_qubits
        self._gates: List[GateApplication] = []

    def add_gate(self, gate: Array, targets: Sequence[int]) -> None:
        """Append ``gate`` acting on ``targets`` to the circuit."""

        self._gates.append(GateApplication(np.asarray(gate, dtype=complex), tuple(targets)))

    def reset(self) -> None:
        """Remove all gate applications."""

        self._gates.clear()

    def initial_state(self) -> Array:
        """Return the ``|0...0>`` basis state for the circuit size."""

        return simulation.initial_state(self.n_qubits)

    def run(self, state: Array | None = None) -> Array:
        """Run the circuit by applying all stored gates sequentially."""

        if state is None:
            current = self.initial_state()
        else:
            current = np.asarray(state, dtype=complex)
        for app in self._gates:
            current = simulation.apply_gate(current, app.gate, list(app.targets))
        return current


__all__ = ["Circuit", "GateApplication"]


