"""Core circuit IR types and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from qconduit.core.device import Device, default_device
from qconduit.backend.statevector import zero_state, apply_gate, apply_two_qubit_gate
from qconduit.gates import standard as stdgates


@dataclass(frozen=True)
class GateOp:
    """
    A single gate application in a quantum circuit.

    This is purely structural, generic QC plumbing: it records
    which named gate is applied to which qubits with which
    (numeric) parameters, if any.

    Attributes
    ----------
    name:
        Gate name, e.g. "X", "H", "CNOT", "RX", "RY", "RZ".
    qubits:
        Tuple of target qubit indices (0-based).
    params:
        Optional tuple of float parameters, e.g. rotation angles.
        For non-parametric gates this is None.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Optional[Tuple[float, ...]] = None


class QuantumCircuit:
    """
    Simple circuit IR: an ordered list of gate applications on n_qubits.

    This is standard quantum-circuit plumbing used in most frameworks:
    it stores gate names, target qubits, and optional numeric parameters.
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize a QuantumCircuit."""
        if n_qubits <= 0:
            raise ValueError("QuantumCircuit requires n_qubits >= 1.")

        self._n_qubits = int(n_qubits)
        self._ops: List[GateOp] = []

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits in this circuit."""
        return self._n_qubits

    @property
    def ops(self) -> Tuple[GateOp, ...]:
        """Return a read-only tuple of all gate operations."""
        return tuple(self._ops)

    def add_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Append a gate application to the circuit.

        Parameters
        ----------
        name:
            A known gate name such as "X", "H", "CNOT", "RX", "RY", or "RZ".
        qubits:
            Target qubit indices (0-based). For single-qubit gates this
            has length 1, for two-qubit gates like CNOT it has length 2.
        params:
            Optional numeric parameters (e.g. rotation angles).
        """
        q_tuple = tuple(int(q) for q in qubits)
        if not q_tuple:
            raise ValueError("GateOp must act on at least one qubit.")
        for q in q_tuple:
            if q < 0 or q >= self._n_qubits:
                raise ValueError(
                    f"Qubit index {q} is out of range for this circuit "
                    f"(n_qubits={self._n_qubits})."
                )

        p_tuple: Optional[Tuple[float, ...]]
        if params is None:
            p_tuple = None
        else:
            p_tuple = tuple(float(p) for p in params)

        self._ops.append(GateOp(name=name, qubits=q_tuple, params=p_tuple))

    def copy(self) -> "QuantumCircuit":
        """Return a deep copy of this circuit."""
        new = QuantumCircuit(self._n_qubits)
        new._ops.extend(self._ops)
        return new

    def __len__(self) -> int:
        """Return the number of gate operations in this circuit."""
        return len(self._ops)

    def num_gates(self) -> int:
        """Return the number of gate operations in this circuit."""
        return len(self._ops)

    def gate_counts(self) -> Dict[str, int]:
        """Return a dictionary mapping gate names to their counts."""
        counts: Dict[str, int] = {}
        for op in self._ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def depth(self) -> int:
        """
        Estimate the circuit depth as the minimum number of sequential layers
        required if gates that act on disjoint qubits can be run in parallel.

        This is a simple textbook-style scheduling heuristic, not hardware-
        specific compilation.
        """
        if not self._ops:
            return 0

        # Track, for each qubit, the latest layer index it is involved in.
        qubit_layer = [0] * self._n_qubits
        max_layer = 0

        for op in self._ops:
            # Determine the earliest layer where all qubits are free.
            earliest = 0
            for q in op.qubits:
                if qubit_layer[q] > earliest:
                    earliest = qubit_layer[q]

            layer = earliest + 1
            for q in op.qubits:
                qubit_layer[q] = layer

            if layer > max_layer:
                max_layer = layer

        return max_layer

    def simulate_state(
        self,
        device: Optional[Device] = None,
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        """
        Simulate this circuit from the all-zero state using the existing
        statevector backend and gate library.

        Returns
        -------
        state:
            Complex tensor of shape (2**n_qubits,) on the chosen device.
        """
        if device is None:
            dev = default_device()
        else:
            dev = device

        torch_device = dev.as_torch_device()

        state = zero_state(
            n_qubits=self._n_qubits,
            batch_shape=None,
            device=dev,
            dtype=dtype,
        )

        for op in self._ops:
            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate = _resolve_single_qubit_gate(name, op.params, dtype, torch_device)
                state = apply_gate(state, gate, qubit=q, n_qubits=self._n_qubits)
            elif len(op.qubits) == 2:
                q0, q1 = op.qubits
                # For CNOT, the first qubit in the list is the control, second is target
                # Based on backend tests, we need control_first=False when control < target
                gate = _resolve_two_qubit_gate(name, q0, q1, dtype, torch_device)
                state = apply_two_qubit_gate(
                    state, gate, qubit1=q0, qubit2=q1, n_qubits=self._n_qubits
                )
            else:
                raise ValueError(
                    f"QuantumCircuit currently supports only 1- and 2-qubit gates; "
                    f"got gate {op.name!r} on qubits {op.qubits}."
                )

        return state

    def to_text_diagram(self) -> str:
        """
        Return a simple ASCII diagram of the circuit.

        This is a generic, textbook-style visualization: each qubit is a
        horizontal line, and each gate is drawn in sequence. Single-qubit
        gates are shown with their name; CNOT uses '●' for control and '⊕'
        for target.
        """
        if not self._ops:
            # Just show empty wires
            lines = [f"q{q}: " for q in range(self._n_qubits)]
            return "\n".join(lines)

        # We use a fixed-width "slot" per operation.
        # For each op, we create a column for all qubits.
        wire_segments: List[List[str]] = [
            [] for _ in range(self._n_qubits)
        ]

        for op in self._ops:
            # default empty segment: wire only
            for q in range(self._n_qubits):
                wire_segments[q].append("───")

            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                label = name
                if len(label) == 1:
                    gate_str = f"─{label}─"
                else:
                    # Take first character to keep width simple
                    gate_str = f"─{label[0]}─"
                wire_segments[q][-1] = gate_str
            elif len(op.qubits) == 2 and name == "CNOT":
                control, target = op.qubits
                # Control
                wire_segments[control][-1] = "─●─"
                # Target
                wire_segments[target][-1] = "─⊕─"
            else:
                # Generic marker for other 2-qubit gates, if any arise
                for q in op.qubits:
                    wire_segments[q][-1] = "─#─"

        lines: List[str] = []
        for q in range(self._n_qubits):
            prefix = f"q{q}: "
            line = prefix + "".join(wire_segments[q])
            lines.append(line)

        return "\n".join(lines)


def _resolve_single_qubit_gate(
    name: str,
    params: Optional[Tuple[float, ...]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Map a gate name and optional parameters to a 2x2 unitary using
    the existing standard gate functions.

    Supported non-parametric names: I, X, Y, Z, H, S, T
    Supported parametric names: RX, RY, RZ
    """
    n = name.upper()
    if n == "I":
        return stdgates.I(dtype=dtype, device=device)
    if n == "X":
        return stdgates.X(dtype=dtype, device=device)
    if n == "Y":
        return stdgates.Y(dtype=dtype, device=device)
    if n == "Z":
        return stdgates.Z(dtype=dtype, device=device)
    if n == "H":
        return stdgates.H(dtype=dtype, device=device)
    if n == "S":
        return stdgates.S(dtype=dtype, device=device)
    if n == "T":
        return stdgates.T(dtype=dtype, device=device)

    if n in ("RX", "RY", "RZ"):
        if not params or len(params) != 1:
            raise ValueError(f"Gate {n} requires exactly one parameter.")
        theta = float(params[0])
        if n == "RX":
            return stdgates.RX(theta, dtype=dtype, device=device)
        if n == "RY":
            return stdgates.RY(theta, dtype=dtype, device=device)
        if n == "RZ":
            return stdgates.RZ(theta, dtype=dtype, device=device)

    raise ValueError(
        f"Unsupported single-qubit gate name {name!r}. "
        "Supported gates: I, X, Y, Z, H, S, T, RX, RY, RZ."
    )


def _resolve_two_qubit_gate(
    name: str,
    control: int,
    target: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Map a gate name to a 4x4 unitary for 2-qubit gates.

    Currently supports "CNOT". The control and target parameters are used
    to determine the correct gate configuration.

    Parameters
    ----------
    name:
        Gate name, e.g. "CNOT".
    control:
        Control qubit index (first qubit in the gate application).
    target:
        Target qubit index (second qubit in the gate application).
    dtype:
        Complex dtype for the gate matrix.
    device:
        PyTorch device for the gate matrix.

    Returns
    -------
    Gate matrix of shape (4, 4).
    """
    n = name.upper()
    if n == "CNOT":
        # The circuit IR convention: first qubit in the list is control, second is target.
        # To match HardwareEfficientAnsatz behavior, we use control_first=True
        # when control < target. This ensures consistency with existing ansatz code.
        control_first = control < target
        return stdgates.CNOT(dtype=dtype, device=device, control_first=control_first)
    raise ValueError(
        f"Unsupported two-qubit gate name {name!r}. "
        "Currently only 'CNOT' is supported."
    )

