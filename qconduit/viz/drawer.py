"""ASCII/text circuit drawer for QuantumCircuit visualization.

This module provides deterministic, dependency-free circuit visualization
using ASCII and box-drawing characters.
"""

from __future__ import annotations

import math
import sys
from typing import IO, List, Optional, Tuple

from qconduit.circuit import GateOp, QuantumCircuit
from qconduit.transpile.analysis import estimate_circuit_depth


def _format_angle(theta: float, atol: float = 1e-8) -> str:
    """
    Format an angle in radians as a readable string.

    Attempts to represent exact rational multiples of π in a compact form
    (e.g., π/2, π/4, 3π/4). Falls back to decimal representation otherwise.

    Parameters
    ----------
    theta:
        Angle in radians.
    atol:
        Absolute tolerance for checking rational multiples.

    Returns
    -------
    str
        Formatted angle string.
    """
    pi = math.pi

    # Check for multiples of π/4
    k = round(theta / (pi / 4.0))
    if math.isclose(theta, k * (pi / 4.0), abs_tol=atol):
        if k == 0:
            return "0"
        if k == 1:
            return "π/4"
        if k == 2:
            return "π/2"
        if k == 3:
            return "3π/4"
        if k == 4:
            return "π"
        if k == -1:
            return "-π/4"
        if k == -2:
            return "-π/2"
        if k == -3:
            return "-3π/4"
        if k == -4:
            return "-π"
        return f"{k}π/4"

    # Check for multiples of π/2
    k = round(theta / (pi / 2.0))
    if math.isclose(theta, k * (pi / 2.0), abs_tol=atol):
        if k == 1:
            return "π/2"
        if k == -1:
            return "-π/2"
        return f"{k}π/2"

    # Check for multiples of π
    k = round(theta / pi)
    if math.isclose(theta, k * pi, abs_tol=atol):
        if k == 1:
            return "π"
        if k == -1:
            return "-π"
        return f"{k}π"

    # Fall back to decimal with 3 decimal places
    return f"{theta:.3f}"


def _compute_layers(circuit: QuantumCircuit) -> List[List[int]]:
    """
    Compute gate layers using a simple greedy scheduling algorithm.

    Returns a list where each element is a list of gate indices that
    can be executed in parallel (act on disjoint qubits).

    Parameters
    ----------
    circuit:
        Circuit to analyze.

    Returns
    -------
    List[List[int]]
        List of layers, where each layer contains gate indices.
    """
    if len(circuit.ops) == 0:
        return []

    layers: List[List[int]] = []
    # Track the latest layer each qubit is used in
    qubit_layers: List[int] = [0] * circuit.n_qubits

    for gate_idx, gate in enumerate(circuit.ops):
        qubits = set(gate.qubits)
        # Find the earliest layer where all qubits are free
        earliest_layer = max(qubit_layers[q] for q in qubits)

        # Try to place in existing layer at earliest_layer
        placed = False
        if earliest_layer < len(layers):
            # Check if we can add to this layer (no conflicts)
            conflicts = False
            for other_idx in layers[earliest_layer]:
                other_qubits = set(circuit.ops[other_idx].qubits)
                if not qubits.isdisjoint(other_qubits):
                    conflicts = True
                    break
            if not conflicts:
                layers[earliest_layer].append(gate_idx)
                placed = True

        if not placed:
            # Create new layer
            target_layer = earliest_layer + 1
            while len(layers) < target_layer:
                layers.append([])
            layers.append([gate_idx])

        # Update qubit layers
        for q in qubits:
            qubit_layers[q] = len(layers) - 1

    return layers


def _render_gate_label(
    gate: GateOp, use_ascii: bool = False
) -> str:
    """
    Render a gate label for display.

    Parameters
    ----------
    gate:
        GateOp instance.
    use_ascii:
        If True, use only ASCII characters.

    Returns
    -------
    str
        Gate label string.
    """
    name = gate.name.upper()

    # Handle parametric gates
    if gate.params is not None and len(gate.params) > 0:
        if name in ("RX", "RY", "RZ"):
            angle_str = _format_angle(gate.params[0])
            return f"{name}({angle_str})"
        # Generic parametric gate
        params_str = ",".join(_format_angle(p) if abs(p) < 10 else f"{p:.2f}" for p in gate.params)
        return f"{name}({params_str})"

    # Standard gate names
    if len(name) <= 3:
        return name
    return name[:3]


def _render_gate_column(
    circuit: QuantumCircuit,
    gate_indices: List[int],
    n_qubits: int,
    use_ascii: bool = False,
) -> List[str]:
    """
    Render a column of gates (one layer) as wire segments.

    Parameters
    ----------
    circuit:
        Circuit being drawn.
    gate_indices:
        List of gate indices in this layer.
    n_qubits:
        Number of qubits.
    use_ascii:
        If True, use only ASCII characters.

    Returns
    -------
    List[str]
        List of strings, one per qubit wire.
    """
    # Initialize wire segments
    segments: List[str] = [""] * n_qubits

    # Process each gate in the layer
    for gate_idx in gate_indices:
        gate = circuit.ops[gate_idx]
        name = gate.name.upper()
        qubits = gate.qubits

        if len(qubits) == 1:
            # Single-qubit gate
            q = qubits[0]
            label = _render_gate_label(gate, use_ascii)
            segments[q] = f"[{label}]"

        elif len(qubits) == 2:
            # Two-qubit gate
            q0, q1 = min(qubits), max(qubits)
            label = _render_gate_label(gate, use_ascii)

            if name == "CNOT":
                # CNOT: control and target
                if use_ascii:
                    segments[q0] = "[*]"  # Control
                    segments[q1] = "[X]"  # Target
                else:
                    segments[q0] = "●"  # Control
                    segments[q1] = "⊕"  # Target
            else:
                # Generic two-qubit gate
                if use_ascii:
                    segments[q0] = "[*]"
                    segments[q1] = f"[{label}]"
                else:
                    segments[q0] = "●"
                    segments[q1] = f"[{label}]"

        else:
            # Multi-qubit gate
            label = _render_gate_label(gate, use_ascii)
            min_q = min(qubits)
            max_q = max(qubits)
            for q in qubits:
                if q == min_q:
                    segments[q] = f"[{label}"
                elif q == max_q:
                    segments[q] = f"{label}]"
                else:
                    segments[q] = "─"

    # Fill empty wires with wire continuation
    wire_char = "-" if use_ascii else "─"
    for q in range(n_qubits):
        if not segments[q]:
            segments[q] = wire_char * 3

    return segments


def to_text(
    circuit: QuantumCircuit,
    max_width: int = 80,
    use_ascii: bool = False,
) -> str:
    """
    Convert a QuantumCircuit to a multi-line ASCII/text diagram.

    Parameters
    ----------
    circuit:
        Circuit to visualize.
    max_width:
        Maximum width in characters. If circuit is wider, pagination is used.
    use_ascii:
        If True, use only ASCII characters (no box-drawing characters).

    Returns
    -------
    str
        Multi-line string representation of the circuit.
    """
    if len(circuit.ops) == 0:
        # Empty circuit
        lines = [f"q{q}: " for q in range(circuit.n_qubits)]
        return "\n".join(lines)

    # Compute layers
    layers = _compute_layers(circuit)

    if not layers:
        lines = [f"q{q}: " for q in range(circuit.n_qubits)]
        return "\n".join(lines)

    # Estimate column width (approximate)
    col_width = 8  # Approximate width per column
    cols_per_page = max(1, (max_width - 10) // col_width)  # Reserve space for "q0: " prefix

    all_lines: List[str] = []
    current_page_lines: List[str] = None

    for page_start in range(0, len(layers), cols_per_page):
        page_end = min(page_start + cols_per_page, len(layers))
        page_layers = layers[page_start:page_end]

        # Initialize wire lines for this page
        wire_lines: List[List[str]] = [[] for _ in range(circuit.n_qubits)]

        # Render each layer
        for layer in page_layers:
            column_segments = _render_gate_column(circuit, layer, circuit.n_qubits, use_ascii)
            for q in range(circuit.n_qubits):
                wire_lines[q].append(column_segments[q])

        # Build output lines
        page_lines = []
        for q in range(circuit.n_qubits):
            prefix = f"q{q}: "
            wire_str = "".join(wire_lines[q])
            page_lines.append(prefix + wire_str)

        all_lines.extend(page_lines)

        # Add continuation marker if there are more pages
        if page_end < len(layers):
            all_lines.append("-- continuing --")

    return "\n".join(all_lines)


def print_circuit(
    circuit: QuantumCircuit,
    file: Optional[IO[str]] = None,
    max_width: int = 80,
    use_ascii: bool = False,
) -> None:
    """
    Print a circuit diagram to stdout or a file.

    Parameters
    ----------
    circuit:
        Circuit to visualize.
    file:
        File-like object to write to. If None, writes to sys.stdout.
    max_width:
        Maximum width in characters.
    use_ascii:
        If True, use only ASCII characters.
    """
    if file is None:
        file = sys.stdout
    text = to_text(circuit, max_width=max_width, use_ascii=use_ascii)
    print(text, file=file)

