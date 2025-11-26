"""Circuit summary and comparison utilities.

This module provides high-level introspection tools for analyzing and
comparing quantum circuits.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import IO, Dict, Optional

import torch

from qconduit.circuit import QuantumCircuit
from qconduit.transpile.analysis import estimate_circuit_depth, summarize_gate_counts


def circuit_summary(circuit: QuantumCircuit) -> Dict[str, any]:
    """
    Generate a comprehensive summary dictionary for a circuit.

    Parameters
    ----------
    circuit:
        Circuit to analyze.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - n_qubits: int
        - n_gates: int
        - gate_counts: Dict[str, int]
        - t_count: int
        - clifford_count: int
        - estimated_depth: int
        - params_count: int
        - uses_param_gates: bool
    """
    gate_summary = summarize_gate_counts(circuit)
    depth = estimate_circuit_depth(circuit)

    # Count parameters
    params_count = 0
    uses_param_gates = False
    for op in circuit.ops:
        if op.params is not None:
            params_count += len(op.params)
            uses_param_gates = True

    return {
        "n_qubits": circuit.n_qubits,
        "n_gates": gate_summary.total_gates,
        "gate_counts": dict(gate_summary.counts),
        "t_count": gate_summary.t_count,
        "clifford_count": gate_summary.clifford_count,
        "estimated_depth": depth,
        "params_count": params_count,
        "uses_param_gates": uses_param_gates,
    }


def print_circuit_summary(
    circuit: QuantumCircuit,
    file: Optional[IO[str]] = None,
) -> None:
    """
    Pretty-print a circuit summary to stdout or a file.

    This is a utility function for human-readable output, so it uses print()
    intentionally. For programmatic access, use circuit_summary() instead.

    Parameters
    ----------
    circuit:
        Circuit to summarize.
    file:
        File-like object to write to. If None, writes to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    summary = circuit_summary(circuit)

    print("Circuit Summary", file=file)
    print("=" * 50, file=file)
    print(f"Qubits: {summary['n_qubits']}", file=file)
    print(f"Total Gates: {summary['n_gates']}", file=file)
    print(f"Estimated Depth: {summary['estimated_depth']}", file=file)
    print(f"T-count: {summary['t_count']}", file=file)
    print(f"Clifford-count: {summary['clifford_count']}", file=file)
    print(f"Parameters: {summary['params_count']}", file=file)
    print(f"Uses Parametric Gates: {summary['uses_param_gates']}", file=file)
    print("\nGate Counts:", file=file)
    for gate_name, count in sorted(summary["gate_counts"].items()):
        print(f"  {gate_name}: {count}", file=file)


def _circuit_to_unitary_small(
    circuit: QuantumCircuit,
    max_dim: int = 16,
) -> Optional[torch.Tensor]:
    """
    Compute unitary matrix for a small circuit.

    Only computes if dimension <= max_dim to avoid memory issues.

    Parameters
    ----------
    circuit:
        Circuit to convert.
    max_dim:
        Maximum dimension (2**n_qubits) to allow.

    Returns
    -------
    Optional[torch.Tensor]
        Unitary matrix of shape (dim, dim), or None if too large.
    """
    dim = 2**circuit.n_qubits
    if dim > max_dim:
        return None

    # Import here to avoid circular imports
    from qconduit.batched.apply import _circuit_to_unitary

    device = torch.device("cpu")
    dtype = torch.complex128
    unitary = _circuit_to_unitary(circuit, device=device, dtype=dtype)
    return unitary


def _compare_unitaries(
    u1: torch.Tensor,
    u2: torch.Tensor,
    atol: float = 1e-6,
) -> bool:
    """
    Compare two unitaries up to global phase.

    Two unitaries u1 and u2 are equivalent up to global phase if
    u1 = e^(iφ) * u2 for some phase φ.

    Parameters
    ----------
    u1:
        First unitary matrix.
    u2:
        Second unitary matrix.
    atol:
        Absolute tolerance for comparison.

    Returns
    -------
    bool
        True if unitaries are equivalent up to global phase.
    """
    if u1.shape != u2.shape:
        return False

    # Check if u1 and u2 are proportional (up to global phase)
    # Compute ratio: u1 / u2 (element-wise, avoiding division by zero)
    # For unitary matrices, if u1 = e^(iφ) u2, then u1 @ u2^dagger = e^(iφ) I
    u2_dag = u2.conj().T
    product = u1 @ u2_dag

    # Extract diagonal elements (should all be equal to e^(iφ))
    diag = torch.diag(product)
    if len(diag) == 0:
        return True

    # Check if all diagonal elements are approximately equal
    first_phase = diag[0]
    if torch.abs(first_phase) < atol:
        # Check if matrices are zero
        return torch.allclose(u1, u2, atol=atol)

    # Normalize by first element
    normalized = diag / first_phase
    if not torch.allclose(normalized, torch.ones_like(normalized), atol=atol):
        return False

    # Check off-diagonal elements are zero
    mask = ~torch.eye(len(diag), dtype=torch.bool, device=product.device)
    off_diag = product[mask]
    return torch.allclose(off_diag, torch.zeros_like(off_diag), atol=atol)


def compare_circuits(
    circ_a: QuantumCircuit,
    circ_b: QuantumCircuit,
    max_dim_for_unitary: int = 16,
) -> Dict[str, any]:
    """
    Compare two circuits and return differences.

    Parameters
    ----------
    circ_a:
        First circuit.
    circ_b:
        Second circuit.
    max_dim_for_unitary:
        Maximum dimension (2**n_qubits) for computing unitary comparison.
        If either circuit exceeds this, same_unitary will be None.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - same_unitary: Optional[bool] - True if unitaries are equivalent
          (up to global phase), False if different, None if not computed.
        - delta_gate_counts: Dict[str, int] - Per-gate count differences
          (count_b - count_a).
        - depth_diff: int - Difference in depth (depth_b - depth_a).
        - param_diff: int - Difference in parameter count.
    """
    summary_a = circuit_summary(circ_a)
    summary_b = circuit_summary(circ_b)

    # Gate count differences
    counts_a = Counter(summary_a["gate_counts"])
    counts_b = Counter(summary_b["gate_counts"])
    delta_gate_counts = dict(counts_b - counts_a)

    # Depth difference
    depth_diff = summary_b["estimated_depth"] - summary_a["estimated_depth"]

    # Parameter count difference
    param_diff = summary_b["params_count"] - summary_a["params_count"]

    # Unitary comparison (only for small circuits)
    same_unitary: Optional[bool] = None
    if circ_a.n_qubits == circ_b.n_qubits:
        dim = 2**circ_a.n_qubits
        if dim <= max_dim_for_unitary:
            u_a = _circuit_to_unitary_small(circ_a, max_dim=max_dim_for_unitary)
            u_b = _circuit_to_unitary_small(circ_b, max_dim=max_dim_for_unitary)
            if u_a is not None and u_b is not None:
                same_unitary = _compare_unitaries(u_a, u_b)

    return {
        "same_unitary": same_unitary,
        "delta_gate_counts": delta_gate_counts,
        "depth_diff": depth_diff,
        "param_diff": param_diff,
    }



