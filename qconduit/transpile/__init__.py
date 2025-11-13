"""Gate decompositions and basis transpiler for quantum circuits."""

from .decompose import (
    decompose_h_to_rz_rx_rz,
    decompose_x_to_rx,
    decompose_y_to_ry,
    decompose_y_to_rz_rx_rz,
    decompose_z_to_rz,
    decompose_rz_to_clifford_t,
    decompose_gate_to_basis,
)

from .basis import (
    transpile_to_basis,
    transpile_to_rx_rz_cx_basis,
    transpile_to_clifford_t,
)

from .analysis import (
    GateCountSummary,
    summarize_gate_counts,
    estimate_circuit_depth,
)

__all__ = [
    "decompose_h_to_rz_rx_rz",
    "decompose_x_to_rx",
    "decompose_y_to_ry",
    "decompose_y_to_rz_rx_rz",
    "decompose_z_to_rz",
    "decompose_rz_to_clifford_t",
    "decompose_gate_to_basis",
    "transpile_to_basis",
    "transpile_to_rx_rz_cx_basis",
    "transpile_to_clifford_t",
    "GateCountSummary",
    "summarize_gate_counts",
    "estimate_circuit_depth",
]

