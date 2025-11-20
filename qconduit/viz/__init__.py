"""Visualization and introspection utilities for quantum circuits.

This module provides:
- Circuit text/ASCII drawing
- Bloch sphere coordinate computation and plotting
- Circuit summary and comparison utilities
"""

from .drawer import to_text, print_circuit
from .bloch import (
    bloch_coords_from_statevector,
    bloch_coords_from_density,
    plot_bloch_vector,
    plot_bloch_projections,
)
from .summary import (
    circuit_summary,
    print_circuit_summary,
    compare_circuits,
)

__all__ = [
    "to_text",
    "print_circuit",
    "bloch_coords_from_statevector",
    "bloch_coords_from_density",
    "plot_bloch_vector",
    "plot_bloch_projections",
    "circuit_summary",
    "print_circuit_summary",
    "compare_circuits",
]

