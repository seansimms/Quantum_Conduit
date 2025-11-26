"""Visualization and introspection utilities for quantum circuits.

This module provides:
- Circuit text/ASCII drawing
- Bloch sphere coordinate computation and plotting
- Circuit summary and comparison utilities
"""

from .bloch import (
    bloch_coords_from_density,
    bloch_coords_from_statevector,
    plot_bloch_projections,
    plot_bloch_vector,
)
from .drawer import print_circuit, to_text
from .summary import (
    circuit_summary,
    compare_circuits,
    print_circuit_summary,
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



