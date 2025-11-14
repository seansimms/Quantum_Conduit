"""Exact diagonalization utilities for small quantum systems."""

from .diagonalize import (
    exact_eigensystem,
    exact_ground_state,
    paulisum_to_dense,
)

__all__ = [
    "paulisum_to_dense",
    "exact_eigensystem",
    "exact_ground_state",
]


