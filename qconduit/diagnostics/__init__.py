"""Diagnostics and debugging utilities for Quantum Conduit."""

from .core import (
    state_norm,
    assert_normalized,
    is_hermitian,
    assert_hermitian,
    fidelity,
    bloch_vector,
)
from .debug_mode import (
    is_debug_enabled,
    set_debug_enabled,
    debug_context,
)

__all__ = [
    "state_norm",
    "assert_normalized",
    "is_hermitian",
    "assert_hermitian",
    "fidelity",
    "bloch_vector",
    "is_debug_enabled",
    "set_debug_enabled",
    "debug_context",
]


