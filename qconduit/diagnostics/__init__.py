"""Diagnostics and debugging utilities for Quantum Conduit."""

from .core import (
    assert_hermitian,
    assert_normalized,
    bloch_vector,
    fidelity,
    is_hermitian,
    state_norm,
)
from .debug_mode import (
    debug_context,
    is_debug_enabled,
    set_debug_enabled,
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





