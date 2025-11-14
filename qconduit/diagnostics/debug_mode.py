"""Debug mode management for Quantum Conduit."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_DEBUG_ENV_VAR = "QCONDUIT_DEBUG"
_debug_enabled: bool = os.getenv(_DEBUG_ENV_VAR, "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def is_debug_enabled() -> bool:
    """
    Return whether Quantum Conduit debug mode is currently enabled.

    Debug mode can be toggled via set_debug_enabled(...) or the
    QCONDUIT_DEBUG environment variable.

    Returns
    -------
    bool
        True if debug mode is enabled, False otherwise.
    """
    return _debug_enabled


def set_debug_enabled(enabled: bool) -> None:
    """
    Globally enable or disable Quantum Conduit debug mode.

    Parameters
    ----------
    enabled:
        Whether to enable debug mode.
    """
    global _debug_enabled
    _debug_enabled = bool(enabled)


@contextmanager
def debug_context(enabled: bool = True) -> Iterator[None]:
    """
    Context manager to temporarily enable or disable debug mode.

    Parameters
    ----------
    enabled:
        Whether to enable debug mode within the context.

    Example
    -------
    >>> with debug_context(True):
    ...     # debug mode enabled inside block
    ...     pass
    """
    global _debug_enabled
    prev = _debug_enabled
    _debug_enabled = bool(enabled)
    try:
        yield
    finally:
        _debug_enabled = prev


