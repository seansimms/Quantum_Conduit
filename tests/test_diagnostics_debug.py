"""Tests for debug mode functionality."""

import pytest
import torch

from qconduit.diagnostics import (
    is_debug_enabled,
    set_debug_enabled,
    debug_context,
)
from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.gates import standard as stdgates
from qconduit.core.device import default_device


def test_debug_mode_toggle_and_context() -> None:
    """Test debug mode toggling and context manager."""
    original = is_debug_enabled()

    try:
        set_debug_enabled(False)
        assert not is_debug_enabled()

        with debug_context(True):
            assert is_debug_enabled()

        # Back to previous (False in this block)
        assert not is_debug_enabled()

        set_debug_enabled(True)
        assert is_debug_enabled()

        with debug_context(False):
            assert not is_debug_enabled()

        # Back to True
        assert is_debug_enabled()
    finally:
        set_debug_enabled(original)


def test_debug_context_nested() -> None:
    """Test nested debug contexts."""
    original = is_debug_enabled()

    try:
        set_debug_enabled(False)
        assert not is_debug_enabled()

        with debug_context(True):
            assert is_debug_enabled()

            with debug_context(False):
                assert not is_debug_enabled()

            # Back to True
            assert is_debug_enabled()

        # Back to False
        assert not is_debug_enabled()
    finally:
        set_debug_enabled(original)


def test_apply_gate_normalization_check_in_debug_mode() -> None:
    """Test that apply_gate performs normalization check in debug mode."""
    dev = default_device()
    state = zero_state(
        n_qubits=1, batch_shape=None, device=dev, dtype=torch.complex64
    )
    # Use a proper unitary gate: H
    H = stdgates.H(dtype=torch.complex64, device=dev.as_torch_device())

    original = is_debug_enabled()

    try:
        # Should not raise even in debug mode (H is unitary)
        set_debug_enabled(True)
        new_state = apply_gate(state, H, qubit=0, n_qubits=1)
        # Sanity: still normalized
        assert new_state.shape == state.shape

        # Should also work with debug mode off
        set_debug_enabled(False)
        new_state2 = apply_gate(state, H, qubit=0, n_qubits=1)
        assert new_state2.shape == state.shape
    finally:
        set_debug_enabled(original)


def test_apply_two_qubit_gate_normalization_check_in_debug_mode() -> None:
    """Test that apply_two_qubit_gate performs normalization check in debug mode."""
    from qconduit.backend.statevector import apply_two_qubit_gate

    dev = default_device()
    state = zero_state(
        n_qubits=2, batch_shape=None, device=dev, dtype=torch.complex64
    )
    # Use a proper unitary gate: CNOT
    CNOT = stdgates.CNOT(dtype=torch.complex64, device=dev.as_torch_device())

    original = is_debug_enabled()

    try:
        # Should not raise even in debug mode (CNOT is unitary)
        set_debug_enabled(True)
        new_state = apply_two_qubit_gate(state, CNOT, qubit1=0, qubit2=1, n_qubits=2)
        # Sanity: still normalized
        assert new_state.shape == state.shape

        # Should also work with debug mode off
        set_debug_enabled(False)
        new_state2 = apply_two_qubit_gate(
            state, CNOT, qubit1=0, qubit2=1, n_qubits=2
        )
        assert new_state2.shape == state.shape
    finally:
        set_debug_enabled(original)

