"""Tests for Kraus channel implementations."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.noise import (
    KrausChannel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    phase_damping_channel,
    amplitude_damping_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
)
from qconduit.core.device import default_device


def test_bit_flip_channel_construction():
    """Test bit flip channel construction and trace-preserving property."""
    ch = bit_flip_channel(0.2)
    assert ch.num_qubits == 1
    assert len(ch.kraus_ops) == 2

    # Check trace-preserving: ∑ K_i† K_i = I
    dim = 2
    identity = torch.eye(dim, dtype=torch.complex128, device=ch.kraus_ops[0].device)
    sum_kdag_k = torch.zeros((dim, dim), dtype=torch.complex128, device=ch.kraus_ops[0].device)

    for K in ch.kraus_ops:
        Kdag = K.conj().T
        sum_kdag_k = sum_kdag_k + Kdag @ K

    diff = torch.abs(sum_kdag_k - identity)
    max_diff = torch.max(diff).item()
    assert max_diff < 1e-7


def test_depolarizing_channel_trace_preserving():
    """Test depolarizing channel is trace-preserving for various p values."""
    for p in [0.0, 0.3, 1.0]:
        ch = depolarizing_channel(p)
        assert ch.num_qubits == 1
        assert ch.is_trace_preserving()


def test_amplitude_damping_limits():
    """Test amplitude damping channel at limit cases."""
    # gamma=0 should be identity channel
    ch = amplitude_damping_channel(0.0)
    assert ch.num_qubits == 1
    assert ch.is_trace_preserving()

    # Check that K0 is identity
    k0 = ch.kraus_ops[0]
    identity = torch.eye(2, dtype=k0.dtype, device=k0.device)
    diff = torch.abs(k0 - identity)
    assert torch.max(diff).item() < 1e-7

    # gamma=1 should fully decay
    ch = amplitude_damping_channel(1.0)
    assert ch.is_trace_preserving()


def test_generalized_amplitude_damping_validation():
    """Test parameter validation for generalized amplitude damping."""
    # Valid parameters
    ch = generalized_amplitude_damping_channel(0.5, 0.3)
    assert ch.num_qubits == 1
    assert ch.is_trace_preserving()

    # Invalid gamma
    with pytest.raises(ValueError, match="gamma must be in"):
        generalized_amplitude_damping_channel(-0.1, 0.5)

    with pytest.raises(ValueError, match="gamma must be in"):
        generalized_amplitude_damping_channel(1.1, 0.5)

    # Invalid p_excited
    with pytest.raises(ValueError, match="p_excited must be in"):
        generalized_amplitude_damping_channel(0.5, -0.1)

    with pytest.raises(ValueError, match="p_excited must be in"):
        generalized_amplitude_damping_channel(0.5, 1.1)


def test_two_qubit_depolarizing_channel():
    """Test two-qubit depolarizing channel construction."""
    ch = two_qubit_depolarizing_channel(0.5)
    assert ch.num_qubits == 2
    assert len(ch.kraus_ops) == 16  # 1 identity + 15 Pauli terms

    # Check each Kraus operator has shape (4, 4)
    for K in ch.kraus_ops:
        assert K.shape == (4, 4)

    # Check trace-preserving
    assert ch.is_trace_preserving()


def test_kraus_channel_validation_errors():
    """Test KrausChannel validation raises appropriate errors."""
    device = default_device().as_torch_device()
    dtype = torch.complex128

    # Wrong dimension
    with pytest.raises(ValueError, match="must have shape"):
        wrong_k = torch.eye(1, dtype=dtype, device=device)
        KrausChannel(name="test", kraus_ops=(wrong_k,), num_qubits=1)

    # Non-trace-preserving
    with pytest.raises(ValueError, match="trace-preserving"):
        # Scale identity by 2, so K†K = 4I ≠ I
        non_tp_k = 2.0 * torch.eye(2, dtype=dtype, device=device)
        KrausChannel(name="test", kraus_ops=(non_tp_k,), num_qubits=1)

    # num_qubits <= 0
    with pytest.raises(ValueError, match="num_qubits must be at least 1"):
        k = torch.eye(2, dtype=dtype, device=device)
        KrausChannel(name="test", kraus_ops=(k,), num_qubits=0)

    # Empty kraus_ops
    with pytest.raises(ValueError, match="kraus_ops must contain at least one"):
        KrausChannel(name="test", kraus_ops=(), num_qubits=1)


def test_kraus_channel_to():
    """Test KrausChannel.to() method."""
    ch = depolarizing_channel(0.1)
    assert ch.is_trace_preserving()

    # Change dtype
    ch2 = ch.to(dtype=torch.complex64)
    assert ch2.is_trace_preserving()
    assert ch2.kraus_ops[0].dtype == torch.complex64

    # Change device (if CUDA available)
    if torch.cuda.is_available():
        ch3 = ch.to(device=torch.device("cuda"))
        assert ch3.is_trace_preserving()
        assert ch3.kraus_ops[0].device.type == "cuda"


def test_phase_flip_channel():
    """Test phase flip channel construction."""
    ch = phase_flip_channel(0.3)
    assert ch.num_qubits == 1
    assert ch.is_trace_preserving()


def test_bit_phase_flip_channel():
    """Test bit-phase flip channel construction."""
    ch = bit_phase_flip_channel(0.4)
    assert ch.num_qubits == 1
    assert ch.is_trace_preserving()


def test_phase_damping_channel():
    """Test phase damping channel construction."""
    ch = phase_damping_channel(0.5)
    assert ch.num_qubits == 1
    assert ch.is_trace_preserving()


def test_channel_parameter_validation():
    """Test that all channels validate parameter ranges."""
    # Bit flip
    with pytest.raises(ValueError):
        bit_flip_channel(-0.1)
    with pytest.raises(ValueError):
        bit_flip_channel(1.1)

    # Phase flip
    with pytest.raises(ValueError):
        phase_flip_channel(-0.1)
    with pytest.raises(ValueError):
        phase_flip_channel(1.1)

    # Bit-phase flip
    with pytest.raises(ValueError):
        bit_phase_flip_channel(-0.1)
    with pytest.raises(ValueError):
        bit_phase_flip_channel(1.1)

    # Depolarizing
    with pytest.raises(ValueError):
        depolarizing_channel(-0.1)
    with pytest.raises(ValueError):
        depolarizing_channel(1.1)

    # Phase damping
    with pytest.raises(ValueError):
        phase_damping_channel(-0.1)
    with pytest.raises(ValueError):
        phase_damping_channel(1.1)

    # Amplitude damping
    with pytest.raises(ValueError):
        amplitude_damping_channel(-0.1)
    with pytest.raises(ValueError):
        amplitude_damping_channel(1.1)

    # Two-qubit depolarizing
    with pytest.raises(ValueError):
        two_qubit_depolarizing_channel(-0.1)
    with pytest.raises(ValueError):
        two_qubit_depolarizing_channel(1.1)


