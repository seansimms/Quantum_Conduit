"""Tests for density matrix utilities and channel application."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.noise import (
    bit_flip_channel,
    phase_flip_channel,
    depolarizing_channel,
    phase_damping_channel,
    amplitude_damping_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
    to_density_matrix,
    apply_kraus_channel_to_density_matrix,
    apply_kraus_channel_to_statevector,
    compose_kraus_channels,
)
from qconduit.core.device import default_device


def test_to_density_matrix_zero_state():
    """Test converting |0⟩ to density matrix."""
    state = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    rho = to_density_matrix(state)

    assert rho.shape == (2, 2)
    assert torch.abs(rho[0, 0] - 1.0).item() < 1e-10
    assert torch.abs(rho[0, 1]).item() < 1e-10
    assert torch.abs(rho[1, 0]).item() < 1e-10
    assert torch.abs(rho[1, 1]).item() < 1e-10


def test_to_density_matrix_plus_state():
    """Test converting |+⟩ to density matrix."""
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex128)
    rho = to_density_matrix(state)

    assert rho.shape == (2, 2)
    # Diagonal entries should be 0.5
    assert torch.abs(rho[0, 0] - 0.5).item() < 1e-10
    assert torch.abs(rho[1, 1] - 0.5).item() < 1e-10
    # Off-diagonal entries should be 0.5
    assert torch.abs(rho[0, 1] - 0.5).item() < 1e-10
    assert torch.abs(rho[1, 0] - 0.5).item() < 1e-10


def test_to_density_matrix_invalid_dimensions():
    """Test to_density_matrix raises error for invalid input."""
    # 2D input
    state_2d = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="must be 1D"):
        to_density_matrix(state_2d)

    # Non-power-of-2 dimension
    state_3d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="not a power of 2"):
        to_density_matrix(state_3d)


def test_bit_flip_on_zero_state():
    """Test bit flip channel on |0⟩."""
    # Start with |0⟩
    state_0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    rho0 = to_density_matrix(state_0)

    # Apply bit flip channel with p=0.3
    ch = bit_flip_channel(0.3)
    rho1 = apply_kraus_channel_to_density_matrix(rho0, ch)

    # Analytical result: ρ' = (1-p)|0⟩⟨0| + p|1⟩⟨1|
    # Diagonal: [1-p, p] = [0.7, 0.3]
    assert torch.abs(rho1[0, 0] - 0.7).item() < 1e-7
    assert torch.abs(rho1[1, 1] - 0.3).item() < 1e-7
    # Off-diagonals should be zero
    assert torch.abs(rho1[0, 1]).item() < 1e-7
    assert torch.abs(rho1[1, 0]).item() < 1e-7


def test_bit_flip_on_one_state():
    """Test bit flip channel on |1⟩."""
    # Start with |1⟩
    state_1 = torch.tensor([0.0, 1.0], dtype=torch.complex128)
    rho0 = to_density_matrix(state_1)

    # Apply bit flip channel with p=0.3
    ch = bit_flip_channel(0.3)
    rho1 = apply_kraus_channel_to_density_matrix(rho0, ch)

    # Analytical result: ρ' = p|0⟩⟨0| + (1-p)|1⟩⟨1|
    # Diagonal: [p, 1-p] = [0.3, 0.7]
    assert torch.abs(rho1[0, 0] - 0.3).item() < 1e-7
    assert torch.abs(rho1[1, 1] - 0.7).item() < 1e-7


def test_amplitude_damping_on_one_state():
    """Test amplitude damping channel on |1⟩."""
    # Start with |1⟩
    state_1 = torch.tensor([0.0, 1.0], dtype=torch.complex128)
    rho_in = to_density_matrix(state_1)

    # Apply amplitude damping with gamma=0.6
    gamma = 0.6
    ch = amplitude_damping_channel(gamma)
    rho_out = apply_kraus_channel_to_density_matrix(rho_in, ch)

    # Analytical result:
    # P(|0⟩) = gamma = 0.6
    # P(|1⟩) = 1 - gamma = 0.4
    assert torch.abs(rho_out[0, 0] - gamma).item() < 1e-7
    assert torch.abs(rho_out[1, 1] - (1 - gamma)).item() < 1e-7
    # Off-diagonals should be zero
    assert torch.abs(rho_out[0, 1]).item() < 1e-7
    assert torch.abs(rho_out[1, 0]).item() < 1e-7


def test_phase_damping_on_plus_state():
    """Test phase damping channel on |+⟩."""
    # Start with |+⟩ = (1/√2)(|0⟩ + |1⟩)
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state_plus = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex128)
    rho_in = to_density_matrix(state_plus)

    # Apply phase damping with gamma=0.5
    gamma = 0.5
    ch = phase_damping_channel(gamma)
    rho_out = apply_kraus_channel_to_density_matrix(rho_in, ch)

    # Diagonal should be unchanged: 0.5, 0.5
    assert torch.abs(rho_out[0, 0] - 0.5).item() < 1e-7
    assert torch.abs(rho_out[1, 1] - 0.5).item() < 1e-7

    # Off-diagonal coherence should be multiplied by sqrt(1 - gamma)
    # Original off-diagonal: 0.5
    # After phase damping: 0.5 * sqrt(1 - gamma) = 0.5 * sqrt(0.5) ≈ 0.3536
    expected_off_diag = 0.5 * math.sqrt(1 - gamma)
    assert torch.abs(rho_out[0, 1] - expected_off_diag).item() < 1e-7
    assert torch.abs(rho_out[1, 0] - expected_off_diag).item() < 1e-7


def test_apply_kraus_channel_to_statevector_consistency():
    """Test consistency between statevector and density matrix application."""
    # Start with |0⟩
    state = torch.tensor([1.0, 0.0], dtype=torch.complex128)

    # Apply via statevector
    ch = depolarizing_channel(0.4)
    rho1 = apply_kraus_channel_to_statevector(state, ch)

    # Apply via density matrix
    rho0 = to_density_matrix(state)
    rho2 = apply_kraus_channel_to_density_matrix(rho0, ch)

    # Results should be identical
    diff = torch.abs(rho1 - rho2)
    assert torch.max(diff).item() < 1e-7


def test_compose_kraus_channels():
    """Test composition of two Kraus channels."""
    # Compose two bit-flip channels
    ch1 = bit_flip_channel(0.3)
    ch2 = bit_flip_channel(0.5)
    ch12 = compose_kraus_channels(ch1, ch2)

    # Start with |0⟩
    state_0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    rho0 = to_density_matrix(state_0)

    # Apply channels sequentially
    rho_a = apply_kraus_channel_to_density_matrix(rho0, ch1)
    rho_b = apply_kraus_channel_to_density_matrix(rho_a, ch2)

    # Apply composite channel
    rho_c = apply_kraus_channel_to_density_matrix(rho0, ch12)

    # Results should be identical
    diff = torch.abs(rho_b - rho_c)
    assert torch.max(diff).item() < 1e-7


def test_compose_kraus_channels_qubit_mismatch():
    """Test that composing channels with different qubit counts raises error."""
    ch1 = bit_flip_channel(0.3)  # 1 qubit
    ch2 = two_qubit_depolarizing_channel(0.5)  # 2 qubits

    with pytest.raises(ValueError, match="different qubit counts"):
        compose_kraus_channels(ch1, ch2)


def test_two_qubit_depolarizing_sanity():
    """Test two-qubit depolarizing channel on |00⟩."""
    # Start with |00⟩ = [1, 0, 0, 0]
    state_00 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
    rho_in = to_density_matrix(state_00)

    # Apply two-qubit depolarizing with p=0.5
    p = 0.5
    ch = two_qubit_depolarizing_channel(p)
    rho_out = apply_kraus_channel_to_density_matrix(rho_in, ch)

    # Check Hermiticity
    diff_herm = torch.abs(rho_out - rho_out.conj().T)
    assert torch.max(diff_herm).item() < 1e-7

    # Check trace is 1
    trace = torch.trace(rho_out).real.item()
    assert abs(trace - 1.0) < 1e-7

    # For p=0, should be identity (unchanged)
    ch_identity = two_qubit_depolarizing_channel(0.0)
    rho_identity = apply_kraus_channel_to_density_matrix(rho_in, ch_identity)
    diff_identity = torch.abs(rho_identity - rho_in)
    assert torch.max(diff_identity).item() < 1e-7

    # For p=1, the channel is: E(ρ) = (1/15) * sum_{P != I⊗I} P ρ P
    # This does not necessarily give I/4 for all inputs
    # Instead, verify that the channel is trace-preserving and Hermitian
    ch_max_mixed = two_qubit_depolarizing_channel(1.0)
    rho_max_mixed = apply_kraus_channel_to_density_matrix(rho_in, ch_max_mixed)
    # Check trace is 1
    trace = torch.trace(rho_max_mixed).real.item()
    assert abs(trace - 1.0) < 1e-7
    # Check Hermiticity
    diff_herm = torch.abs(rho_max_mixed - rho_max_mixed.conj().T)
    assert torch.max(diff_herm).item() < 1e-7


def test_apply_channel_dimension_mismatch():
    """Test that applying channel with wrong dimension raises error."""
    # 1-qubit channel on 2-qubit density matrix
    ch = bit_flip_channel(0.3)
    rho_2q = torch.eye(4, dtype=torch.complex128) / 4.0  # 2-qubit maximally mixed

    with pytest.raises(ValueError, match="Channel qubit count"):
        apply_kraus_channel_to_density_matrix(rho_2q, ch)


def test_apply_channel_non_square_rho():
    """Test that applying channel to non-square rho raises error."""
    ch = bit_flip_channel(0.3)
    rho_non_square = torch.zeros((2, 3), dtype=torch.complex128)

    with pytest.raises(ValueError, match="must be square"):
        apply_kraus_channel_to_density_matrix(rho_non_square, ch)


def test_generalized_amplitude_damping():
    """Test generalized amplitude damping channel."""
    # Start with |1⟩
    state_1 = torch.tensor([0.0, 1.0], dtype=torch.complex128)
    rho_in = to_density_matrix(state_1)

    # Apply generalized amplitude damping
    gamma = 0.6
    p_excited = 0.3
    ch = generalized_amplitude_damping_channel(gamma, p_excited)
    rho_out = apply_kraus_channel_to_density_matrix(rho_in, ch)

    # Check trace is preserved
    trace = torch.trace(rho_out).real.item()
    assert abs(trace - 1.0) < 1e-7

    # Check Hermiticity
    diff_herm = torch.abs(rho_out - rho_out.conj().T)
    assert torch.max(diff_herm).item() < 1e-7

