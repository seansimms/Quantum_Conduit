"""Tests for measurement and sampling utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.measurement import (
    basis_probabilities_from_statevector,
    sample_bitstrings_from_probabilities,
    sample_bitstrings_from_statevector,
    bitstring_counts,
    empirical_probabilities_from_bitstrings,
    estimate_pauli_z_expectation_from_samples,
)
from qconduit.core.device import default_device


def test_basis_probabilities_simple_states():
    """Test basis probabilities for simple states."""
    # |0⟩ for 1 qubit
    state = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    probs = basis_probabilities_from_statevector(state)
    assert probs.shape == (2,)
    assert torch.allclose(probs[0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(probs[1], torch.tensor(0.0, dtype=torch.float64))
    
    # |+⟩ for 1 qubit: (1/√2)(|0⟩ + |1⟩)
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state_plus = torch.tensor([sqrt2_inv + 0j, sqrt2_inv + 0j], dtype=torch.complex128)
    probs_plus = basis_probabilities_from_statevector(state_plus)
    assert probs_plus.shape == (2,)
    assert torch.allclose(probs_plus[0], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(probs_plus[1], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)
    
    # Invalid state: non-power-of-two length
    state_invalid = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="power of 2"):
        basis_probabilities_from_statevector(state_invalid)


def test_sampling_from_probabilities():
    """Test sampling from probability distributions."""
    device = default_device().as_torch_device()
    probs = torch.tensor([0.25, 0.75], dtype=torch.float64, device=device)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    
    bits = sample_bitstrings_from_probabilities(
        probs, num_qubits=1, n_shots=1000, generator=gen
    )
    
    assert bits.shape == (1000, 1)
    assert torch.all((bits == 0) | (bits == 1))
    
    # Check empirical frequencies are approximately correct
    counts = bitstring_counts(bits)
    emp_probs = empirical_probabilities_from_bitstrings(bits)
    
    # Should be approximately (0.25, 0.75) within ±0.05
    assert abs(emp_probs[0].item() - 0.25) < 0.05
    assert abs(emp_probs[1].item() - 0.75) < 0.05


def test_sampling_from_statevector():
    """Test sampling from statevectors matches probabilities."""
    # 2-qubit example: |ψ⟩ = |00⟩
    state = torch.tensor([1.0 + 0j, 0, 0, 0], dtype=torch.complex128)
    
    device = default_device().as_torch_device()
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    
    bitstrings, probs = sample_bitstrings_from_statevector(
        state, n_shots=256, generator=gen
    )
    
    assert bitstrings.shape == (256, 2)
    # All sampled bitstrings must be [0, 0]
    assert torch.all(bitstrings == 0)
    assert torch.allclose(probs[0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(probs[1:], torch.tensor(0.0, dtype=torch.float64))
    
    # Another 2-qubit example: |+⟩⊗|0⟩ = (1/√2)(|00⟩ + |10⟩)
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state_plus_zero = torch.tensor(
        [sqrt2_inv + 0j, 0, sqrt2_inv + 0j, 0], dtype=torch.complex128
    )
    
    gen2 = torch.Generator(device=device)
    gen2.manual_seed(123)
    
    bitstrings2, probs2 = sample_bitstrings_from_statevector(
        state_plus_zero, n_shots=1000, generator=gen2
    )
    
    assert bitstrings2.shape == (1000, 2)
    # Analytical probabilities: P(00) = 0.5, P(10) = 0.5
    assert torch.allclose(probs2[0], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(probs2[2], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(probs2[1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(probs2[3], torch.tensor(0.0, dtype=torch.float64))
    
    # Check empirical frequencies
    emp_probs2 = empirical_probabilities_from_bitstrings(bitstrings2)
    assert abs(emp_probs2[0].item() - 0.5) < 0.05
    assert abs(emp_probs2[2].item() - 0.5) < 0.05


def test_bitstring_counts_and_empirical_probs():
    """Test histogram and empirical probability computation."""
    # Build a small bitstring tensor by hand for 2 qubits
    bitstrings = torch.tensor(
        [[0, 0], [0, 1], [0, 0], [1, 1]], dtype=torch.int64
    )
    
    counts = bitstring_counts(bitstrings)
    
    # Indices: 0→00, 1→01, 2→10, 3→11
    # Expected counts: [2, 1, 0, 1]
    assert counts.shape == (4,)
    assert counts[0].item() == 2  # 00 appears twice
    assert counts[1].item() == 1  # 01 appears once
    assert counts[2].item() == 0  # 10 appears zero times
    assert counts[3].item() == 1  # 11 appears once
    
    probs = empirical_probabilities_from_bitstrings(bitstrings)
    assert torch.allclose(probs[0], torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(probs[1], torch.tensor(0.25, dtype=torch.float64))
    assert torch.allclose(probs[2], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(probs[3], torch.tensor(0.25, dtype=torch.float64))
    
    # Non-binary entries should raise ValueError
    bitstrings_invalid = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="only 0 and 1"):
        bitstring_counts(bitstrings_invalid)


def test_estimate_pauli_z_expectation():
    """Test Z expectation estimation from samples."""
    # For n=1, bitstrings with all zeros: ⟨Z⟩ = +1
    bitstrings_all_zeros = torch.zeros((10, 1), dtype=torch.int64)
    mean, stderr = estimate_pauli_z_expectation_from_samples(
        bitstrings_all_zeros, qubit_index=0
    )
    assert abs(mean - 1.0) < 1e-10
    assert stderr == 0.0  # All values are the same
    
    # For half zeros, half ones: ⟨Z⟩ ≈ 0
    bitstrings_mixed = torch.tensor([[0], [1]] * 5, dtype=torch.int64)
    mean2, stderr2 = estimate_pauli_z_expectation_from_samples(
        bitstrings_mixed, qubit_index=0
    )
    assert abs(mean2) < 1e-10
    assert stderr2 > 0  # There is variance
    
    # Invalid qubit_index
    with pytest.raises(ValueError, match="qubit_index must be in"):
        estimate_pauli_z_expectation_from_samples(
            bitstrings_all_zeros, qubit_index=-1
        )
    
    with pytest.raises(ValueError, match="qubit_index must be in"):
        estimate_pauli_z_expectation_from_samples(
            bitstrings_all_zeros, qubit_index=1
        )


def test_sampling_error_paths():
    """Test error handling in sampling functions."""
    # Invalid num_qubits
    probs = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="num_qubits must be >= 1"):
        sample_bitstrings_from_probabilities(probs, num_qubits=0, n_shots=10)
    
    # Invalid n_shots
    with pytest.raises(ValueError, match="n_shots must be >= 1"):
        sample_bitstrings_from_probabilities(probs, num_qubits=1, n_shots=0)
    
    # Wrong probs shape (2D)
    probs_2d = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="probs must be 1D"):
        sample_bitstrings_from_probabilities(probs_2d, num_qubits=1, n_shots=10)
    
    # Wrong probs shape (wrong length)
    probs_wrong = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="probs.shape"):
        sample_bitstrings_from_probabilities(probs_wrong, num_qubits=1, n_shots=10)
    
    # Negative probabilities
    probs_neg = torch.tensor([-0.1, 1.1], dtype=torch.float64)
    with pytest.raises(ValueError, match="negative"):
        sample_bitstrings_from_probabilities(probs_neg, num_qubits=1, n_shots=10)
    
    # Zero-sum probabilities
    probs_zero = torch.tensor([0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="sum to zero"):
        sample_bitstrings_from_probabilities(probs_zero, num_qubits=1, n_shots=10)
    
    # Invalid statevector (non-1D)
    state_2d = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="1D"):
        basis_probabilities_from_statevector(state_2d)
    
    # Zero norm statevector
    state_zero = torch.tensor([0.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="zero norm"):
        basis_probabilities_from_statevector(state_zero)
    
    # Empty statevector (dim <= 0)
    # This is hard to create directly, but we can test with explicit device/dtype
    state = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    probs = basis_probabilities_from_statevector(state, device=torch.device("cpu"), dtype=torch.float32)
    assert probs.dtype == torch.float32
    
    # Test bitstring_counts with 1D input
    bitstrings_1d = torch.tensor([0, 1, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="must be 2D"):
        bitstring_counts(bitstrings_1d)
    
    # Test empirical_probabilities_from_bitstrings with empty input
    bitstrings_empty = torch.zeros((0, 2), dtype=torch.int64)
    with pytest.raises(ValueError, match="empty"):
        empirical_probabilities_from_bitstrings(bitstrings_empty)
    
    # Test estimate_pauli_z_expectation_from_samples with 1D input
    bitstrings_1d = torch.tensor([0, 1, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="must be 2D"):
        estimate_pauli_z_expectation_from_samples(bitstrings_1d, qubit_index=0)
    
    # Test estimate_pauli_z_expectation_from_samples with invalid bits
    bitstrings_invalid = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="only 0 and 1"):
        estimate_pauli_z_expectation_from_samples(bitstrings_invalid, qubit_index=1)
    
    # Test n_shots == 1 case (should have stderr == 0.0)
    bitstrings_single = torch.tensor([[0]], dtype=torch.int64)
    mean, stderr = estimate_pauli_z_expectation_from_samples(bitstrings_single, qubit_index=0)
    assert abs(mean - 1.0) < 1e-10
    assert stderr == 0.0

