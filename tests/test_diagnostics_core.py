"""Tests for core diagnostic functions."""

import pytest
import torch

from qconduit.diagnostics import (
    state_norm,
    assert_normalized,
    is_hermitian,
    assert_hermitian,
    fidelity,
    bloch_vector,
)


def test_state_norm_and_assert_normalized() -> None:
    """Test state_norm and assert_normalized on normalized states."""
    state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    n = state_norm(state)
    assert n.shape == ()
    assert torch.allclose(n, torch.tensor(1.0))

    # Should not raise
    assert_normalized(state, atol=1e-6)


def test_assert_normalized_raises_for_non_unit_state() -> None:
    """Test that assert_normalized raises for non-normalized states."""
    state = torch.tensor([2.0, 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="not normalized"):
        assert_normalized(state, atol=1e-6)


def test_state_norm_batched() -> None:
    """Test state_norm on batched states."""
    # Batch of 3 states
    states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0 / torch.sqrt(torch.tensor(2.0)), 1.0 / torch.sqrt(torch.tensor(2.0))]],
        dtype=torch.complex64,
    )
    norms = state_norm(states)
    assert norms.shape == (3,)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


def test_is_hermitian_and_assert() -> None:
    """Test is_hermitian and assert_hermitian on Hermitian matrices."""
    mat = torch.tensor([[1.0, 1.0j], [-1.0j, 2.0]], dtype=torch.complex64)
    assert is_hermitian(mat)

    # Should not raise
    assert_hermitian(mat)

    non_herm = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex64)
    assert not is_hermitian(non_herm)

    with pytest.raises(ValueError, match="not Hermitian"):
        assert_hermitian(non_herm)


def test_is_hermitian_real_symmetric() -> None:
    """Test that real symmetric matrices are Hermitian."""
    mat = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    # Convert to complex for the function
    mat_complex = mat.to(torch.complex64)
    assert is_hermitian(mat_complex)


def test_fidelity_pure_states() -> None:
    """Test fidelity for pure statevectors."""
    # |0>, |1>
    zero = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    one = torch.tensor([0.0, 1.0], dtype=torch.complex64)
    f_same = fidelity(zero, zero)
    f_orth = fidelity(zero, one)
    assert torch.allclose(f_same, torch.tensor(1.0))
    assert torch.allclose(f_orth, torch.tensor(0.0))

    # |+> and |-> states
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    plus = torch.tensor([1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.complex64)
    minus = torch.tensor([1.0 / sqrt2, -1.0 / sqrt2], dtype=torch.complex64)
    f_plus_minus = fidelity(plus, minus)
    assert torch.allclose(f_plus_minus, torch.tensor(0.0), atol=1e-6)

    # |+> and |+> should have fidelity 1
    f_plus_plus = fidelity(plus, plus)
    assert torch.allclose(f_plus_plus, torch.tensor(1.0), atol=1e-6)


def test_fidelity_diagonal_density_matrices() -> None:
    """Test fidelity for diagonal density matrices."""
    # rho = diag(0.5, 0.5), sigma = diag(1.0, 0.0)
    rho = torch.zeros(2, 2, dtype=torch.complex64)
    rho[0, 0] = 0.5
    rho[1, 1] = 0.5

    sigma = torch.zeros(2, 2, dtype=torch.complex64)
    sigma[0, 0] = 1.0

    f = fidelity(rho, sigma)
    # sqrt(0.5 * 1) + sqrt(0.5 * 0) = sqrt(0.5); squared is 0.5
    assert torch.allclose(f, torch.tensor(0.5), atol=1e-6)

    # Same density matrix should have fidelity 1
    f_same = fidelity(rho, rho)
    assert torch.allclose(f_same, torch.tensor(1.0), atol=1e-6)


def test_fidelity_shape_mismatch() -> None:
    """Test that fidelity raises on shape mismatch."""
    state_a = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    state_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="same shape"):
        fidelity(state_a, state_b)


def test_bloch_vector_basic_states() -> None:
    """Test bloch_vector on standard single-qubit states."""
    zero = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    one = torch.tensor([0.0, 1.0], dtype=torch.complex64)
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    plus = torch.tensor([1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.complex64)

    v_zero = bloch_vector(zero)
    v_one = bloch_vector(one)
    v_plus = bloch_vector(plus)

    # |0> -> (0, 0, 1)
    assert torch.allclose(v_zero, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

    # |1> -> (0, 0, -1)
    assert torch.allclose(v_one, torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)

    # |+> -> (1, 0, 0)
    assert torch.allclose(v_plus, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_bloch_vector_wrong_dimension() -> None:
    """Test that bloch_vector raises on non-single-qubit states."""
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="last dimension 2"):
        bloch_vector(state)


def test_bloch_vector_batched() -> None:
    """Test bloch_vector on batched single-qubit states."""
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0 / sqrt2, 1.0 / sqrt2]],
        dtype=torch.complex64,
    )
    vectors = bloch_vector(states)
    assert vectors.shape == (3, 3)
    # Check first state |0>
    assert torch.allclose(vectors[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    # Check second state |1>
    assert torch.allclose(vectors[1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)
    # Check third state |+>
    assert torch.allclose(vectors[2], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_state_norm_empty_tensor() -> None:
    """Test that state_norm works on empty tensor (edge case)."""
    # An empty tensor still has 1 dimension, so state_norm should work
    # but the norm will be 0
    state = torch.tensor([], dtype=torch.complex64)
    n = state_norm(state)
    assert n.shape == ()
    assert torch.allclose(n, torch.tensor(0.0))


def test_assert_normalized_non_finite() -> None:
    """Test that assert_normalized raises on non-finite norms."""
    state = torch.tensor([float("inf"), 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="non-finite"):
        assert_normalized(state, atol=1e-6)

