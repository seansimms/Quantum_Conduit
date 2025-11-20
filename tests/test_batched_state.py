"""Tests for batched state utilities."""


import pytest
import torch

from qconduit.batched import BatchedState


def test_batched_state_construction_from_batched():
    """Test BatchedState construction from already-batched tensor."""
    B = 4
    dim = 8  # 3 qubits
    states = torch.randn(B, dim, dtype=torch.complex128)
    # Normalize each row
    norms = torch.linalg.norm(states, dim=1, keepdim=True)
    states = states / torch.clamp(norms, min=1e-12)

    batched = BatchedState(states, n_qubits=3)
    assert batched.states.shape == (B, dim)
    assert batched.n_qubits == 3


def test_batched_state_construction_from_single():
    """Test BatchedState.from_statevector with single statevector."""
    dim = 4  # 2 qubits
    state = torch.randn(dim, dtype=torch.complex128)
    state = state / torch.linalg.norm(state)

    batched = BatchedState.from_statevector(state)
    assert batched.states.shape == (1, dim)
    assert batched.n_qubits == 2


def test_batched_state_validation_non_power_of_two():
    """Test that non-power-of-two dimensions raise ValueError."""
    states = torch.randn(3, 5, dtype=torch.complex128)  # 5 is not a power of 2
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    with pytest.raises(ValueError, match="not a power of 2"):
        BatchedState(states, n_qubits=2)


def test_batched_state_validation_wrong_n_qubits():
    """Test that mismatched n_qubits raises ValueError."""
    dim = 8  # 3 qubits
    states = torch.randn(3, dim, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    with pytest.raises(ValueError, match="does not match"):
        BatchedState(states, n_qubits=2)


def test_batched_state_validation_zero_norm():
    """Test that zero-norm states raise ValueError."""
    B = 3
    dim = 4
    states = torch.randn(B, dim, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    # Set one row to zero
    states[1] = 0.0

    with pytest.raises(ValueError, match="zero-norm"):
        BatchedState(states, n_qubits=2)


def test_batched_state_validation_wrong_ndim():
    """Test that non-2D states raise ValueError."""
    state = torch.randn(4, dtype=torch.complex128)
    state = state / torch.linalg.norm(state)

    with pytest.raises(ValueError, match="must be 2D"):
        BatchedState(state.unsqueeze(0).unsqueeze(0), n_qubits=2)  # 3D


def test_batched_state_to_device_dtype():
    """Test device and dtype conversion."""
    states = torch.randn(3, 4, dtype=torch.complex64)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=2)

    # Convert to complex128
    batched_128 = batched.to(dtype=torch.complex128)
    assert batched_128.states.dtype == torch.complex128
    assert batched_128.n_qubits == batched.n_qubits


def test_batched_state_unstack():
    """Test unstack method."""
    B = 3
    dim = 4
    states = torch.randn(B, dim, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=2)

    unstacked = batched.unstack()
    assert len(unstacked) == B
    for i, state in enumerate(unstacked):
        assert state.shape == (dim,)
        assert torch.allclose(state, batched.states[i])


def test_batched_state_norms():
    """Test norms method."""
    B = 4
    dim = 8
    states = torch.randn(B, dim, dtype=torch.complex128)
    # Normalize each row
    norms_target = torch.ones(B) * 1.5  # Different norms
    states = states / torch.linalg.norm(states, dim=1, keepdim=True) * norms_target.unsqueeze(1)

    # Create with normalized states (validation will normalize)
    states_normalized = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states_normalized, n_qubits=3)

    norms = batched.norms()
    assert norms.shape == (B,)
    assert torch.allclose(norms, torch.ones(B, dtype=torch.float64), atol=1e-10)


def test_batched_state_renormalize():
    """Test renormalize method."""
    B = 3
    dim = 4
    # Create unnormalized states
    states = torch.randn(B, dim, dtype=torch.complex128)
    states[0] = states[0] * 2.0
    states[1] = states[1] * 0.5
    states[2] = states[2] * 3.0
    # Normalize for validation
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    batched = BatchedState(states, n_qubits=2)

    # Create unnormalized version
    states_unnorm = batched.states * torch.tensor([2.0, 0.5, 3.0], dtype=torch.complex128).unsqueeze(1)
    batched_unnorm = BatchedState(states_unnorm, n_qubits=2)

    # Renormalize
    normalized = batched_unnorm.renormalize()
    norms = normalized.norms()
    assert torch.allclose(norms, torch.ones(B, dtype=torch.float64), atol=1e-10)


def test_batched_state_from_statevector_already_batched():
    """Test from_statevector with already-batched tensor."""
    B = 5
    dim = 4
    states = torch.randn(B, dim, dtype=torch.complex128)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    batched = BatchedState.from_statevector(states)
    assert batched.states.shape == (B, dim)
    assert batched.n_qubits == 2


def test_batched_state_complex_dtype_conversion():
    """Test that real tensors are converted to complex."""
    states = torch.randn(3, 4, dtype=torch.float32)
    states = states / torch.linalg.norm(states, dim=1, keepdim=True)

    batched = BatchedState(states, n_qubits=2)
    assert torch.is_complex(batched.states)
    assert batched.states.dtype == torch.complex128

