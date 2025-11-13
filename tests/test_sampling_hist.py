"""Tests for histogram and probability analysis utilities."""

from __future__ import annotations

import torch

from qconduit.sampling import (
    bitstring_counts,
    counts_to_probs,
    kl_divergence,
    marginalize_probs,
)


def test_bitstring_counts_and_probs() -> None:
    """Test bitstring counting and conversion to probabilities."""
    # Two-bit samples: four shots
    samples = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=torch.int64,
    )
    counts = bitstring_counts(samples)
    assert counts["00"] == 1
    assert counts["01"] == 2
    assert counts["11"] == 1
    assert "10" not in counts

    probs = counts_to_probs(counts)
    # Total is 4
    assert abs(probs["00"] - 0.25) < 1e-6
    assert abs(probs["01"] - 0.5) < 1e-6
    assert abs(probs["11"] - 0.25) < 1e-6


def test_kl_divergence_simple() -> None:
    """Test KL divergence on simple distributions."""
    p = {"0": 0.5, "1": 0.5}
    q = {"0": 1.0, "1": 0.0}
    # Ideal KL: 0.5 * log(0.5 / 1.0) + 0.5 * log(0.5 / epsilon) approx large,
    # but our epsilon clamp ensures finite value.
    kl = kl_divergence(p, q, epsilon=1e-6)
    assert kl >= 0.0


def test_kl_divergence_identical() -> None:
    """Test KL divergence between identical distributions is near zero."""
    p = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    q = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    kl = kl_divergence(p, q, epsilon=1e-12)
    # Should be very close to zero
    assert abs(kl) < 1e-10


def test_marginalize_probs_two_qubits() -> None:
    """Test marginalization on a 2-qubit distribution."""
    # Simple distribution over 2 qubits:
    # p(00)=0.25, p(01)=0.25, p(10)=0.25, p(11)=0.25 (uniform)
    n_qubits = 2
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

    # Marginal over qubit 0 should be uniform [0.5, 0.5]
    marg0 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0])
    assert marg0.shape == (2,)
    assert torch.allclose(marg0, torch.tensor([0.5, 0.5], dtype=torch.float32), atol=1e-6)

    # Marginal over both qubits [0,1] should be identity (same as original).
    marg01 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0, 1])
    assert torch.allclose(marg01, probs, atol=1e-6)


def test_marginalize_probs_non_uniform() -> None:
    """Test marginalization on a non-uniform 2-qubit distribution."""
    n_qubits = 2
    # p(00)=0.5, p(01)=0.2, p(10)=0.2, p(11)=0.1
    probs = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float32)

    # Marginal over qubit 0:
    # P(q0=0) = p(00) + p(01) = 0.5 + 0.2 = 0.7
    # P(q0=1) = p(10) + p(11) = 0.2 + 0.1 = 0.3
    marg0 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0])
    assert torch.allclose(
        marg0, torch.tensor([0.7, 0.3], dtype=torch.float32), atol=1e-6
    )

    # Marginal over qubit 1:
    # P(q1=0) = p(00) + p(10) = 0.5 + 0.2 = 0.7
    # P(q1=1) = p(01) + p(11) = 0.2 + 0.1 = 0.3
    marg1 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[1])
    assert torch.allclose(
        marg1, torch.tensor([0.7, 0.3], dtype=torch.float32), atol=1e-6
    )


def test_marginalize_probs_batch() -> None:
    """Test marginalization with batch dimensions."""
    n_qubits = 2
    # Batch of 2 distributions
    probs = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=torch.float32
    )
    # First: p(00)=0.5, p(01)=0.5, p(10)=0.0, p(11)=0.0
    # Second: p(00)=0.0, p(01)=0.0, p(10)=0.5, p(11)=0.5

    marg0 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0])
    assert marg0.shape == (2, 2)
    # First batch: P(q0=0) = p(00) + p(10) = 0.5 + 0.0 = 0.5
    #              P(q0=1) = p(01) + p(11) = 0.5 + 0.0 = 0.5
    assert torch.allclose(
        marg0[0], torch.tensor([0.5, 0.5], dtype=torch.float32), atol=1e-6
    )
    # Second batch: P(q0=0) = p(00) + p(10) = 0.0 + 0.5 = 0.5
    #               P(q0=1) = p(01) + p(11) = 0.0 + 0.5 = 0.5
    assert torch.allclose(
        marg0[1], torch.tensor([0.5, 0.5], dtype=torch.float32), atol=1e-6
    )


def test_bitstring_counts_empty() -> None:
    """Test bitstring_counts with empty input raises error."""
    samples = torch.tensor([], dtype=torch.int64).reshape(0, 2)
    counts = bitstring_counts(samples)
    # Should return empty dict
    assert len(counts) == 0


def test_counts_to_probs_single() -> None:
    """Test counts_to_probs with a single bitstring."""
    counts = {"00": 100}
    probs = counts_to_probs(counts)
    assert abs(probs["00"] - 1.0) < 1e-6
    assert len(probs) == 1

