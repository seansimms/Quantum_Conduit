"""Smoke test to verify test determinism.

This module runs a subset of randomized tests twice and asserts identical outputs
when using the seeded RNG fixture. This helps catch flaky tests early.
"""

import os

import numpy as np
import pytest
import torch

from qconduit.sampling import sample_from_probs


def test_deterministic_sampling_reproducibility(rng: np.random.Generator, torch_rng: torch.Generator) -> None:
    """Test that sampling with same seed produces identical results."""
    # Create a simple probability distribution
    probs = torch.tensor([0.3, 0.7], dtype=torch.float32)
    
    # Sample twice with same seed
    torch_rng.manual_seed(42)
    samples1 = sample_from_probs(probs, n_qubits=1, n_shots=100, generator=torch_rng)
    
    torch_rng.manual_seed(42)
    samples2 = sample_from_probs(probs, n_qubits=1, n_shots=100, generator=torch_rng)
    
    # Results should be identical
    assert torch.equal(samples1, samples2), "Sampling should be deterministic with same seed"


def test_numpy_rng_reproducibility(rng: np.random.Generator) -> None:
    """Test that numpy RNG fixture produces reproducible results."""
    # Generate random numbers
    values1 = rng.random(10)
    
    # Create new RNG with same seed
    seed = int(os.environ.get("TEST_RNG_SEED", "0"))
    rng2 = np.random.default_rng(seed)
    values2 = rng2.random(10)
    
    # Results should be identical
    np.testing.assert_array_equal(values1, values2)


def test_torch_rng_reproducibility(torch_rng: torch.Generator) -> None:
    """Test that torch RNG fixture produces reproducible results."""
    # Generate random numbers
    torch_rng.manual_seed(123)
    values1 = torch.rand(10, generator=torch_rng)
    
    # Reset and generate again
    torch_rng.manual_seed(123)
    values2 = torch.rand(10, generator=torch_rng)
    
    # Results should be identical
    assert torch.equal(values1, values2), "Torch RNG should be deterministic with same seed"

