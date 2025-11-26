"""Pytest configuration and shared fixtures for Quantum Conduit tests.

This module provides:
- Deterministic RNG fixtures for numpy and torch
- Test utilities for reproducible random number generation
"""

import os
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    """Provide a deterministic numpy RNG for tests.
    
    Uses seed from TEST_RNG_SEED environment variable (default: 0).
    This ensures tests are reproducible while allowing override for debugging.
    
    Returns:
        A seeded numpy.random.Generator instance.
    """
    seed = int(os.environ.get("TEST_RNG_SEED", "0"))
    return np.random.default_rng(seed)


@pytest.fixture(scope="function")
def torch_rng() -> torch.Generator:
    """Provide a deterministic torch RNG for tests.
    
    Uses seed from TEST_RNG_SEED environment variable (default: 0).
    Device is determined by default_device().
    
    Returns:
        A seeded torch.Generator instance.
    """
    from qconduit.core.device import default_device
    
    seed = int(os.environ.get("TEST_RNG_SEED", "0"))
    device = default_device().as_torch_device()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@pytest.fixture(scope="function", autouse=True)
def set_random_seeds(rng: np.random.Generator, torch_rng: torch.Generator) -> None:
    """Auto-use fixture to set global random seeds for reproducibility.
    
    This fixture runs automatically for every test to ensure deterministic behavior.
    """
    # Set numpy global seed (for legacy code that uses np.random directly)
    np.random.seed(int(os.environ.get("TEST_RNG_SEED", "0")))
    
    # Set torch global seed
    torch.manual_seed(int(os.environ.get("TEST_RNG_SEED", "0")))
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(os.environ.get("TEST_RNG_SEED", "0")))

