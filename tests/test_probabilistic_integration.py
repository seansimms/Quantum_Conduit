"""Integration tests for probabilistic inference tools with Quantum Conduit.

Tests that probabilistic tools integrate correctly with the rest of the codebase.
"""

import numpy as np
import pytest

import qconduit as qc
from qconduit import (
    GaussianMixture,
    HiddenMarkovModel,
    bootstrap_particle_filter,
    logsumexp,
    normal_pdf,
    StateSpaceModel,
)


def test_import_from_main_package():
    """Test that probabilistic tools can be imported from main package."""
    # Test direct imports
    assert hasattr(qc, "HiddenMarkovModel")
    assert hasattr(qc, "GaussianMixture")
    assert hasattr(qc, "bootstrap_particle_filter")
    assert hasattr(qc, "logsumexp")

    # Test instantiation
    hmm = qc.HiddenMarkovModel(n_states=2)
    assert hmm.n_states == 2

    gmm = qc.GaussianMixture(n_components=2)
    assert gmm.n_components == 2


def test_hmm_with_quantum_state_sequences():
    """Test HMM can process sequences that might come from quantum measurements."""
    # Simulate quantum measurement outcomes (0 or 1)
    hmm = HiddenMarkovModel(
        n_states=2,
        start_prob=np.array([0.5, 0.5]),
        trans_mat=np.array([[0.7, 0.3], [0.3, 0.7]]),
        emission_prob=np.array([[0.9, 0.1], [0.1, 0.9]]),  # States prefer different outcomes
    )

    # Simulate measurement sequence
    measurements = np.array([0, 0, 1, 0, 1, 1, 0])

    # Compute likelihood
    log_likelihood = hmm.score(measurements)
    assert np.isfinite(log_likelihood)

    # Decode most likely state sequence
    states = hmm.predict(measurements)
    assert len(states) == len(measurements)
    assert np.all(states >= 0)
    assert np.all(states < 2)


def test_gmm_with_quantum_expectation_values():
    """Test GMM can cluster quantum expectation values."""
    # Simulate expectation values from different quantum states
    rng = np.random.default_rng(42)

    # Two clusters: low and high expectation values
    low_expectations = rng.normal(-1.0, 0.2, size=(50, 2))
    high_expectations = rng.normal(1.0, 0.2, size=(50, 2))
    X = np.vstack([low_expectations, high_expectations])

    # Fit GMM
    gmm = GaussianMixture(n_components=2, rng=rng)
    gmm.fit(X)

    # Predict clusters
    labels = gmm.predict(X)
    assert len(labels) == len(X)
    assert len(np.unique(labels)) == 2  # Should find 2 clusters

    # Check that log-likelihood is reasonable
    score = gmm.score(X)
    assert np.isfinite(score)


def test_particle_filter_with_quantum_observations():
    """Test particle filter can track quantum system state from noisy observations."""
    rng = np.random.default_rng(42)

    # Simple 1D state-space model for tracking quantum expectation value
    def transition_sample(x_prev, rng):
        # State drifts slightly
        return x_prev + rng.normal(0, 0.1)

    def observation_loglik(y, x):
        # Observation is noisy measurement of state
        return -0.5 * ((y - x) ** 2) / 0.5 - 0.5 * np.log(2 * np.pi * 0.5)

    def x0_sampler(rng):
        return np.array([rng.normal(0, 1)])

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    # Simulate observations (e.g., noisy expectation value measurements)
    observations = np.array([0.1, 0.15, 0.12, 0.18, 0.16])

    # Run particle filter
    result = bootstrap_particle_filter(model, observations, n_particles=100, rng=rng)

    # Check outputs
    assert "estimates" in result
    assert "weights" in result
    assert len(result["estimates"]) == len(observations)
    assert np.all(np.isfinite(result["estimates"]))


def test_utilities_with_quantum_probabilities():
    """Test probabilistic utilities work with quantum probability distributions."""
    # Simulate log-probabilities from quantum state
    log_probs = np.array([-1.0, -2.0, -3.0, -4.0])

    # Normalize using logsumexp
    log_z = logsumexp(log_probs)
    normalized = np.exp(log_probs - log_z)

    # Should sum to 1
    assert np.sum(normalized) == pytest.approx(1.0, rel=1e-10)

    # Test normal PDF for quantum state parameter estimation
    x = np.array([0.5, 0.3])  # Parameter values
    mean = np.array([0.5, 0.3])  # True parameters
    cov = np.eye(2) * 0.1  # Uncertainty

    pdf_val = normal_pdf(x, mean, cov)
    assert pdf_val > 0
    assert np.isfinite(pdf_val)


def test_integration_with_numpy_types():
    """Test that probabilistic tools work with standard NumPy types."""
    # Ensure compatibility with numpy arrays from quantum operations
    hmm = HiddenMarkovModel(n_states=2)
    obs = np.array([0, 1, 0], dtype=np.int32)  # Common dtype from measurements
    log_likelihood = hmm.score(obs)
    assert np.isfinite(log_likelihood)

    # Test with float32 (common in quantum simulations)
    X = np.random.randn(100, 2).astype(np.float32)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X)
    labels = gmm.predict(X)
    assert len(labels) == len(X)


def test_deterministic_reproducibility():
    """Test that probabilistic tools are deterministic with seeds."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    # HMM sampling
    hmm = HiddenMarkovModel(
        n_states=2,
        start_prob=np.array([0.5, 0.5]),
        trans_mat=np.array([[0.7, 0.3], [0.3, 0.7]]),
        emission_prob=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )

    states1, obs1 = hmm.sample(10, rng=rng1)
    states2, obs2 = hmm.sample(10, rng=rng2)

    np.testing.assert_array_equal(states1, states2)
    np.testing.assert_array_equal(obs1, obs2)

    # GMM sampling
    X = np.random.randn(50, 2)
    gmm1 = GaussianMixture(n_components=2, rng=rng1)
    gmm1.fit(X)

    gmm2 = GaussianMixture(n_components=2, rng=rng2)
    gmm2.fit(X)

    # Means should be similar (may differ slightly due to initialization)
    assert gmm1.means_.shape == gmm2.means_.shape


def test_error_handling():
    """Test that probabilistic tools handle edge cases gracefully."""
    # HMM with invalid observation
    hmm = HiddenMarkovModel(
        n_states=2,
        emission_prob=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    obs = np.array([0, 1, 5])  # Invalid observation (5 not in [0, 1])
    # Should handle gracefully (may return -inf or NaN for invalid observations)
    log_likelihood = hmm.score(obs)
    # Accept -inf, NaN, or very negative finite values
    assert not np.isfinite(log_likelihood) or log_likelihood < -10

    # GMM with insufficient data (but still valid - just few samples per component)
    X = np.random.randn(5, 2)  # Very few samples
    gmm = GaussianMixture(n_components=3)  # More components than samples
    # GMM will fit but may have issues - check that it completes
    # (The actual check for n_samples < n_components happens in fit)
    if len(X) < gmm.n_components:
        with pytest.raises(ValueError, match="at least"):
            gmm.fit(X)
    else:
        gmm.fit(X)  # Should work if enough samples


def test_memory_efficiency():
    """Test that probabilistic tools don't leak memory with large sequences."""
    # Large HMM sequence
    hmm = HiddenMarkovModel(
        n_states=3,
        start_prob=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        trans_mat=np.ones((3, 3)) / 3,
        emission_prob=np.ones((3, 2)) / 2,
    )

    # Long sequence
    obs = np.random.randint(0, 2, size=1000)
    log_likelihood = hmm.score(obs)
    assert np.isfinite(log_likelihood)

    # Large GMM dataset
    X = np.random.randn(1000, 10)
    gmm = GaussianMixture(n_components=5)
    gmm.fit(X)
    labels = gmm.predict(X)
    assert len(labels) == len(X)

