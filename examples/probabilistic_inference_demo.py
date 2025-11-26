"""Example: Probabilistic Inference Tools with Quantum Conduit

Demonstrates HMM, GMM, and particle filters for quantum measurement analysis.
"""

import numpy as np
import qconduit as qc
from qconduit import (
    GaussianMixture,
    HiddenMarkovModel,
    bootstrap_particle_filter,
    StateSpaceModel,
)


def example_hmm_quantum_measurements():
    """Example: Using HMM to decode quantum measurement sequences."""
    print("=" * 60)
    print("Example 1: HMM for Quantum Measurement Sequences")
    print("=" * 60)

    # Model quantum system with two hidden states (e.g., ground/excited)
    # and binary measurement outcomes
    hmm = HiddenMarkovModel(
        n_states=2,
        start_prob=np.array([0.6, 0.4]),  # Initial state distribution
        trans_mat=np.array([[0.7, 0.3], [0.4, 0.6]]),  # State transitions
        emission_prob=np.array([[0.9, 0.1], [0.2, 0.8]]),  # Measurement probabilities
    )

    # Simulate measurement sequence
    rng = np.random.default_rng(42)
    true_states, measurements = hmm.sample(20, rng=rng)

    print(f"True states:     {true_states}")
    print(f"Measurements:    {measurements}")

    # Decode most likely state sequence (Viterbi)
    decoded_states = hmm.predict(measurements)
    print(f"Decoded states:  {decoded_states}")

    # Compute log-likelihood
    log_likelihood = hmm.score(measurements)
    print(f"Log-likelihood: {log_likelihood:.4f}")

    # Forward-backward: posterior state probabilities
    gamma, _ = hmm.forward_backward(measurements)
    print(f"\nPosterior probabilities (first 5 time steps):")
    for t in range(min(5, len(measurements))):
        print(f"  t={t}: P(state=0)={gamma[t,0]:.3f}, P(state=1)={gamma[t,1]:.3f}")

    print()


def example_gmm_quantum_data_clustering():
    """Example: Using GMM to cluster quantum expectation values."""
    print("=" * 60)
    print("Example 2: GMM for Clustering Quantum Expectation Values")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate expectation values from different quantum states
    # Cluster 1: Low expectation values
    cluster1 = rng.normal([-1.5, -1.5], 0.3, size=(100, 2))
    # Cluster 2: High expectation values
    cluster2 = rng.normal([1.5, 1.5], 0.3, size=(100, 2))

    X = np.vstack([cluster1, cluster2])

    # Fit GMM
    gmm = GaussianMixture(n_components=2, rng=rng)
    gmm.fit(X)

    # Predict cluster assignments
    labels = gmm.predict(X)
    print(f"Clustered {len(X)} samples into 2 components")
    print(f"Component 0: {np.sum(labels == 0)} samples")
    print(f"Component 1: {np.sum(labels == 1)} samples")

    # Component parameters
    print(f"\nComponent means:")
    for k in range(2):
        print(f"  Component {k}: {gmm.means_[k]}")

    # Compute log-likelihood
    score = gmm.score(X)
    print(f"\nAverage log-likelihood: {score:.4f}")

    # Sample from fitted model
    samples = gmm.sample(10, rng=rng)
    print(f"\nSampled {len(samples)} points from fitted GMM")
    print()


def example_particle_filter_quantum_tracking():
    """Example: Using particle filter to track quantum system state."""
    print("=" * 60)
    print("Example 3: Particle Filter for Quantum State Tracking")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Model: Track quantum expectation value from noisy measurements
    def transition_sample(x_prev, rng):
        # State evolves with small random walk
        return x_prev + rng.normal(0, 0.1)

    def observation_loglik(y, x):
        # Noisy measurement: y ~ N(x, 0.5)
        return -0.5 * ((y - x) ** 2) / 0.5 - 0.5 * np.log(2 * np.pi * 0.5)

    def x0_sampler(rng):
        return np.array([rng.normal(0, 1)])

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    # Simulate noisy observations
    true_state = 0.5
    observations = []
    for _ in range(10):
        observation = true_state + rng.normal(0, np.sqrt(0.5))
        observations.append(observation)
    observations = np.array(observations)

    # Run particle filter
    result = bootstrap_particle_filter(
        model, observations, n_particles=200, rng=rng, resample_threshold=0.5
    )

    print(f"Observations: {observations[:5]}... (showing first 5)")
    print(f"Filtered estimates: {result['estimates'][:5, 0]}... (showing first 5)")
    print(f"Resampling occurred at: {np.where(result['resampled'])[0]}")

    # Log-evidence (marginal likelihood)
    total_log_evidence = np.sum(result["log_evidences"])
    print(f"Total log-evidence: {total_log_evidence:.4f}")

    print()


def example_utilities():
    """Example: Using probabilistic utilities."""
    print("=" * 60)
    print("Example 4: Probabilistic Utilities")
    print("=" * 60)

    # Log-sum-exp for stable probability computations
    log_probs = np.array([-10.0, -11.0, -12.0])
    log_sum = qc.logsumexp(log_probs)
    print(f"Log probabilities: {log_probs}")
    print(f"Log-sum-exp: {log_sum:.4f}")
    print(f"Sum of exp: {np.sum(np.exp(log_probs)):.6e}")

    # Normal PDF for parameter estimation
    x = np.array([0.5, 0.3])
    mean = np.array([0.5, 0.3])
    cov = np.eye(2) * 0.1
    pdf_val = qc.normal_pdf(x, mean, cov)
    print(f"\nMultivariate normal PDF: {pdf_val:.6f}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Probabilistic Inference Tools - Quantum Conduit Examples")
    print("=" * 60 + "\n")

    example_hmm_quantum_measurements()
    example_gmm_quantum_data_clustering()
    example_particle_filter_quantum_tracking()
    example_utilities()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

