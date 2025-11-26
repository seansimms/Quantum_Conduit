"""Particle filters (Sequential Monte Carlo) for state-space models.

Implements bootstrap particle filter with systematic resampling for
non-linear/non-Gaussian state-space models.

References:
    Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering
    and smoothing: fifteen years later. Handbook of nonlinear filtering, 12(656-704), 3.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .utils import effective_sample_size, normalize_log_weights, systematic_resample


@dataclass
class StateSpaceModel:
    """State-space model specification for particle filtering.

    Attributes:
        transition_sample: Function(x_{t-1}, rng) -> x_t, samples next state.
        transition_logpdf: Optional function(x_t, x_{t-1}) -> log p(x_t | x_{t-1}).
        observation_loglik: Function(y_t, x_t) -> log p(y_t | x_t).
        observation_sample: Optional function(x_t, rng) -> y_t, samples observation.
        x0_sampler: Function(rng) -> x_0, samples initial state.
    """

    transition_sample: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    transition_logpdf: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    observation_loglik: Callable[[np.ndarray, np.ndarray], float] = None
    observation_sample: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None
    x0_sampler: Callable[[np.random.Generator], np.ndarray] = None


def bootstrap_particle_filter(
    model: StateSpaceModel,
    observations: np.ndarray,
    n_particles: int = 100,
    rng: Optional[np.random.Generator] = None,
    resample_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Bootstrap particle filter for state-space models.

    Uses importance sampling with bootstrap proposal (transition_sampler) and
    weights via observation_loglik. Resamples when effective sample size drops
    below threshold.

    Args:
        model: StateSpaceModel specifying transition and observation models.
        observations: Observation sequence, shape (T, obs_dim) or (T,) if 1D.
        n_particles: Number of particles.
        rng: Random number generator. If None, uses default_rng(0).
        resample_threshold: Resample when ESS <= resample_threshold * n_particles.

    Returns:
        Dictionary with keys:
        - 'particles': Array shape (T, n_particles, state_dim) of particle states.
        - 'weights': Array shape (T, n_particles) of normalized weights.
        - 'log_evidences': Array shape (T,) of incremental log-evidence per time.
        - 'ancestors': Array shape (T, n_particles) of ancestor indices (after resampling).
        - 'estimates': Array shape (T, state_dim) of filtered state estimates (weighted mean).
        - 'resampled': Array shape (T,) of bool indicating if resampling occurred.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    observations = np.asarray(observations)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    T = len(observations)

    # Initialize particles from x0_sampler
    particles_t = np.array([model.x0_sampler(rng) for _ in range(n_particles)])
    state_dim = particles_t.shape[1] if particles_t.ndim > 1 else 1
    if particles_t.ndim == 1:
        particles_t = particles_t.reshape(-1, 1)

    # Storage
    particles = np.zeros((T, n_particles, state_dim))
    weights = np.zeros((T, n_particles))
    log_evidences = np.zeros(T)
    ancestors = np.zeros((T, n_particles), dtype=int)
    estimates = np.zeros((T, state_dim))
    resampled = np.zeros(T, dtype=bool)

    # Initial weights: uniform (no observation at t=0, or use first obs)
    if T > 0:
        # Compute initial weights from first observation
        log_weights = np.array(
            [
                model.observation_loglik(observations[0], particles_t[i, :])
                for i in range(n_particles)
            ]
        )
        weights_t, log_evidence_0 = normalize_log_weights(log_weights)
        # Ensure weights_t is 1D
        weights_t = np.asarray(weights_t).flatten()
        log_evidences[0] = log_evidence_0
        weights[0, :] = weights_t
        estimates[0, :] = np.sum(weights_t[:, np.newaxis] * particles_t, axis=0)
        particles[0, :, :] = particles_t

    # Main loop: t = 1, ..., T-1
    for t in range(1, T):
        # Resample if needed
        ess = effective_sample_size(weights[t - 1, :])
        if ess <= resample_threshold * n_particles:
            # Systematic resampling
            ancestor_indices = systematic_resample(weights[t - 1, :], rng)
            particles_t = particles_t[ancestor_indices, :]
            ancestors[t, :] = ancestor_indices
            resampled[t] = True
            # Reset weights to uniform after resampling
            weights_t = np.ones(n_particles) / n_particles
        else:
            # No resampling: propagate current particles
            ancestors[t, :] = np.arange(n_particles)
            resampled[t] = False
            weights_t = weights[t - 1, :].copy()
            # Ensure 1D
            weights_t = np.asarray(weights_t).flatten()

        # Propagate: sample x_t from transition
        particles_new = np.zeros_like(particles_t)
        for i in range(n_particles):
            particles_new[i, :] = model.transition_sample(particles_t[i, :], rng)
        particles_t = particles_new

        # Weight: compute observation log-likelihoods
        log_weights = np.array(
            [
                model.observation_loglik(observations[t], particles_t[i, :])
                for i in range(n_particles)
            ]
        )
        # Ensure log_weights is 1D
        log_weights = np.asarray(log_weights).flatten()

        # If we resampled, weights were uniform; otherwise multiply by previous weights
        if not resampled[t]:
            # Multiply weights: log(w_new) = log(w_old) + log_likelihood
            # Ensure weights_t is 1D
            weights_t = np.asarray(weights_t).flatten()
            log_weights = np.log(weights_t + 1e-12) + log_weights

        # Normalize
        weights_t, log_evidence_t = normalize_log_weights(log_weights)
        # Ensure weights_t is 1D with correct length
        weights_t = np.asarray(weights_t).flatten()
        log_evidences[t] = log_evidence_t

        # Store
        weights[t, :] = weights_t
        particles[t, :, :] = particles_t
        estimates[t, :] = np.sum(weights_t[:, np.newaxis] * particles_t, axis=0)

    return {
        "particles": particles,
        "weights": weights,
        "log_evidences": log_evidences,
        "ancestors": ancestors,
        "estimates": estimates,
        "resampled": resampled,
    }

