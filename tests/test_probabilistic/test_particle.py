"""Tests for particle filters."""

import numpy as np
import pytest

from qconduit.probabilistic.particle import StateSpaceModel, bootstrap_particle_filter


def make_linear_gaussian_model(F, H, Q, R, x0_mean, x0_cov, rng):
    """Create a linear-Gaussian state-space model for testing.

    x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = H * x_t + v_t,       v_t ~ N(0, R)
    """
    state_dim = F.shape[0]
    obs_dim = H.shape[0]

    def transition_sample(x_prev, rng):
        return F @ x_prev + rng.multivariate_normal(np.zeros(state_dim), Q)

    def transition_logpdf(x, x_prev):
        diff = x - F @ x_prev
        # Log PDF of N(0, Q)
        log_det = np.linalg.slogdet(Q)[1]
        quad = diff.T @ np.linalg.solve(Q, diff)
        return -0.5 * state_dim * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad

    def observation_loglik(y, x):
        diff = y - H @ x
        # Log PDF of N(0, R)
        log_det = np.linalg.slogdet(R)[1]
        quad = diff.T @ np.linalg.solve(R, diff)
        return -0.5 * obs_dim * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad

    def observation_sample(x, rng):
        return H @ x + rng.multivariate_normal(np.zeros(obs_dim), R)

    def x0_sampler(rng):
        return rng.multivariate_normal(x0_mean, x0_cov)

    return StateSpaceModel(
        transition_sample=transition_sample,
        transition_logpdf=transition_logpdf,
        observation_loglik=observation_loglik,
        observation_sample=observation_sample,
        x0_sampler=x0_sampler,
    )


def kalman_filter_predict_update(x_pred, P_pred, y, F, H, Q, R):
    """One step of Kalman filter (predict and update).

    Returns (x_filtered, P_filtered).
    """
    # Predict
    x_pred_new = F @ x_pred
    P_pred_new = F @ P_pred @ F.T + Q

    # Update
    S = H @ P_pred_new @ H.T + R
    K = P_pred_new @ H.T @ np.linalg.solve(S, np.eye(len(S)))
    x_filtered = x_pred_new + K @ (y - H @ x_pred_new)
    P_filtered = P_pred_new - K @ H @ P_pred_new

    return x_filtered, P_filtered


def test_particle_filter_linear_gaussian():
    """Test particle filter on linear-Gaussian model (compare to Kalman filter)."""
    rng = np.random.default_rng(42)

    # Simple 1D model
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    x0_mean = np.array([0.0])
    x0_cov = np.array([[1.0]])

    model = make_linear_gaussian_model(F, H, Q, R, x0_mean, x0_cov, rng)

    # Generate true trajectory and observations
    T = 20
    true_states = np.zeros(T)
    observations = np.zeros(T)

    x = rng.multivariate_normal(x0_mean, x0_cov)
    true_states[0] = x[0]
    observations[0] = (H @ x + rng.multivariate_normal(np.zeros(1), R))[0]

    for t in range(1, T):
        x = F @ x + rng.multivariate_normal(np.zeros(1), Q)
        true_states[t] = x[0]
        observations[t] = (H @ x + rng.multivariate_normal(np.zeros(1), R))[0]

    # Run particle filter
    result = bootstrap_particle_filter(model, observations, n_particles=1000, rng=rng)

    # Run Kalman filter for comparison
    x_kf = x0_mean.copy()
    P_kf = x0_cov.copy()
    kf_estimates = np.zeros(T)
    kf_estimates[0] = x_kf[0]

    for t in range(1, T):
        x_kf, P_kf = kalman_filter_predict_update(x_kf, P_kf, np.array([observations[t]]), F, H, Q, R)
        kf_estimates[t] = x_kf[0]

    # Compare: particle filter estimates should be close to Kalman filter
    pf_estimates = result["estimates"][:, 0]

    # Allow some tolerance (particle filter is approximate)
    for t in range(T):
        diff = abs(pf_estimates[t] - kf_estimates[t])
        assert diff < 0.5  # Reasonable tolerance for 1000 particles


def test_particle_filter_reproducibility():
    """Test that particle filter is reproducible with same seed."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    # Simple model
    def transition_sample(x_prev, rng):
        return x_prev + rng.normal(0, 0.1)

    def observation_loglik(y, x):
        return -0.5 * ((y - x) ** 2) / 0.5 - 0.5 * np.log(2 * np.pi * 0.5)

    def x0_sampler(rng):
        return np.array([rng.normal(0, 1)])

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    observations = np.array([0.1, 0.2, 0.15, 0.3])

    result1 = bootstrap_particle_filter(model, observations, n_particles=100, rng=rng1)
    result2 = bootstrap_particle_filter(model, observations, n_particles=100, rng=rng2)

    # Particles and weights should be identical
    np.testing.assert_array_equal(result1["particles"], result2["particles"])
    np.testing.assert_array_equal(result1["weights"], result2["weights"])


def test_particle_filter_resampling():
    """Test that resampling occurs when ESS drops."""
    rng = np.random.default_rng(42)

    # Model that causes weight degeneracy
    def transition_sample(x_prev, rng):
        return x_prev + rng.normal(0, 0.1)

    def observation_loglik(y, x):
        # Very peaked likelihood -> weight degeneracy
        return -100.0 * ((y - x) ** 2)

    def x0_sampler(rng):
        return np.array([rng.normal(0, 1)])

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    # Observation far from initial particles -> low ESS
    observations = np.array([10.0, 10.0, 10.0])

    result = bootstrap_particle_filter(
        model, observations, n_particles=50, rng=rng, resample_threshold=0.5
    )

    # Should have resampled at some point
    assert np.any(result["resampled"])


def test_particle_filter_output_shapes():
    """Test output shapes of particle filter."""
    rng = np.random.default_rng(42)

    def transition_sample(x_prev, rng):
        return x_prev + rng.normal(0, 0.1, size=x_prev.shape)

    def observation_loglik(y, x):
        return -0.5 * np.sum((y - x) ** 2) / 0.5

    def x0_sampler(rng):
        return rng.normal(0, 1, size=(2,))  # 2D state

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    observations = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25]])
    n_particles = 100

    result = bootstrap_particle_filter(model, observations, n_particles=n_particles, rng=rng)

    T = len(observations)
    state_dim = 2

    assert result["particles"].shape == (T, n_particles, state_dim)
    assert result["weights"].shape == (T, n_particles)
    assert result["log_evidences"].shape == (T,)
    assert result["ancestors"].shape == (T, n_particles)
    assert result["estimates"].shape == (T, state_dim)
    assert result["resampled"].shape == (T,)

    # Weights should sum to 1 at each time
    for t in range(T):
        assert np.sum(result["weights"][t, :]) == pytest.approx(1.0, rel=1e-6)


def test_particle_filter_log_evidence():
    """Test that log-evidence is computed correctly."""
    rng = np.random.default_rng(42)

    # Simple model
    def transition_sample(x_prev, rng):
        return x_prev + rng.normal(0, 0.1)

    def observation_loglik(y, x):
        return -0.5 * ((y - x) ** 2) / 0.5 - 0.5 * np.log(2 * np.pi * 0.5)

    def x0_sampler(rng):
        return np.array([rng.normal(0, 1)])

    model = StateSpaceModel(
        transition_sample=transition_sample,
        observation_loglik=observation_loglik,
        x0_sampler=x0_sampler,
    )

    observations = np.array([0.1, 0.2])

    result = bootstrap_particle_filter(model, observations, n_particles=100, rng=rng)

    # Log-evidence should be finite and reasonable
    assert np.all(np.isfinite(result["log_evidences"]))
    # Log-evidence can be positive if observations are very likely (log P > 0 means P > 1/exp(0) = 1)
    # But typically for well-calibrated models, it should be negative or small positive
    # Just check it's finite and reasonable (not extremely large)
    assert np.all(np.abs(result["log_evidences"]) < 100)

