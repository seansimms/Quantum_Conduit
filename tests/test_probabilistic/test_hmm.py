"""Tests for Hidden Markov Model."""

import numpy as np
import pytest

from qconduit.probabilistic.hmm import HiddenMarkovModel


def test_hmm_forward_simple():
    """Test forward algorithm on simple 2-state HMM."""
    # Simple HMM: 2 states, 2 observations
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)

    # Single observation
    obs = np.array([0])
    log_alpha, log_likelihood = hmm.forward(obs)

    # Check shape
    assert log_alpha.shape == (1, 2)
    # Check log-likelihood is reasonable
    assert np.isfinite(log_likelihood)
    assert log_likelihood <= 0  # Log probability

    # Two observations
    obs = np.array([0, 1])
    log_alpha, log_likelihood = hmm.forward(obs)
    assert log_alpha.shape == (2, 2)
    assert np.isfinite(log_likelihood)


def test_hmm_forward_backward_consistency():
    """Test forward-backward consistency."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)
    obs = np.array([0, 0, 1, 0, 1])

    gamma, xi = hmm.forward_backward(obs)

    # Gamma should sum to 1 at each time
    assert gamma.shape == (len(obs), 2)
    for t in range(len(obs)):
        assert np.sum(gamma[t, :]) == pytest.approx(1.0, rel=1e-6)

    # Xi should sum correctly (if T > 1)
    if xi is not None:
        for t in range(len(obs) - 1):
            assert np.sum(xi[t, :, :]) == pytest.approx(1.0, rel=1e-6)


def test_hmm_viterbi_simple():
    """Test Viterbi on simple sequence."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)

    # Short sequence where we can check manually
    obs = np.array([0, 0])  # Both observations favor state 0
    path, log_prob = hmm.viterbi(obs)

    assert len(path) == len(obs)
    assert np.all(path >= 0)
    assert np.all(path < 2)
    assert np.isfinite(log_prob)
    assert log_prob <= 0


def test_hmm_viterbi_vs_brute_force():
    """Compare Viterbi to brute-force for very short sequences."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)
    obs = np.array([0, 1])

    path_viterbi, log_prob_viterbi = hmm.viterbi(obs)

    # Brute force: try all paths
    best_path = None
    best_log_prob = -np.inf
    for s0 in range(2):
        for s1 in range(2):
            log_prob = (
                np.log(start[s0] + 1e-12)
                + np.log(emit[s0, obs[0]] + 1e-12)
                + np.log(trans[s0, s1] + 1e-12)
                + np.log(emit[s1, obs[1]] + 1e-12)
            )
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_path = np.array([s0, s1])

    # Viterbi should match
    assert np.array_equal(path_viterbi, best_path)
    assert log_prob_viterbi == pytest.approx(best_log_prob, rel=1e-6)


def test_hmm_baum_welch_monotonic():
    """Test Baum-Welch increases likelihood monotonically."""
    # True model
    true_start = np.array([0.6, 0.4])
    true_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    true_hmm = HiddenMarkovModel(2, start_prob=true_start, trans_mat=true_trans, emission_prob=true_emit)

    # Generate sequences
    rng = np.random.default_rng(42)
    sequences = []
    for _ in range(5):
        _, obs = true_hmm.sample(20, rng=rng)
        sequences.append(obs)

    # Initialize HMM randomly
    hmm = HiddenMarkovModel(n_states=2)
    result = hmm.baum_welch(sequences, n_iter=10, rng=np.random.default_rng(0))

    # Check likelihood increases
    log_likelihoods = result["log_likelihoods"]
    for i in range(1, len(log_likelihoods)):
        assert log_likelihoods[i] >= log_likelihoods[i - 1] - 1e-10  # Allow tiny numerical errors


def test_hmm_baum_welch_parameter_recovery():
    """Test Baum-Welch recovers parameters approximately."""
    # True model
    true_start = np.array([0.6, 0.4])
    true_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    true_hmm = HiddenMarkovModel(2, start_prob=true_start, trans_mat=true_trans, emission_prob=true_emit)

    # Generate many sequences
    rng = np.random.default_rng(42)
    sequences = []
    for _ in range(20):
        _, obs = true_hmm.sample(50, rng=rng)
        sequences.append(obs)

    # Fit
    hmm = HiddenMarkovModel(n_states=2)
    hmm.baum_welch(sequences, n_iter=50, rng=np.random.default_rng(0))

    # Check approximate recovery (within tolerance)
    # Note: exact recovery is not guaranteed, but should be close
    assert np.allclose(hmm.start_prob, true_start, atol=0.2)
    assert np.allclose(hmm.trans_mat, true_trans, atol=0.2)
    assert np.allclose(hmm.emission_prob, true_emit, atol=0.2)


def test_hmm_sample():
    """Test HMM sampling."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)

    rng = np.random.default_rng(42)
    states, obs = hmm.sample(10, rng=rng)

    assert len(states) == 10
    assert len(obs) == 10
    assert np.all(states >= 0)
    assert np.all(states < 2)
    assert np.all(obs >= 0)
    assert np.all(obs < 2)


def test_hmm_score():
    """Test log-likelihood computation."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)
    obs = np.array([0, 1, 0])

    log_likelihood = hmm.score(obs)
    assert np.isfinite(log_likelihood)
    assert log_likelihood <= 0


def test_hmm_predict():
    """Test state prediction (Viterbi)."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)
    obs = np.array([0, 0, 1])

    path = hmm.predict(obs)
    assert len(path) == len(obs)
    assert np.all(path >= 0)
    assert np.all(path < 2)


def test_hmm_single_state():
    """Test edge case: single-state HMM."""
    hmm = HiddenMarkovModel(n_states=1)
    obs = np.array([0, 1, 0])

    log_alpha, log_likelihood = hmm.forward(obs)
    assert log_alpha.shape == (3, 1)
    assert np.isfinite(log_likelihood)

    path, _ = hmm.viterbi(obs)
    assert len(path) == 3
    assert np.all(path == 0)


def test_hmm_single_observation():
    """Test edge case: single observation."""
    start = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit = np.array([[0.9, 0.1], [0.2, 0.8]])

    hmm = HiddenMarkovModel(2, start_prob=start, trans_mat=trans, emission_prob=emit)
    obs = np.array([0])

    log_alpha, log_likelihood = hmm.forward(obs)
    assert log_alpha.shape == (1, 2)

    gamma, xi = hmm.forward_backward(obs)
    assert gamma.shape == (1, 2)
    assert xi is None  # No transitions for single observation

    path, _ = hmm.viterbi(obs)
    assert len(path) == 1

