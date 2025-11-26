"""Tests for RL utility functions."""

import numpy as np
import pytest

from qconduit.rl.envs import ChainMDP
from qconduit.rl.utils import (
    compute_advantage,
    constant_schedule,
    discounted_returns,
    epsilon_greedy_action,
    evaluate_policy,
    greedy_policy_from_Q,
    linear_decay_schedule,
    seed_rng,
    softmax,
)


class TestSeedRNG:
    """Tests for RNG seeding."""

    def test_seed_rng(self):
        """Test RNG seeding."""
        rng1 = seed_rng(42)
        rng2 = seed_rng(42)

        # Same seed should produce same sequence
        val1 = rng1.integers(100)
        val2 = rng2.integers(100)
        assert val1 == val2

    def test_default_seed(self):
        """Test default seed."""
        rng = seed_rng(None)
        assert rng is not None


class TestSoftmax:
    """Tests for softmax function."""

    def test_softmax_basic(self):
        """Test basic softmax computation."""
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)

        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert probs[2] > probs[1] > probs[0]  # Higher logit -> higher prob

    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        # Large logits that would overflow without numerical trick
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)

        assert np.allclose(probs.sum(), 1.0)
        assert np.all(np.isfinite(probs))

    def test_softmax_single_value(self):
        """Test softmax with single value."""
        logits = np.array([5.0])
        probs = softmax(logits)
        assert np.allclose(probs, [1.0])


class TestEpsilonGreedyAction:
    """Tests for epsilon-greedy action selection."""

    def test_greedy_action(self):
        """Test greedy action selection (epsilon=0)."""
        rng = seed_rng(0)
        Q = np.array([0.1, 0.9, 0.3])
        action = epsilon_greedy_action(Q, epsilon=0.0, rng=rng)
        assert action == 1  # argmax

    def test_random_action(self):
        """Test random action selection (epsilon=1)."""
        rng = seed_rng(0)
        Q = np.array([0.1, 0.9, 0.3])
        action = epsilon_greedy_action(Q, epsilon=1.0, rng=rng)
        assert action in [0, 1, 2]

    def test_tie_breaking(self):
        """Test deterministic tie-breaking."""
        rng = seed_rng(0)
        Q = np.array([0.5, 0.5, 0.3])
        action = epsilon_greedy_action(Q, epsilon=0.0, rng=rng)
        assert action == 0  # First argmax

    def test_invalid_epsilon(self):
        """Test validation of epsilon."""
        rng = seed_rng(0)
        Q = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="epsilon must be in"):
            epsilon_greedy_action(Q, epsilon=1.5, rng=rng)


class TestDiscountedReturns:
    """Tests for discounted returns computation."""

    def test_discounted_returns(self):
        """Test discounted returns computation."""
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.9
        returns = discounted_returns(rewards, gamma)

        # G_0 = r_0 + gamma*r_1 + gamma^2*r_2
        expected_0 = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
        assert np.allclose(returns[0], expected_0)

        # G_1 = r_1 + gamma*r_2
        expected_1 = 2.0 + 0.9 * 3.0
        assert np.allclose(returns[1], expected_1)

        # G_2 = r_2
        assert np.allclose(returns[2], 3.0)

    def test_single_reward(self):
        """Test with single reward."""
        rewards = [5.0]
        returns = discounted_returns(rewards, gamma=0.9)
        assert np.allclose(returns, [5.0])

    def test_zero_gamma(self):
        """Test with gamma=0 (only immediate reward)."""
        rewards = [1.0, 2.0, 3.0]
        returns = discounted_returns(rewards, gamma=0.0)
        assert np.allclose(returns, rewards)

    def test_invalid_gamma(self):
        """Test validation of gamma."""
        with pytest.raises(ValueError, match="gamma must be in"):
            discounted_returns([1.0, 2.0], gamma=1.5)


class TestComputeAdvantage:
    """Tests for advantage computation."""

    def test_compute_advantage(self):
        """Test advantage computation."""
        returns = np.array([10.0, 5.0, 2.0])
        baselines = np.array([8.0, 4.0, 1.0])
        advantages = compute_advantage(returns, baselines)

        expected = np.array([2.0, 1.0, 1.0])
        assert np.allclose(advantages, expected)

    def test_shape_mismatch(self):
        """Test shape validation."""
        returns = np.array([10.0, 5.0])
        baselines = np.array([8.0, 4.0, 1.0])
        with pytest.raises(ValueError, match="must have same shape"):
            compute_advantage(returns, baselines)


class TestEvaluatePolicy:
    """Tests for policy evaluation."""

    def test_evaluate_policy(self):
        """Test policy evaluation."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        def policy(state):
            return 1  # Always go right

        avg_return, std_return = evaluate_policy(env, policy, num_episodes=10, rng=rng)

        assert avg_return >= 0.0
        assert std_return >= 0.0


class TestGreedyPolicyFromQ:
    """Tests for greedy policy construction."""

    def test_greedy_policy(self):
        """Test greedy policy from Q-table."""
        Q = np.array([[0.1, 0.9], [0.8, 0.2]])
        policy = greedy_policy_from_Q(Q)

        assert policy(0) == 1  # argmax of [0.1, 0.9]
        assert policy(1) == 0  # argmax of [0.8, 0.2]

    def test_tie_breaking(self):
        """Test deterministic tie-breaking."""
        Q = np.array([[0.5, 0.5], [0.3, 0.7]])
        policy = greedy_policy_from_Q(Q)

        assert policy(0) == 0  # First argmax

    def test_invalid_shape(self):
        """Test validation of Q shape."""
        Q = np.array([0.1, 0.9])  # 1D instead of 2D
        with pytest.raises(ValueError, match="Q must be 2D"):
            greedy_policy_from_Q(Q)


class TestSchedules:
    """Tests for schedule functions."""

    def test_constant_schedule(self):
        """Test constant schedule."""
        schedule = constant_schedule(0.1)
        assert schedule(0) == 0.1
        assert schedule(100) == 0.1

    def test_linear_decay_schedule(self):
        """Test linear decay schedule."""
        schedule = linear_decay_schedule(start=1.0, end=0.0, total_steps=10)

        assert schedule(0) == 1.0
        assert np.allclose(schedule(5), 0.5)
        assert schedule(10) == 0.0
        assert schedule(20) == 0.0  # Beyond total_steps

    def test_linear_decay_invalid(self):
        """Test validation of linear decay."""
        with pytest.raises(ValueError, match="total_steps must be > 0"):
            linear_decay_schedule(1.0, 0.0, 0)

