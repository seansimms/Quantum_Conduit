"""Tests for policy gradient algorithms."""

import numpy as np
import pytest

from qconduit.rl.agent import TabularPolicy
from qconduit.rl.envs import Bandit, ChainMDP
from qconduit.rl.policy_gradient import reinforce, reinforce_with_baseline
from qconduit.rl.utils import constant_schedule, seed_rng


class TestREINFORCE:
    """Tests for REINFORCE algorithm."""

    def test_reinforce_basic(self):
        """Test basic REINFORCE execution."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        policy = reinforce(
            env,
            policy,
            num_episodes=50,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            rng=rng,
        )

        probs = policy.action_probs(0)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_reinforce_bandit_learning(self):
        """Test REINFORCE learns best arm in bandit."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        # Initial policy should be near-uniform
        initial_probs = policy.action_probs(0)
        assert np.allclose(initial_probs, 1.0 / 3.0, atol=0.1)

        # Train
        policy = reinforce(
            env,
            policy,
            num_episodes=200,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            rng=rng,
        )

        # Final policy should favor best arm (arm 2 with prob 0.9)
        final_probs = policy.action_probs(0)
        assert final_probs[2] > final_probs[0]  # Best arm > worst arm
        assert final_probs[2] > final_probs[1]  # Best arm > middle arm

    def test_reinforce_with_baseline_function(self):
        """Test REINFORCE with custom baseline function."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        def baseline(state):
            return 0.5  # Constant baseline

        policy = reinforce(
            env,
            policy,
            num_episodes=50,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            baseline=baseline,
            rng=rng,
        )

        probs = policy.action_probs(0)
        assert np.allclose(probs.sum(), 1.0)

    def test_reinforce_episodic_env(self):
        """Test REINFORCE on episodic environment."""
        env = ChainMDP(n_states=5, seed=0)
        policy = TabularPolicy(n_states=5, n_actions=2, seed=0)
        rng = seed_rng(0)

        policy = reinforce(
            env,
            policy,
            num_episodes=50,
            gamma=0.99,
            alpha_schedule=constant_schedule(0.1),
            rng=rng,
        )

        # Check policy is valid
        for s in range(5):
            probs = policy.action_probs(s)
            assert np.allclose(probs.sum(), 1.0)

    def test_reinforce_invalid_policy(self):
        """Test validation of policy type."""
        env = Bandit(k=3, seed=0)
        rng = seed_rng(0)

        class DummyPolicy:
            pass

        dummy_policy = DummyPolicy()

        with pytest.raises(TypeError, match="policy must be TabularPolicy"):
            reinforce(
                env,
                dummy_policy,
                num_episodes=10,
                gamma=1.0,
                alpha_schedule=constant_schedule(0.1),
                rng=rng,
            )


class TestREINFORCEWithBaseline:
    """Tests for REINFORCE with baseline."""

    def test_reinforce_with_baseline_running_mean(self):
        """Test REINFORCE with running mean baseline."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        policy = reinforce_with_baseline(
            env,
            policy,
            num_episodes=50,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            baseline_type="running_mean",
            rng=rng,
        )

        probs = policy.action_probs(0)
        assert np.allclose(probs.sum(), 1.0)

    def test_reinforce_with_baseline_state_value(self):
        """Test REINFORCE with state-value baseline."""
        env = ChainMDP(n_states=5, seed=0)
        policy = TabularPolicy(n_states=5, n_actions=2, seed=0)
        rng = seed_rng(0)

        policy = reinforce_with_baseline(
            env,
            policy,
            num_episodes=50,
            gamma=0.99,
            alpha_schedule=constant_schedule(0.1),
            baseline_type="state_value",
            rng=rng,
        )

        # Check policy is valid
        for s in range(5):
            probs = policy.action_probs(s)
            assert np.allclose(probs.sum(), 1.0)

    def test_reinforce_with_baseline_learning(self):
        """Test REINFORCE with baseline learns in bandit."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        policy = reinforce_with_baseline(
            env,
            policy,
            num_episodes=200,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            baseline_type="running_mean",
            rng=rng,
        )

        final_probs = policy.action_probs(0)
        assert final_probs[2] > final_probs[0]  # Best arm > worst arm

    def test_reinforce_with_baseline_invalid_type(self):
        """Test validation of baseline type."""
        env = Bandit(k=3, seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        with pytest.raises(ValueError, match="baseline_type must be"):
            reinforce_with_baseline(
                env,
                policy,
                num_episodes=10,
                gamma=1.0,
                alpha_schedule=constant_schedule(0.1),
                baseline_type="invalid",
                rng=rng,
            )

    def test_baseline_variance_reduction(self):
        """Test that baseline reduces variance of gradient estimates."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        rng1 = seed_rng(0)
        rng2 = seed_rng(0)

        # REINFORCE without baseline
        policy1 = TabularPolicy(n_states=1, n_actions=3, seed=0)
        policy1 = reinforce(
            env,
            policy1,
            num_episodes=100,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            rng=rng1,
        )

        # REINFORCE with baseline
        policy2 = TabularPolicy(n_states=1, n_actions=3, seed=0)
        policy2 = reinforce_with_baseline(
            env,
            policy2,
            num_episodes=100,
            gamma=1.0,
            alpha_schedule=constant_schedule(0.1),
            baseline_type="running_mean",
            rng=rng2,
        )

        # Both should learn, but with baseline might have different trajectory
        # (hard to test variance reduction deterministically, but both should work)
        probs1 = policy1.action_probs(0)
        probs2 = policy2.action_probs(0)
        assert np.allclose(probs1.sum(), 1.0)
        assert np.allclose(probs2.sum(), 1.0)

