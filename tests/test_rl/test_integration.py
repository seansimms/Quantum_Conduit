"""Integration tests for RL module production readiness."""

import numpy as np
import pytest

from qconduit.rl import (
    Bandit,
    ChainMDP,
    GridWorldTiny,
    TabularPolicy,
    compute_advantage,
    discounted_returns,
    epsilon_greedy_action,
    evaluate_policy,
    greedy_policy_from_Q,
    mc_control_on_policy,
    q_learning,
    reinforce,
    reinforce_with_baseline,
    sarsa,
    seed_rng,
    softmax,
    td_lambda,
)


class TestIntegration:
    """Integration tests for RL module."""

    def test_all_imports_work(self):
        """Test that all RL module exports can be imported."""
        # This test verifies the __init__.py exports are correct
        assert ChainMDP is not None
        assert GridWorldTiny is not None
        assert Bandit is not None
        assert q_learning is not None
        assert sarsa is not None
        assert mc_control_on_policy is not None
        assert td_lambda is not None
        assert reinforce is not None
        assert reinforce_with_baseline is not None
        assert seed_rng is not None
        assert softmax is not None
        assert epsilon_greedy_action is not None
        assert discounted_returns is not None
        assert compute_advantage is not None
        assert evaluate_policy is not None
        assert greedy_policy_from_Q is not None
        assert TabularPolicy is not None

    def test_end_to_end_q_learning(self):
        """Test complete Q-learning workflow."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        rng = seed_rng(0)

        # Train
        Q = q_learning(
            env,
            num_episodes=50,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng,
        )

        # Extract policy
        policy = greedy_policy_from_Q(Q)

        # Evaluate
        avg_return, std_return = evaluate_policy(env, policy, num_episodes=10, rng=rng)

        assert Q.shape == (5, 2)
        assert avg_return >= -1.0
        assert std_return >= 0.0

    def test_end_to_end_reinforce(self):
        """Test complete REINFORCE workflow."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        rng = seed_rng(0)

        # Train
        policy = reinforce(
            env,
            policy,
            num_episodes=50,
            gamma=1.0,
            alpha_schedule=lambda e: 0.1,
            rng=rng,
        )

        # Check policy is valid
        probs = policy.action_probs(0)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_utility_functions_work_together(self):
        """Test utility functions work correctly together."""
        rng = seed_rng(42)

        # Test softmax
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        assert np.allclose(probs.sum(), 1.0)

        # Test epsilon-greedy
        Q_row = np.array([0.1, 0.9, 0.3])
        action = epsilon_greedy_action(Q_row, epsilon=0.0, rng=rng)
        assert action == 1  # Greedy

        # Test discounted returns
        rewards = [1.0, 2.0, 3.0]
        returns = discounted_returns(rewards, gamma=0.9)
        assert len(returns) == 3
        assert returns[2] == 3.0  # Last return equals last reward

        # Test advantage
        baselines = np.array([0.5, 1.0, 1.5])
        advantages = compute_advantage(returns, baselines)
        assert len(advantages) == 3

    def test_deterministic_reproducibility(self):
        """Test that algorithms are deterministic with same seed."""
        env1 = ChainMDP(n_states=5, seed=42)
        env2 = ChainMDP(n_states=5, seed=42)
        rng1 = seed_rng(42)
        rng2 = seed_rng(42)

        Q1 = q_learning(
            env1,
            num_episodes=20,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng1,
        )

        Q2 = q_learning(
            env2,
            num_episodes=20,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng2,
        )

        # Should be identical with same seed
        np.testing.assert_array_almost_equal(Q1, Q2, decimal=10)

    def test_multiple_algorithms_same_env(self):
        """Test that multiple algorithms can work on the same environment."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        # Q-learning
        Q_qlearn = q_learning(
            env,
            num_episodes=20,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng,
        )

        # Reset RNG for SARSA
        rng = seed_rng(0)
        Q_sarsa = sarsa(
            env,
            num_episodes=20,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng,
        )

        assert Q_qlearn.shape == Q_sarsa.shape
        assert Q_qlearn.shape == (5, 2)

    def test_policy_gradient_variants(self):
        """Test both REINFORCE variants work."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        rng = seed_rng(0)

        # REINFORCE
        policy1 = TabularPolicy(n_states=1, n_actions=3, seed=0)
        policy1 = reinforce(
            env,
            policy1,
            num_episodes=20,
            gamma=1.0,
            alpha_schedule=lambda e: 0.1,
            rng=rng,
        )

        # REINFORCE with baseline
        rng = seed_rng(0)
        policy2 = TabularPolicy(n_states=1, n_actions=3, seed=0)
        policy2 = reinforce_with_baseline(
            env,
            policy2,
            num_episodes=20,
            gamma=1.0,
            alpha_schedule=lambda e: 0.1,
            baseline_type="running_mean",
            rng=rng,
        )

        # Both should produce valid policies
        probs1 = policy1.action_probs(0)
        probs2 = policy2.action_probs(0)

        assert np.allclose(probs1.sum(), 1.0)
        assert np.allclose(probs2.sum(), 1.0)

    def test_gridworld_integration(self):
        """Test GridWorld environment with algorithms."""
        import numpy as np

        terminal = np.zeros((3, 3), dtype=bool)
        terminal[2, 2] = True
        env = GridWorldTiny(size=3, terminal_mask=terminal, seed=0)
        rng = seed_rng(0)

        Q = q_learning(
            env,
            num_episodes=30,
            alpha_schedule=lambda e: 0.1,
            epsilon_schedule=lambda e: 0.1,
            gamma=0.99,
            rng=rng,
        )

        policy = greedy_policy_from_Q(Q)
        avg_return, _ = evaluate_policy(env, policy, num_episodes=5, rng=rng)

        assert Q.shape == (9, 4)  # 9 states, 4 actions
        assert avg_return >= -1.0

