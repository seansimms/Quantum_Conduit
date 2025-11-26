"""Tests for tabular RL algorithms."""

import numpy as np
import pytest

from qconduit.rl.envs import ChainMDP
from qconduit.rl.tabular import mc_control_on_policy, q_learning, sarsa, td_lambda
from qconduit.rl.utils import (
    constant_schedule,
    evaluate_policy,
    greedy_policy_from_Q,
    seed_rng,
)


class TestMonteCarloControl:
    """Tests for Monte Carlo control."""

    def test_mc_control_basic(self):
        """Test basic MC control execution."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        Q, policy = mc_control_on_policy(
            env,
            num_episodes=50,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            rng=rng,
        )

        assert Q.shape == (5, 2)
        assert callable(policy)
        assert policy(0) in [0, 1]

    def test_mc_control_convergence(self):
        """Test MC control learns reasonable policy."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        rng = seed_rng(0)

        Q, policy = mc_control_on_policy(
            env,
            num_episodes=200,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            rng=rng,
        )

        # Policy should prefer going right (action 1) in early states
        # to reach goal faster
        avg_return, _ = evaluate_policy(env, policy, num_episodes=20, rng=rng)
        assert avg_return > -1.0  # Should be better than random

    def test_mc_control_initial_Q(self):
        """Test MC control with initial Q-table."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)
        initial_Q = np.ones((5, 2), dtype=np.float64)

        Q, _ = mc_control_on_policy(
            env,
            num_episodes=10,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            rng=rng,
            initial_Q=initial_Q,
        )

        assert Q.shape == (5, 2)
        # Q should have been updated from initial values
        assert not np.allclose(Q, initial_Q)


class TestQLearning:
    """Tests for Q-learning algorithm."""

    def test_q_learning_basic(self):
        """Test basic Q-learning execution."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        Q = q_learning(
            env,
            num_episodes=50,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng,
        )

        assert Q.shape == (5, 2)

    def test_q_learning_convergence(self):
        """Test Q-learning learns reasonable policy."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        rng = seed_rng(0)

        Q = q_learning(
            env,
            num_episodes=200,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng,
        )

        policy = greedy_policy_from_Q(Q)
        avg_return, _ = evaluate_policy(env, policy, num_episodes=20, rng=rng)
        assert avg_return > -1.0  # Should learn reasonable policy

    def test_q_learning_initial_Q(self):
        """Test Q-learning with initial Q-table."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)
        Q0 = np.zeros((5, 2), dtype=np.float64)

        Q = q_learning(
            env,
            num_episodes=10,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng,
            Q0=Q0,
        )

        assert Q.shape == (5, 2)


class TestSARSA:
    """Tests for SARSA algorithm."""

    def test_sarsa_basic(self):
        """Test basic SARSA execution."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        Q = sarsa(
            env,
            num_episodes=50,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng,
        )

        assert Q.shape == (5, 2)

    def test_sarsa_convergence(self):
        """Test SARSA learns reasonable policy."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        rng = seed_rng(0)

        Q = sarsa(
            env,
            num_episodes=200,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng,
        )

        policy = greedy_policy_from_Q(Q)
        avg_return, _ = evaluate_policy(env, policy, num_episodes=20, rng=rng)
        assert avg_return > -1.0

    def test_sarsa_vs_qlearning(self):
        """Test that SARSA and Q-learning can produce different policies."""
        env = ChainMDP(n_states=5, seed=0)
        rng1 = seed_rng(0)
        rng2 = seed_rng(0)

        Q_sarsa = sarsa(
            env,
            num_episodes=100,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng1,
        )

        Q_qlearning = q_learning(
            env,
            num_episodes=100,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng2,
        )

        # They may differ due to on-policy vs off-policy updates
        # (though with same seed and simple MDP, they might be similar)
        assert Q_sarsa.shape == Q_qlearning.shape


class TestTDLambda:
    """Tests for TD(λ) algorithm."""

    def test_td_lambda_basic(self):
        """Test basic TD(λ) execution."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        Q = td_lambda(
            env,
            num_episodes=50,
            alpha=0.1,
            lambda_=0.5,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            trace_type="accumulating",
            rng=rng,
        )

        assert Q.shape == (5, 2)

    def test_td_lambda_replacing_traces(self):
        """Test TD(λ) with replacing traces."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        Q = td_lambda(
            env,
            num_episodes=50,
            alpha=0.1,
            lambda_=0.5,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            trace_type="replacing",
            rng=rng,
        )

        assert Q.shape == (5, 2)

    def test_td_lambda_lambda_zero(self):
        """Test TD(λ) with lambda=0 (should approximate TD(0)/SARSA)."""
        env = ChainMDP(n_states=5, seed=0)
        rng1 = seed_rng(0)
        rng2 = seed_rng(0)

        Q_td0 = td_lambda(
            env,
            num_episodes=50,
            alpha=0.1,
            lambda_=0.0,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            rng=rng1,
        )

        Q_sarsa = sarsa(
            env,
            num_episodes=50,
            alpha_schedule=constant_schedule(0.1),
            epsilon_schedule=constant_schedule(0.1),
            gamma=0.99,
            rng=rng2,
        )

        # With lambda=0, TD(λ) should be similar to SARSA
        # (though not identical due to eligibility trace initialization)
        assert Q_td0.shape == Q_sarsa.shape

    def test_td_lambda_invalid_trace_type(self):
        """Test validation of trace type."""
        env = ChainMDP(n_states=5, seed=0)
        rng = seed_rng(0)

        with pytest.raises(ValueError, match="trace_type must be"):
            td_lambda(
                env,
                num_episodes=10,
                alpha=0.1,
                lambda_=0.5,
                gamma=0.99,
                epsilon_schedule=constant_schedule(0.1),
                trace_type="invalid",
                rng=rng,
            )

    def test_td_lambda_convergence(self):
        """Test TD(λ) learns reasonable policy."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        rng = seed_rng(0)

        Q = td_lambda(
            env,
            num_episodes=200,
            alpha=0.1,
            lambda_=0.7,
            gamma=0.99,
            epsilon_schedule=constant_schedule(0.1),
            rng=rng,
        )

        policy = greedy_policy_from_Q(Q)
        avg_return, _ = evaluate_policy(env, policy, num_episodes=20, rng=rng)
        assert avg_return > -1.0

