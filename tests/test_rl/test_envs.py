"""Tests for RL environments."""

import numpy as np
import pytest

from qconduit.rl.envs import Bandit, ChainMDP, GridWorldTiny


class TestChainMDP:
    """Tests for ChainMDP environment."""

    def test_reset(self):
        """Test environment reset returns initial state."""
        env = ChainMDP(n_states=5, seed=0)
        state = env.reset()
        assert state == 0

    def test_deterministic_transitions(self):
        """Test deterministic transitions with fixed seed."""
        env = ChainMDP(n_states=5, p_left=0.0, p_right=1.0, seed=42)
        state = env.reset()
        assert state == 0

        # Go right deterministically
        state, reward, done, _ = env.step(1)
        assert state == 1
        assert not done

        state, reward, done, _ = env.step(1)
        assert state == 2
        assert not done

    def test_terminal_state(self):
        """Test terminal state behavior."""
        env = ChainMDP(n_states=5, seed=0)
        state = env.reset()

        # Move to terminal state
        for _ in range(4):
            state, reward, done, _ = env.step(1)

        assert state == 4  # Terminal state
        assert done
        assert reward == env.reward_goal

        # Terminal state should not transition
        state2, reward2, done2, _ = env.step(1)
        assert state2 == 4
        assert done2
        assert reward2 == 0.0

    def test_rewards(self):
        """Test reward structure."""
        env = ChainMDP(n_states=5, reward_goal=1.0, reward_step=-0.01, seed=0)
        env.reset()

        # Non-terminal step
        _, reward, done, _ = env.step(1)
        assert reward == -0.01
        assert not done

    def test_seed(self):
        """Test seed setting."""
        env = ChainMDP(n_states=5, seed=0)
        env.seed(42)
        state1 = env.reset()
        _, _, _, _ = env.step(1)

        env2 = ChainMDP(n_states=5, seed=42)
        state2 = env2.reset()
        _, _, _, _ = env2.step(1)

        # With same seed, should get same transitions
        assert state1 == state2

    def test_invalid_params(self):
        """Test validation of parameters."""
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            ChainMDP(n_states=1)

        with pytest.raises(ValueError, match="p_left must be in"):
            ChainMDP(n_states=5, p_left=-0.1)

        with pytest.raises(ValueError, match="action must be 0 or 1"):
            env = ChainMDP(n_states=5, seed=0)
            env.reset()
            env.step(2)


class TestGridWorldTiny:
    """Tests for GridWorldTiny environment."""

    def test_reset(self):
        """Test environment reset."""
        env = GridWorldTiny(size=3, seed=0)
        state = env.reset()
        assert state == 0  # Top-left

    def test_moves(self):
        """Test movement actions."""
        env = GridWorldTiny(size=3, seed=0)
        state = env.reset()

        # Move right
        state, _, done, _ = env.step(1)
        assert state == 1
        assert not done

        # Move down
        state, _, done, _ = env.step(2)
        assert state == 4  # Middle row, first col
        assert not done

    def test_boundaries(self):
        """Test boundary handling."""
        env = GridWorldTiny(size=3, seed=0)
        state = env.reset()

        # Try to move left from left edge
        state, _, _, _ = env.step(3)
        assert state == 0  # Should stay at 0

        # Try to move up from top edge
        state, _, _, _ = env.step(0)
        assert state == 0  # Should stay at 0

    def test_terminal_state(self):
        """Test terminal state detection."""
        terminal = np.zeros((3, 3), dtype=bool)
        terminal[2, 2] = True  # Bottom-right
        env = GridWorldTiny(size=3, terminal_mask=terminal, seed=0)
        state = env.reset()

        # Move to terminal
        for _ in range(2):
            state, _, done, _ = env.step(1)  # Right
        for _ in range(2):
            state, _, done, _ = env.step(2)  # Down

        assert state == 8  # Bottom-right
        assert done
        assert env.reward_goal > 0

    def test_seed(self):
        """Test seed setting."""
        env = GridWorldTiny(size=3, seed=0)
        env.seed(42)
        state1 = env.reset()

        env2 = GridWorldTiny(size=3, seed=42)
        state2 = env2.reset()

        assert state1 == state2

    def test_invalid_params(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="size must be >= 2"):
            GridWorldTiny(size=1)

        terminal = np.zeros((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="terminal_mask shape"):
            GridWorldTiny(size=3, terminal_mask=terminal)


class TestBandit:
    """Tests for Bandit environment."""

    def test_reset(self):
        """Test environment reset."""
        env = Bandit(k=3, seed=0)
        state = env.reset()
        assert state == 0  # Single state

    def test_rewards(self):
        """Test reward structure."""
        env = Bandit(k=3, probs=[0.0, 0.0, 1.0], seed=0)
        env.reset()

        # Arm 2 should always give reward 1
        _, reward, done, _ = env.step(2)
        assert reward == 1.0
        assert done  # Bandit episodes terminate after one step

    def test_best_arm(self):
        """Test best arm identification."""
        env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        assert env.best_arm == 2  # Highest probability

    def test_seed(self):
        """Test seed setting."""
        env = Bandit(k=3, seed=0)
        env.seed(42)
        state1 = env.reset()

        env2 = Bandit(k=3, seed=42)
        state2 = env2.reset()

        assert state1 == state2

    def test_invalid_params(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            Bandit(k=0)

        with pytest.raises(ValueError, match="probs must have shape"):
            Bandit(k=3, probs=np.array([0.1, 0.5]))

        with pytest.raises(ValueError, match="probs must be in"):
            Bandit(k=3, probs=np.array([0.1, 1.5, 0.9]))

