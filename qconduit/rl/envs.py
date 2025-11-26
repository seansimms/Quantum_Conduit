"""Simple reference MDP environments for testing and examples.

This module provides minimal, deterministic, gym-like MDPs suitable for
testing tabular and policy-gradient algorithms. All environments support
RNG seeding for reproducibility.
"""

from typing import Any, Optional, Tuple

import numpy as np

from .utils import seed_rng


class Env:
    """Minimal environment interface for tabular RL algorithms.

    States and actions are represented as small integers for compatibility
    with tabular methods.
    """

    observation_space: Any
    action_space: Any

    def reset(self) -> int:
        """Reset environment and return initial state.

        Returns:
            Initial state (integer).
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Apply action and return transition.

        Args:
            action: Action index (integer).

        Returns:
            Tuple of (next_state, reward, done, info).
            - next_state: Next state (integer)
            - reward: Reward received (float)
            - done: Whether episode terminated (bool)
            - info: Additional info dict
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed for environment.

        Args:
            seed: Random seed. If None, uses default seed.
        """
        raise NotImplementedError


class ChainMDP(Env):
    """Simple n-state chain MDP with terminal reward at one end.

    States are 0, 1, ..., n_states-1. State n_states-1 is terminal.
    Actions: 0 = left, 1 = right.

    Transitions can be deterministic or stochastic based on p_left/p_right.
    Default: deterministic (p_left=0.0, p_right=1.0).

    Args:
        n_states: Number of states (>= 2).
        p_left: Probability of moving left when action=0.
        p_right: Probability of moving right when action=1.
        reward_goal: Reward at terminal state.
        reward_step: Reward per non-terminal step (typically negative).
        seed: Random seed for transitions.

    Examples:
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> state = env.reset()
        >>> state
        0
        >>> state, reward, done, _ = env.step(1)  # Go right
        >>> state, done
        (1, False)
    """

    def __init__(
        self,
        n_states: int = 5,
        p_left: float = 0.0,
        p_right: float = 1.0,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        seed: Optional[int] = None,
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if not (0 <= p_left <= 1):
            raise ValueError(f"p_left must be in [0, 1], got {p_left}")
        if not (0 <= p_right <= 1):
            raise ValueError(f"p_right must be in [0, 1], got {p_right}")

        self.n_states = n_states
        self.p_left = p_left
        self.p_right = p_right
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.rng = seed_rng(seed)
        self.state: Optional[int] = None

        # Simple action space: 0=left, 1=right
        self.action_space = 2
        self.observation_space = n_states

    def reset(self) -> int:
        """Reset to initial state 0."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take step in chain MDP."""
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if action not in [0, 1]:
            raise ValueError(f"action must be 0 or 1, got {action}")

        # Terminal state: no transition
        if self.state == self.n_states - 1:
            return self.state, 0.0, True, {}

        # Transition probabilities
        if action == 0:  # Left
            if self.rng.random() < self.p_left:
                next_state = max(0, self.state - 1)
            else:
                next_state = min(self.n_states - 1, self.state + 1)
        else:  # Right
            if self.rng.random() < self.p_right:
                next_state = min(self.n_states - 1, self.state + 1)
            else:
                next_state = max(0, self.state - 1)

        # Rewards
        if next_state == self.n_states - 1:
            reward = self.reward_goal
            done = True
        else:
            reward = self.reward_step
            done = False

        self.state = next_state
        return next_state, reward, done, {}

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        self.rng = seed_rng(seed)


class GridWorldTiny(Env):
    """Small gridworld environment with deterministic moves and terminal cells.

    Grid is size x size. States are flattened: state = row * size + col.
    Actions: 0=up, 1=right, 2=down, 3=left.

    Terminal states are specified by terminal_mask (boolean array).
    Rewards: reward_goal for terminal states, reward_step otherwise.

    Args:
        size: Grid size (e.g., 3 for 3x3 grid).
        terminal_mask: Boolean array of shape (size, size) marking terminal cells.
        reward_goal: Reward at terminal states.
        reward_step: Reward per non-terminal step.
        seed: Random seed (for future stochasticity).

    Examples:
        >>> terminal = np.zeros((3, 3), dtype=bool)
        >>> terminal[2, 2] = True  # Bottom-right is goal
        >>> env = GridWorldTiny(size=3, terminal_mask=terminal, seed=0)
        >>> state = env.reset()
        >>> state  # Start at (0, 0) = state 0
        0
    """

    def __init__(
        self,
        size: int = 3,
        terminal_mask: Optional[np.ndarray] = None,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        seed: Optional[int] = None,
    ):
        if size < 2:
            raise ValueError(f"size must be >= 2, got {size}")

        self.size = size
        self.n_states = size * size
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.rng = seed_rng(seed)

        if terminal_mask is None:
            # Default: bottom-right corner is terminal
            terminal_mask = np.zeros((size, size), dtype=bool)
            terminal_mask[size - 1, size - 1] = True

        if terminal_mask.shape != (size, size):
            raise ValueError(
                f"terminal_mask shape must be ({size}, {size}), "
                f"got {terminal_mask.shape}"
            )
        self.terminal_mask = terminal_mask
        self.terminal_states = np.where(terminal_mask.flatten())[0]

        self.action_space = 4  # up, right, down, left
        self.observation_space = self.n_states

        self.state: Optional[int] = None

    def reset(self) -> int:
        """Reset to initial state (top-left: state 0)."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take step in gridworld."""
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if action not in [0, 1, 2, 3]:
            raise ValueError(f"action must be in [0, 1, 2, 3], got {action}")

        # Terminal state: no transition
        if self.state in self.terminal_states:
            return self.state, 0.0, True, {}

        # Convert state to (row, col)
        row = self.state // self.size
        col = self.state % self.size

        # Apply action (with boundaries)
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)

        next_state = row * self.size + col

        # Check if terminal
        done = next_state in self.terminal_states
        reward = self.reward_goal if done else self.reward_step

        self.state = next_state
        return next_state, reward, done, {}

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        self.rng = seed_rng(seed)


class Bandit(Env):
    """k-armed Bernoulli bandit environment.

    Each arm has a success probability. Pulling an arm returns reward 1
    with that probability, 0 otherwise.

    Useful for testing policy gradient methods on simple multi-armed bandit.

    Args:
        k: Number of arms.
        probs: Success probabilities for each arm, shape (k,). If None,
            uses random probabilities.
        seed: Random seed for reward sampling.

    Examples:
        >>> env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        >>> state = env.reset()
        >>> state  # Bandit has single state
        0
        >>> _, reward, done, _ = env.step(2)  # Pull arm 2 (best)
        >>> reward in [0, 1]
        True
    """

    def __init__(
        self,
        k: int = 3,
        probs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        self.rng = seed_rng(seed)

        if probs is None:
            # Random probabilities
            probs = self.rng.random(k)
        else:
            probs = np.asarray(probs, dtype=np.float64)
            if probs.shape != (k,):
                raise ValueError(f"probs must have shape ({k},), got {probs.shape}")
            if not np.all((probs >= 0) & (probs <= 1)):
                raise ValueError("probs must be in [0, 1]")

        self.probs = probs
        self.best_arm = int(np.argmax(probs))

        # Bandit has single state
        self.action_space = k
        self.observation_space = 1
        self.state = 0

    def reset(self) -> int:
        """Reset to state 0 (bandit has single state)."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Pull arm and return reward."""
        if action < 0 or action >= self.k:
            raise ValueError(f"action must be in [0, {self.k}), got {action}")

        # Bernoulli reward
        reward = 1.0 if self.rng.random() < self.probs[action] else 0.0

        # Bandit episodes terminate after one action (standard for bandit problems)
        return self.state, reward, True, {}

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        self.rng = seed_rng(seed)

