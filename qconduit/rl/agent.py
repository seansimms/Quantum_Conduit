"""Agent and policy interfaces for reinforcement learning.

This module defines base classes and interfaces for tabular agents and
policy representations used by both tabular and policy-gradient algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .utils import seed_rng, softmax


class TabularAgent(ABC):
    """Base class for tabular RL agents.

    Tabular agents maintain Q-tables or value tables and select actions
    based on these estimates.
    """

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Select action for given state.

        Args:
            state: Current state index.

        Returns:
            Selected action index.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update agent parameters based on experience."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset agent to initial state."""
        raise NotImplementedError


class Policy(ABC):
    """Interface for policy representations.

    Policies define action selection probabilities and can sample actions.
    Used by policy-gradient algorithms.
    """

    @abstractmethod
    def action_probs(self, state: int) -> np.ndarray:
        """Get action probabilities for given state.

        Args:
            state: Current state index.

        Returns:
            Action probabilities, shape (n_actions,), sums to 1.0.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_action(
        self,
        state: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample action from policy for given state.

        Args:
            state: Current state index.
            rng: Random number generator.

        Returns:
            Sampled action index.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update policy parameters."""
        raise NotImplementedError


class TabularPolicy(Policy):
    """Tabular softmax policy parameterized by action preferences θ(s,a).

    Policy probabilities are computed via softmax:
    π(a|s) = exp(θ(s,a)) / sum_a' exp(θ(s,a'))

    Args:
        n_states: Number of states.
        n_actions: Number of actions.
        seed: Random seed for initialization.

    Examples:
        >>> policy = TabularPolicy(n_states=5, n_actions=2, seed=0)
        >>> probs = policy.action_probs(state=0)
        >>> np.allclose(probs.sum(), 1.0)
        True
        >>> action = policy.sample_action(state=0, rng=seed_rng(0))
        >>> action in [0, 1]
        True
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        seed: Optional[int] = None,
    ):
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")

        self.n_states = n_states
        self.n_actions = n_actions
        rng = seed_rng(seed)

        # Initialize action preferences (logits) to small random values
        # This ensures initial policy is near-uniform
        self.theta = rng.normal(0.0, 0.01, size=(n_states, n_actions)).astype(
            np.float64
        )

    def action_probs(self, state: int) -> np.ndarray:
        """Get action probabilities via softmax over preferences."""
        if state < 0 or state >= self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states}), got {state}")
        return softmax(self.theta[state])

    def sample_action(
        self,
        state: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample action from policy using action probabilities."""
        probs = self.action_probs(state)
        return int(rng.choice(self.n_actions, p=probs))

    def update(self, *args, **kwargs) -> None:
        """Update policy parameters.

        This is a placeholder; actual updates are done by policy-gradient
        algorithms that modify self.theta directly.
        """
        pass

    def get_theta(self) -> np.ndarray:
        """Get current action preference parameters.

        Returns:
            Theta array, shape (n_states, n_actions).
        """
        return self.theta.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        """Set action preference parameters.

        Args:
            theta: New theta array, shape (n_states, n_actions).
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"theta shape must be ({self.n_states}, {self.n_actions}), "
                f"got {theta.shape}"
            )
        self.theta = theta

