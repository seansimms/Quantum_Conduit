"""Utility functions for reinforcement learning algorithms.

This module provides common helpers for RL algorithms including RNG seeding,
exploration strategies, numerical stability functions, and evaluation utilities.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np


def seed_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a seeded NumPy random number generator.

    Args:
        seed: Random seed. If None, defaults to 0 for reproducibility.

    Returns:
        Seeded NumPy random generator.

    Examples:
        >>> rng = seed_rng(42)
        >>> rng.integers(10)
        6
    """
    if seed is None:
        seed = 0
    return np.random.default_rng(seed)


def epsilon_greedy_action(
    Q_row: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select action using epsilon-greedy policy with deterministic tie-breaking.

    With probability (1-epsilon), selects the greedy action (argmax of Q).
    With probability epsilon, selects uniformly at random.
    Tie-breaking for greedy action is deterministic (first argmax).

    Args:
        Q_row: Q-values for current state, shape (n_actions,).
        epsilon: Exploration probability in [0, 1].
        rng: Random number generator for exploration.

    Returns:
        Selected action index.

    Examples:
        >>> rng = seed_rng(0)
        >>> Q = np.array([0.5, 0.8, 0.3])
        >>> action = epsilon_greedy_action(Q, epsilon=0.0, rng=rng)
        >>> action  # Always greedy
        1
        >>> action = epsilon_greedy_action(Q, epsilon=1.0, rng=rng)
        >>> action in [0, 1, 2]  # Random
        True
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    if rng.random() < epsilon:
        # Explore: uniform random
        return int(rng.integers(len(Q_row)))
    else:
        # Exploit: greedy with deterministic tie-breaking (first argmax)
        return int(np.argmax(Q_row))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities.

    Uses the log-sum-exp trick to avoid numerical overflow:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        logits: Input logits, shape (n,).

    Returns:
        Softmax probabilities, shape (n,), sums to 1.0.

    Examples:
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> probs = softmax(logits)
        >>> np.allclose(probs.sum(), 1.0)
        True
        >>> probs[2] > probs[1] > probs[0]  # Highest logit has highest prob
        True
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def discounted_returns(
    rewards: Sequence[float],
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns G_t for Monte Carlo methods.

    Returns G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k+1}
    computed backwards from terminal state.

    Args:
        rewards: Sequence of rewards [r_1, r_2, ..., r_T].
        gamma: Discount factor in [0, 1].

    Returns:
        Array of returns [G_0, G_1, ..., G_{T-1}], shape (T,).

    Examples:
        >>> rewards = [1.0, 2.0, 3.0]
        >>> returns = discounted_returns(rewards, gamma=0.9)
        >>> expected = 1.0 + 0.9*2.0 + 0.9**2*3.0
        >>> np.allclose(returns[0], expected)
        True
    """
    if gamma < 0 or gamma > 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)

    # Compute backwards: G_t = r_{t+1} + gamma * G_{t+1}
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


def compute_advantage(
    returns: np.ndarray,
    baselines: np.ndarray,
) -> np.ndarray:
    """Compute advantage estimates A_t = G_t - b_t.

    Args:
        returns: Monte Carlo returns G_t, shape (T,).
        baselines: Baseline values b_t, shape (T,).

    Returns:
        Advantage estimates A_t, shape (T,).

    Examples:
        >>> returns = np.array([10.0, 5.0, 2.0])
        >>> baselines = np.array([8.0, 4.0, 1.0])
        >>> advantages = compute_advantage(returns, baselines)
        >>> np.allclose(advantages, [2.0, 1.0, 1.0])
        True
    """
    returns = np.asarray(returns, dtype=np.float64)
    baselines = np.asarray(baselines, dtype=np.float64)
    if returns.shape != baselines.shape:
        raise ValueError(
            f"returns and baselines must have same shape, "
            f"got {returns.shape} and {baselines.shape}"
        )
    return returns - baselines


def evaluate_policy(
    env: Any,  # Env type, but avoid circular import
    policy: Callable[[int], int],
    num_episodes: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Evaluate a policy by running episodes and computing average return.

    Args:
        env: Environment implementing reset() and step().
        policy: Function mapping state -> action.
        num_episodes: Number of episodes to run.
        rng: Random number generator (for environment if needed).

    Returns:
        Tuple of (average_return, std_return) across episodes.

    Examples:
        >>> from qconduit.rl.envs import ChainMDP
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> rng = seed_rng(0)
        >>> def policy(state): return 1  # Always go right
        >>> avg_return, std_return = evaluate_policy(env, policy, 10, rng)
        >>> avg_return >= 0.0
        True
    """
    episode_returns = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        episode_returns.append(episode_reward)

    returns_array = np.array(episode_returns, dtype=np.float64)
    return float(np.mean(returns_array)), float(np.std(returns_array))


def greedy_policy_from_Q(Q: np.ndarray) -> Callable[[int], int]:
    """Build deterministic greedy policy from Q-table.

    Policy selects argmax_a Q(s, a) with deterministic tie-breaking.

    Args:
        Q: Q-table, shape (n_states, n_actions).

    Returns:
        Policy function mapping state -> action.

    Examples:
        >>> Q = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> policy = greedy_policy_from_Q(Q)
        >>> policy(0)  # argmax of [0.1, 0.9]
        1
        >>> policy(1)  # argmax of [0.8, 0.2]
        0
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2D array, got shape {Q.shape}")

    def policy(state: int) -> int:
        return int(np.argmax(Q[state]))

    return policy


def constant_schedule(value: float) -> Callable[[int], float]:
    """Create a constant schedule function.

    Args:
        value: Constant value to return.

    Returns:
        Schedule function that returns value for any episode/step index.

    Examples:
        >>> schedule = constant_schedule(0.1)
        >>> schedule(0), schedule(100)
        (0.1, 0.1)
    """
    return lambda episode: value


def linear_decay_schedule(
    start: float,
    end: float,
    total_steps: int,
) -> Callable[[int], float]:
    """Create a linear decay schedule.

    Args:
        start: Initial value.
        end: Final value.
        total_steps: Number of steps to decay over.

    Returns:
        Schedule function that linearly interpolates from start to end.

    Examples:
        >>> schedule = linear_decay_schedule(1.0, 0.0, 10)
        >>> schedule(0), schedule(5), schedule(10)
        (1.0, 0.5, 0.0)
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    def schedule(step: int) -> float:
        if step >= total_steps:
            return end
        t = step / total_steps
        return start * (1 - t) + end * t

    return schedule

