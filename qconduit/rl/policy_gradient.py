"""Policy gradient reinforcement learning algorithms.

This module implements REINFORCE and REINFORCE with baseline, classic
on-policy Monte Carlo policy gradient methods for episodic tasks.

All algorithms follow standard textbook formulations (Sutton & Barto, 2018).
"""

from collections.abc import Callable
from typing import Optional

import numpy as np

from .agent import Policy, TabularPolicy
from .envs import Env
from .utils import compute_advantage, discounted_returns, seed_rng


def reinforce(
    env: Env,
    policy: Policy,
    num_episodes: int,
    gamma: float,
    alpha_schedule: Callable[[int], float],
    baseline: Optional[Callable[[int], float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Policy:
    """REINFORCE: Monte Carlo policy gradient (episodic).

    On-policy gradient ascent on expected return. Updates policy parameters
    using gradient of log-probability weighted by returns.

    Policy gradient update (Sutton & Barto, 2018, Ch. 13):
    θ_{s,a} <- θ_{s,a} + α * Σ_t ∇_θ log π_θ(a_t|s_t) * (G_t - b_t)

    For softmax policy:
    ∇_θ log π_θ(a|s) = 1_{a=a_t} - π_θ(a|s)

    Args:
        env: Environment implementing reset() and step().
        policy: Policy to update (must be TabularPolicy).
        num_episodes: Number of episodes to run.
        gamma: Discount factor in [0, 1].
        alpha_schedule: Function mapping episode index -> learning rate.
        baseline: Optional baseline function mapping state -> baseline value.
            If None, no baseline is used (vanilla REINFORCE).
        rng: Random number generator. If None, uses seed_rng(0).

    Returns:
        Updated policy (same object, modified in-place).

    Examples:
        >>> from qconduit.rl.envs import Bandit
        >>> from qconduit.rl.agent import TabularPolicy
        >>> env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        >>> policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        >>> rng = seed_rng(0)
        >>> policy = reinforce(
        ...     env, policy, num_episodes=100, gamma=1.0,
        ...     alpha_schedule=lambda e: 0.1, rng=rng
        ... )
        >>> probs = policy.action_probs(0)
        >>> np.allclose(probs.sum(), 1.0)
        True
    """
    if rng is None:
        rng = seed_rng(0)

    if not isinstance(policy, TabularPolicy):
        raise TypeError(f"policy must be TabularPolicy, got {type(policy)}")

    n_actions = env.action_space

    for episode in range(num_episodes):
        alpha = alpha_schedule(episode)

        # Generate episode
        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state, rng)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = discounted_returns(rewards, gamma)

        # Compute baselines if provided
        if baseline is not None:
            baselines = np.array([baseline(s) for s in states], dtype=np.float64)
            advantages = compute_advantage(returns, baselines)
        else:
            advantages = returns

        # Update policy parameters
        # For each (s, a) in episode, update theta[s, a]
        for t, (s, a) in enumerate(zip(states, actions)):
            # Get current action probabilities
            probs = policy.action_probs(s)

            # Policy gradient for softmax:
            # ∇_θ log π(a|s) = 1_{a=a_t} - π(a|s)
            # Update: θ[s, a] += alpha * (1 - π(a|s)) * advantage
            #        θ[s, a'] += alpha * (-π(a'|s)) * advantage for a' != a

            # Update all actions in state s
            for action_idx in range(n_actions):
                if action_idx == a:
                    # Positive gradient for selected action
                    grad = 1.0 - probs[action_idx]
                else:
                    # Negative gradient for other actions
                    grad = -probs[action_idx]

                policy.theta[s, action_idx] += alpha * grad * advantages[t]

    return policy


def reinforce_with_baseline(
    env: Env,
    policy: Policy,
    num_episodes: int,
    gamma: float,
    alpha_schedule: Callable[[int], float],
    baseline_type: str = "running_mean",
    rng: Optional[np.random.Generator] = None,
) -> Policy:
    """REINFORCE with baseline (advantage-based).

    Same as REINFORCE but uses a baseline to reduce variance of gradient
    estimates. Supports two baseline types:
    - "running_mean": Global running average of returns
    - "state_value": State-value function V(s) estimated via MC

    Args:
        env: Environment implementing reset() and step().
        policy: Policy to update (must be TabularPolicy).
        num_episodes: Number of episodes to run.
        gamma: Discount factor in [0, 1].
        alpha_schedule: Function mapping episode index -> learning rate.
        baseline_type: "running_mean" or "state_value".
        rng: Random number generator. If None, uses seed_rng(0).

    Returns:
        Updated policy (same object, modified in-place).

    Examples:
        >>> from qconduit.rl.envs import Bandit
        >>> from qconduit.rl.agent import TabularPolicy
        >>> env = Bandit(k=3, probs=[0.1, 0.5, 0.9], seed=0)
        >>> policy = TabularPolicy(n_states=1, n_actions=3, seed=0)
        >>> rng = seed_rng(0)
        >>> policy = reinforce_with_baseline(
        ...     env, policy, num_episodes=100, gamma=1.0,
        ...     alpha_schedule=lambda e: 0.1, rng=rng
        ... )
        >>> probs = policy.action_probs(0)
        >>> np.allclose(probs.sum(), 1.0)
        True
    """
    if rng is None:
        rng = seed_rng(0)

    if not isinstance(policy, TabularPolicy):
        raise TypeError(f"policy must be TabularPolicy, got {type(policy)}")

    if baseline_type not in ["running_mean", "state_value"]:
        raise ValueError(
            f"baseline_type must be 'running_mean' or 'state_value', "
            f"got {baseline_type}"
        )

    n_states = env.observation_space

    # Initialize baseline
    if baseline_type == "running_mean":
        # Global running average
        running_mean = 0.0
        episode_count = 0

        def baseline_fn(state: int) -> float:
            return running_mean

    else:  # state_value
        # State-value function V(s) estimated via MC
        V = np.zeros(n_states, dtype=np.float64)
        returns_sum = np.zeros(n_states, dtype=np.float64)
        returns_count = np.zeros(n_states, dtype=np.int64)

        def baseline_fn(state: int) -> float:
            return V[state]

    for episode in range(num_episodes):
        alpha = alpha_schedule(episode)

        # Generate episode
        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state, rng)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = discounted_returns(rewards, gamma)

        # Update baseline
        if baseline_type == "running_mean":
            # Update global running mean
            episode_return = returns[0] if len(returns) > 0 else 0.0
            episode_count += 1
            running_mean = running_mean + (episode_return - running_mean) / episode_count

        else:  # state_value
            # Update V(s) using first-visit MC
            visited = set()
            for t, s in enumerate(states):
                if s not in visited:
                    visited.add(s)
                    G_t = returns[t]

                    # Incremental averaging
                    returns_count[s] += 1
                    n = returns_count[s]
                    returns_sum[s] += G_t
                    V[s] = returns_sum[s] / n

        # Compute advantages
        baselines = np.array([baseline_fn(s) for s in states], dtype=np.float64)
        advantages = compute_advantage(returns, baselines)

        # Update policy (same as REINFORCE but with advantages)
        n_actions = env.action_space
        for t, (s, a) in enumerate(zip(states, actions)):
            probs = policy.action_probs(s)

            for action_idx in range(n_actions):
                if action_idx == a:
                    grad = 1.0 - probs[action_idx]
                else:
                    grad = -probs[action_idx]

                policy.theta[s, action_idx] += alpha * grad * advantages[t]

    return policy

