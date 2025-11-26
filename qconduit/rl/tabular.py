"""Tabular reinforcement learning algorithms.

This module implements canonical tabular RL algorithms:
- Monte Carlo control (on-policy, first-visit)
- Q-learning (off-policy TD control)
- SARSA (on-policy TD control)
- TD(λ) with eligibility traces

All algorithms follow standard textbook formulations (Sutton & Barto, 2018)
and are deterministic when RNG seeds are fixed.
"""

from collections.abc import Callable
from typing import Optional, Tuple

import numpy as np

from .envs import Env
from .utils import epsilon_greedy_action, greedy_policy_from_Q, seed_rng


def mc_control_on_policy(
    env: Env,
    num_episodes: int,
    gamma: float,
    epsilon_schedule: Callable[[int], float],
    rng: Optional[np.random.Generator] = None,
    initial_Q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Callable[[int], int]]:
    """Monte Carlo control (on-policy) with first-visit returns.

    Uses epsilon-greedy behavior policy and first-visit MC for return
    estimation. Returns Q-table and greedy policy derived from Q.

    Algorithm (Sutton & Barto, 2018, Ch. 5):
    1. Generate episode using epsilon-greedy policy
    2. For first visit to each (s, a) in episode, compute return G_t
    3. Update Q(s, a) incrementally: Q(s, a) += alpha * (G_t - Q(s, a))

    Args:
        env: Environment implementing reset() and step().
        num_episodes: Number of episodes to run.
        gamma: Discount factor in [0, 1].
        epsilon_schedule: Function mapping episode index -> epsilon.
        rng: Random number generator. If None, uses seed_rng(0).
        initial_Q: Initial Q-table, shape (n_states, n_actions).
            If None, initializes to zeros.

    Returns:
        Tuple of (Q, policy) where:
        - Q: Learned Q-table, shape (n_states, n_actions)
        - policy: Greedy policy function mapping state -> action

    Examples:
        >>> from qconduit.rl.envs import ChainMDP
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> rng = seed_rng(0)
        >>> Q, policy = mc_control_on_policy(
        ...     env, num_episodes=100, gamma=0.99,
        ...     epsilon_schedule=lambda e: 0.1, rng=rng
        ... )
        >>> Q.shape
        (5, 2)
        >>> policy(0) in [0, 1]
        True
    """
    if rng is None:
        rng = seed_rng(0)

    n_states = env.observation_space
    n_actions = env.action_space

    # Initialize Q-table
    if initial_Q is None:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        Q = np.asarray(initial_Q, dtype=np.float64).copy()
        if Q.shape != (n_states, n_actions):
            raise ValueError(
                f"initial_Q shape must be ({n_states}, {n_actions}), "
                f"got {Q.shape}"
            )

    # Visit counts for incremental averaging (first-visit)
    returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_count = np.zeros((n_states, n_actions), dtype=np.int64)

    for episode in range(num_episodes):
        epsilon = epsilon_schedule(episode)

        # Generate episode
        episode_data = []
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            action = epsilon_greedy_action(Q[state], epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # Compute returns backwards
        T = len(episode_data)
        G = 0.0
        returns = []
        for t in range(T - 1, -1, -1):
            _, _, reward = episode_data[t]
            G = reward + gamma * G
            returns.insert(0, G)

        # First-visit MC: update Q only for first occurrence of (s, a)
        visited = set()
        for t, (state, action, _) in enumerate(episode_data):
            if (state, action) not in visited:
                visited.add((state, action))
                G_t = returns[t]

                # Incremental averaging
                returns_count[state, action] += 1
                n = returns_count[state, action]
                returns_sum[state, action] += G_t
                Q[state, action] = returns_sum[state, action] / n

    policy = greedy_policy_from_Q(Q)
    return Q, policy


def q_learning(
    env: Env,
    num_episodes: int,
    alpha_schedule: Callable[[int], float],
    epsilon_schedule: Callable[[int], float],
    gamma: float,
    rng: Optional[np.random.Generator] = None,
    Q0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Q-learning (off-policy TD control).

    Updates Q(s, a) using max over next state actions:
    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

    Uses epsilon-greedy behavior policy for exploration.

    Algorithm (Sutton & Barto, 2018, Ch. 6):
    For each step in episode:
        1. Select action a using epsilon-greedy from Q
        2. Take action, observe r, s'
        3. Update: Q(s, a) += alpha * (r + gamma * max Q(s', a') - Q(s, a))

    Args:
        env: Environment implementing reset() and step().
        num_episodes: Number of episodes to run.
        alpha_schedule: Function mapping episode index -> learning rate.
        epsilon_schedule: Function mapping episode index -> epsilon.
        gamma: Discount factor in [0, 1].
        rng: Random number generator. If None, uses seed_rng(0).
        Q0: Initial Q-table, shape (n_states, n_actions). If None, zeros.

    Returns:
        Learned Q-table, shape (n_states, n_actions).

    Examples:
        >>> from qconduit.rl.envs import ChainMDP
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> rng = seed_rng(0)
        >>> Q = q_learning(
        ...     env, num_episodes=100,
        ...     alpha_schedule=lambda e: 0.1,
        ...     epsilon_schedule=lambda e: 0.1,
        ...     gamma=0.99, rng=rng
        ... )
        >>> Q.shape
        (5, 2)
    """
    if rng is None:
        rng = seed_rng(0)

    n_states = env.observation_space
    n_actions = env.action_space

    if Q0 is None:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        Q = np.asarray(Q0, dtype=np.float64).copy()
        if Q.shape != (n_states, n_actions):
            raise ValueError(
                f"Q0 shape must be ({n_states}, {n_actions}), got {Q.shape}"
            )

    step_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            epsilon = epsilon_schedule(episode)
            alpha = alpha_schedule(episode)

            # Epsilon-greedy action selection
            action = epsilon_greedy_action(Q[state], epsilon, rng)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Q-learning update (off-policy: uses max over next state)
            max_next_Q = np.max(Q[next_state]) if not done else 0.0
            td_target = reward + gamma * max_next_Q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            step_count += 1

    return Q


def sarsa(
    env: Env,
    num_episodes: int,
    alpha_schedule: Callable[[int], float],
    epsilon_schedule: Callable[[int], float],
    gamma: float,
    rng: Optional[np.random.Generator] = None,
    Q0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SARSA (on-policy TD control).

    Updates Q(s, a) using next action a' sampled from behavior policy:
    Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))

    Uses epsilon-greedy behavior policy for both action selection and
    next-action estimation.

    Algorithm (Sutton & Barto, 2018, Ch. 6):
    For each step in episode:
        1. Select action a using epsilon-greedy from Q
        2. Take action, observe r, s'
        3. Select next action a' using epsilon-greedy from Q
        4. Update: Q(s, a) += alpha * (r + gamma * Q(s', a') - Q(s, a))

    Args:
        env: Environment implementing reset() and step().
        num_episodes: Number of episodes to run.
        alpha_schedule: Function mapping episode index -> learning rate.
        epsilon_schedule: Function mapping episode index -> epsilon.
        gamma: Discount factor in [0, 1].
        rng: Random number generator. If None, uses seed_rng(0).
        Q0: Initial Q-table, shape (n_states, n_actions). If None, zeros.

    Returns:
        Learned Q-table, shape (n_states, n_actions).

    Examples:
        >>> from qconduit.rl.envs import ChainMDP
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> rng = seed_rng(0)
        >>> Q = sarsa(
        ...     env, num_episodes=100,
        ...     alpha_schedule=lambda e: 0.1,
        ...     epsilon_schedule=lambda e: 0.1,
        ...     gamma=0.99, rng=rng
        ... )
        >>> Q.shape
        (5, 2)
    """
    if rng is None:
        rng = seed_rng(0)

    n_states = env.observation_space
    n_actions = env.action_space

    if Q0 is None:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        Q = np.asarray(Q0, dtype=np.float64).copy()
        if Q.shape != (n_states, n_actions):
            raise ValueError(
                f"Q0 shape must be ({n_states}, {n_actions}), got {Q.shape}"
            )

    for episode in range(num_episodes):
        state = env.reset()
        epsilon = epsilon_schedule(episode)

        # Select initial action
        action = epsilon_greedy_action(Q[state], epsilon, rng)
        done = False

        while not done:
            alpha = alpha_schedule(episode)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Select next action (on-policy: from behavior policy)
            if not done:
                next_action = epsilon_greedy_action(Q[next_state], epsilon, rng)
                next_Q = Q[next_state, next_action]
            else:
                next_Q = 0.0

            # SARSA update
            td_target = reward + gamma * next_Q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action if not done else 0

    return Q


def td_lambda(
    env: Env,
    num_episodes: int,
    alpha: float,
    lambda_: float,
    gamma: float,
    epsilon_schedule: Callable[[int], float],
    trace_type: str = "accumulating",
    rng: Optional[np.random.Generator] = None,
    Q0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """TD(λ) with eligibility traces (action-value version: Q(λ)).

    Implements Q(λ) algorithm with eligibility traces for tabular action-value
    learning. Supports both accumulating and replacing traces.

    Algorithm (Sutton & Barto, 2018, Ch. 12):
    For each step:
        1. Select action a using epsilon-greedy
        2. Take action, observe r, s'
        3. Compute TD error: delta = r + gamma * Q(s', a') - Q(s, a)
        4. Update eligibility traces: e(s, a) += 1 (accumulating) or e(s, a) = 1 (replacing)
        5. Update all Q-values: Q(s, a) += alpha * delta * e(s, a)
        6. Decay traces: e(s, a) *= gamma * lambda

    Args:
        env: Environment implementing reset() and step().
        num_episodes: Number of episodes to run.
        alpha: Learning rate (constant).
        lambda_: Trace decay parameter in [0, 1].
        gamma: Discount factor in [0, 1].
        epsilon_schedule: Function mapping episode index -> epsilon.
        trace_type: "accumulating" or "replacing" traces.
        rng: Random number generator. If None, uses seed_rng(0).
        Q0: Initial Q-table, shape (n_states, n_actions). If None, zeros.

    Returns:
        Learned Q-table, shape (n_states, n_actions).

    Examples:
        >>> from qconduit.rl.envs import ChainMDP
        >>> env = ChainMDP(n_states=5, seed=0)
        >>> rng = seed_rng(0)
        >>> Q = td_lambda(
        ...     env, num_episodes=100, alpha=0.1, lambda_=0.5,
        ...     gamma=0.99, epsilon_schedule=lambda e: 0.1, rng=rng
        ... )
        >>> Q.shape
        (5, 2)
    """
    if rng is None:
        rng = seed_rng(0)

    if trace_type not in ["accumulating", "replacing"]:
        raise ValueError(f"trace_type must be 'accumulating' or 'replacing', got {trace_type}")

    n_states = env.observation_space
    n_actions = env.action_space

    if Q0 is None:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        Q = np.asarray(Q0, dtype=np.float64).copy()
        if Q.shape != (n_states, n_actions):
            raise ValueError(
                f"Q0 shape must be ({n_states}, {n_actions}), got {Q.shape}"
            )

    for episode in range(num_episodes):
        state = env.reset()
        epsilon = epsilon_schedule(episode)

        # Initialize eligibility traces
        e = np.zeros((n_states, n_actions), dtype=np.float64)

        # Select initial action
        action = epsilon_greedy_action(Q[state], epsilon, rng)
        done = False

        while not done:
            # Take step
            next_state, reward, done, _ = env.step(action)

            # Select next action
            if not done:
                next_action = epsilon_greedy_action(Q[next_state], epsilon, rng)
                next_Q = Q[next_state, next_action]
            else:
                next_Q = 0.0
                next_action = 0

            # TD error
            td_error = reward + gamma * next_Q - Q[state, action]

            # Update eligibility trace for current (s, a)
            if trace_type == "accumulating":
                e[state, action] += 1.0
            else:  # replacing
                e[state, action] = 1.0

            # Update all Q-values using traces
            Q += alpha * td_error * e

            # Decay traces
            e *= gamma * lambda_

            state = next_state
            action = next_action

    return Q

