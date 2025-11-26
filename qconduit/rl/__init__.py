"""Classic Reinforcement Learning Algorithms.

This module provides textbook implementations of canonical tabular and policy-gradient
reinforcement learning algorithms, including:

- Tabular algorithms: Monte Carlo control, Q-learning, SARSA, TD(λ)
- Policy gradient: REINFORCE and REINFORCE with baseline
- Simple reference environments: ChainMDP, GridWorldTiny, Bandit

All algorithms are deterministic when RNG seeds are fixed and follow standard
textbook formulations (Sutton & Barto, 2018).
"""

from .agent import Policy, TabularAgent, TabularPolicy
from .envs import Bandit, ChainMDP, Env, GridWorldTiny
from .policy_gradient import reinforce, reinforce_with_baseline
from .tabular import (
    mc_control_on_policy,
    q_learning,
    sarsa,
    td_lambda,
)
from .utils import (
    compute_advantage,
    discounted_returns,
    epsilon_greedy_action,
    evaluate_policy,
    greedy_policy_from_Q,
    seed_rng,
    softmax,
)

__all__ = [
    # Environments
    "Env",
    "ChainMDP",
    "GridWorldTiny",
    "Bandit",
    # Agents and policies
    "TabularAgent",
    "Policy",
    "TabularPolicy",
    # Tabular algorithms
    "mc_control_on_policy",
    "q_learning",
    "sarsa",
    "td_lambda",
    # Policy gradient algorithms
    "reinforce",
    "reinforce_with_baseline",
    # Utilities
    "seed_rng",
    "epsilon_greedy_action",
    "softmax",
    "discounted_returns",
    "compute_advantage",
    "evaluate_policy",
    "greedy_policy_from_Q",
]

