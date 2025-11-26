"""Hidden Markov Model (HMM) implementation.

Provides forward-backward algorithm, Viterbi decoding, and Baum-Welch EM
for parameter estimation. Supports discrete and continuous emissions.

References:
    Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from .utils import logsumexp


class HiddenMarkovModel:
    """Hidden Markov Model with discrete or continuous emissions.

    Supports:
    - Forward-backward algorithm for posterior state probabilities
    - Viterbi algorithm for most likely state sequence
    - Baum-Welch EM for parameter estimation
    - Discrete or continuous emission models

    Attributes:
        n_states: Number of hidden states.
        n_obs: Number of discrete observation symbols (if discrete emissions).
        start_prob: Initial state distribution, shape (n_states,).
        trans_mat: Transition matrix, shape (n_states, n_states).
        emission_prob: Emission probabilities for discrete mode, shape (n_states, n_obs).
        emission_model: Callable for continuous emissions: log_likelihood(obs, state_idx).
        state_names: Optional names for states.
    """

    def __init__(
        self,
        n_states: int,
        start_prob: Optional[np.ndarray] = None,
        trans_mat: Optional[np.ndarray] = None,
        emission_prob: Optional[np.ndarray] = None,
        emission_model: Optional[Callable[[np.ndarray, int], float]] = None,
        state_names: Optional[List[str]] = None,
    ):
        """Initialize HMM.

        Args:
            n_states: Number of hidden states.
            start_prob: Initial state probabilities, shape (n_states,). If None, uniform.
            trans_mat: Transition matrix, shape (n_states, n_states). If None, uniform.
            emission_prob: For discrete emissions, shape (n_states, n_obs).
                If None and emission_model is None, will be initialized randomly.
            emission_model: For continuous emissions, callable(observation, state_index)
                -> log_likelihood.
            state_names: Optional state names.
        """
        self.n_states = n_states
        self.state_names = state_names

        # Initialize start probabilities
        if start_prob is not None:
            start_prob = np.asarray(start_prob)
            if start_prob.shape != (n_states,):
                raise ValueError(f"start_prob shape {start_prob.shape} != ({n_states},)")
            self.start_prob = start_prob / np.sum(start_prob)  # Normalize
        else:
            self.start_prob = np.ones(n_states) / n_states

        # Initialize transition matrix
        if trans_mat is not None:
            trans_mat = np.asarray(trans_mat)
            if trans_mat.shape != (n_states, n_states):
                raise ValueError(f"trans_mat shape {trans_mat.shape} != ({n_states}, {n_states})")
            # Normalize rows
            row_sums = np.sum(trans_mat, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            self.trans_mat = trans_mat / row_sums
        else:
            self.trans_mat = np.ones((n_states, n_states)) / n_states

        # Emission model
        self.emission_model = emission_model
        if emission_prob is not None:
            emission_prob = np.asarray(emission_prob)
            if emission_prob.ndim != 2:
                raise ValueError("emission_prob must be 2D")
            if emission_prob.shape[0] != n_states:
                raise ValueError(f"emission_prob first dim {emission_prob.shape[0]} != {n_states}")
            # Normalize rows
            row_sums = np.sum(emission_prob, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.emission_prob = emission_prob / row_sums
            self.n_obs = emission_prob.shape[1]
            self._discrete_emissions = True
        elif emission_model is not None:
            self.emission_prob = None
            self.n_obs = None
            self._discrete_emissions = False
        else:
            # Default: discrete with 2 symbols
            self.n_obs = 2
            self.emission_prob = np.ones((n_states, self.n_obs)) / self.n_obs
            self._discrete_emissions = True

        # Log-domain versions for stability
        self._log_start = np.log(self.start_prob + 1e-12)
        self._log_trans = np.log(self.trans_mat + 1e-12)

    def _emission_loglik(self, obs: np.ndarray, state: int) -> float:
        """Compute log-likelihood of observation given state.

        Args:
            obs: Observation (scalar for discrete, array for continuous).
            state: State index.

        Returns:
            Log-likelihood.
        """
        if self._discrete_emissions:
            obs_idx = int(obs)
            if obs_idx < 0 or obs_idx >= self.n_obs:
                return -np.inf
            return np.log(self.emission_prob[state, obs_idx] + 1e-12)
        else:
            if self.emission_model is None:
                raise ValueError("emission_model required for continuous emissions")
            return self.emission_model(obs, state)

    def forward(self, obs_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm: compute forward log-probabilities and log-likelihood.

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            Tuple of (log_alpha, log_likelihood) where:
            - log_alpha: shape (T, n_states), log forward probabilities
            - log_likelihood: log P(obs_seq)
        """
        obs_seq = np.asarray(obs_seq)
        T = len(obs_seq)
        log_alpha = np.zeros((T, self.n_states))

        # Initialization: t=0
        for s in range(self.n_states):
            log_alpha[0, s] = self._log_start[s] + self._emission_loglik(obs_seq[0], s)

        # Recursion: t = 1, ..., T-1
        for t in range(1, T):
            for s in range(self.n_states):
                # log_alpha[t, s] = log(sum_i exp(log_alpha[t-1, i] + log_trans[i, s]))
                #                  + emission_loglik(obs_seq[t], s)
                log_probs = log_alpha[t - 1, :] + self._log_trans[:, s]
                log_alpha[t, s] = logsumexp(log_probs) + self._emission_loglik(obs_seq[t], s)

        # Log-likelihood: log P(obs_seq) = logsumexp(log_alpha[T-1, :])
        log_likelihood = logsumexp(log_alpha[T - 1, :])

        return log_alpha, log_likelihood

    def backward(self, obs_seq: np.ndarray) -> np.ndarray:
        """Backward algorithm: compute backward log-probabilities.

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            log_beta: shape (T, n_states), log backward probabilities.
        """
        obs_seq = np.asarray(obs_seq)
        T = len(obs_seq)
        log_beta = np.zeros((T, self.n_states))

        # Initialization: t = T-1 (final time)
        log_beta[T - 1, :] = 0.0  # log(1) = 0

        # Recursion: t = T-2, ..., 0
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                # log_beta[t, s] = log(sum_j exp(log_trans[s, j] + emission_loglik(obs_seq[t+1], j)
                #                              + log_beta[t+1, j]))
                log_probs = (
                    self._log_trans[s, :]
                    + np.array(
                        [
                            self._emission_loglik(obs_seq[t + 1], j)
                            for j in range(self.n_states)
                        ]
                    )
                    + log_beta[t + 1, :]
                )
                log_beta[t, s] = logsumexp(log_probs)

        return log_beta

    def forward_backward(self, obs_seq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward-backward algorithm: compute posterior state probabilities.

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            Tuple of (gamma, xi) where:
            - gamma: shape (T, n_states), posterior P(state_t = s | obs_seq)
            - xi: shape (T-1, n_states, n_states), posterior
                P(state_t = s, state_{t+1} = i | obs_seq)
                (None if T=1)
        """
        log_alpha, log_likelihood = self.forward(obs_seq)
        log_beta = self.backward(obs_seq)
        T = len(obs_seq)

        # Gamma: posterior state probabilities
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        # Xi: posterior transition probabilities (only if T > 1)
        if T > 1:
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for s in range(self.n_states):
                    for s_next in range(self.n_states):
                        log_xi_val = (
                            log_alpha[t, s]
                            + self._log_trans[s, s_next]
                            + self._emission_loglik(obs_seq[t + 1], s_next)
                            + log_beta[t + 1, s_next]
                            - log_likelihood
                        )
                        xi[t, s, s_next] = np.exp(log_xi_val)
        else:
            xi = None

        return gamma, xi

    def viterbi(self, obs_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """Viterbi algorithm: find most likely state sequence.

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            Tuple of (path, log_prob) where:
            - path: shape (T,), most likely state sequence
            - log_prob: log-probability of this path
        """
        obs_seq = np.asarray(obs_seq)
        T = len(obs_seq)

        # Viterbi log-probabilities and backpointers
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        for s in range(self.n_states):
            log_delta[0, s] = self._log_start[s] + self._emission_loglik(obs_seq[0], s)

        # Recursion
        for t in range(1, T):
            for s in range(self.n_states):
                log_probs = log_delta[t - 1, :] + self._log_trans[:, s]
                best_prev = np.argmax(log_probs)
                log_delta[t, s] = log_probs[best_prev] + self._emission_loglik(obs_seq[t], s)
                psi[t, s] = best_prev

        # Termination: find best final state
        best_final = np.argmax(log_delta[T - 1, :])
        log_prob = log_delta[T - 1, best_final]

        # Backtracking
        path = np.zeros(T, dtype=int)
        path[T - 1] = best_final
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, log_prob

    def score(self, obs_seq: np.ndarray) -> float:
        """Compute log-likelihood of observation sequence.

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            Log-likelihood.
        """
        _, log_likelihood = self.forward(obs_seq)
        return log_likelihood

    def sample(
        self, length: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample state and observation sequences from the model.

        Args:
            length: Sequence length T.
            rng: Random number generator. If None, uses default_rng(0).

        Returns:
            Tuple of (states, observations) where both are shape (T,).
        """
        if rng is None:
            rng = np.random.default_rng(0)

        if not self._discrete_emissions:
            raise ValueError("sample() requires discrete emissions")

        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)

        # Sample initial state
        states[0] = rng.choice(self.n_states, p=self.start_prob)

        # Sample observation
        observations[0] = rng.choice(self.n_obs, p=self.emission_prob[states[0], :])

        # Sample subsequent states and observations
        for t in range(1, length):
            states[t] = rng.choice(self.n_states, p=self.trans_mat[states[t - 1], :])
            observations[t] = rng.choice(self.n_obs, p=self.emission_prob[states[t], :])

        return states, observations

    def baum_welch(
        self,
        sequences: List[np.ndarray],
        n_iter: int = 50,
        tol: float = 1e-6,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Baum-Welch EM algorithm for parameter estimation.

        Estimates start_prob, trans_mat, and emission_prob (if discrete) from
        observation sequences.

        Args:
            sequences: List of observation sequences, each shape (T_i,).
            n_iter: Maximum number of EM iterations.
            tol: Convergence tolerance (stop if log-likelihood improvement < tol).
            rng: Random number generator for initialization. If None, uses default_rng(0).

        Returns:
            Dictionary with keys:
            - 'log_likelihoods': list of log-likelihoods per iteration
            - 'n_iter': number of iterations run
            - 'converged': whether converged (bool)
        """
        if rng is None:
            rng = np.random.default_rng(0)

        if not self._discrete_emissions:
            raise ValueError("baum_welch() currently only supports discrete emissions")

        # Initialize parameters randomly if not already set
        if np.allclose(self.start_prob, 1.0 / self.n_states):
            self.start_prob = rng.dirichlet(np.ones(self.n_states))
            self._log_start = np.log(self.start_prob + 1e-12)

        if np.allclose(self.trans_mat, 1.0 / self.n_states):
            self.trans_mat = rng.dirichlet(np.ones(self.n_states), size=self.n_states)
            # Normalize rows
            row_sums = np.sum(self.trans_mat, axis=1, keepdims=True)
            self.trans_mat = self.trans_mat / row_sums
            self._log_trans = np.log(self.trans_mat + 1e-12)

        if np.allclose(self.emission_prob, 1.0 / self.n_obs):
            self.emission_prob = rng.dirichlet(np.ones(self.n_obs), size=self.n_states)
            row_sums = np.sum(self.emission_prob, axis=1, keepdims=True)
            self.emission_prob = self.emission_prob / row_sums

        log_likelihoods = []
        prev_log_likelihood = -np.inf

        for iteration in range(n_iter):
            # E-step: compute sufficient statistics
            total_gamma_start = np.zeros(self.n_states)
            total_gamma = np.zeros((self.n_states,))
            total_xi = np.zeros((self.n_states, self.n_states))
            total_gamma_obs = np.zeros((self.n_states, self.n_obs))

            total_log_likelihood = 0.0

            for obs_seq in sequences:
                gamma, xi = self.forward_backward(obs_seq)
                total_log_likelihood += self.score(obs_seq)

                # Accumulate statistics
                total_gamma_start += gamma[0, :]
                total_gamma += np.sum(gamma, axis=0)

                if xi is not None:
                    total_xi += np.sum(xi, axis=0)

                # Emission statistics
                for t, obs in enumerate(obs_seq):
                    obs_idx = int(obs)
                    if 0 <= obs_idx < self.n_obs:
                        total_gamma_obs[:, obs_idx] += gamma[t, :]

            log_likelihoods.append(total_log_likelihood)

            # Check convergence
            if iteration > 0:
                improvement = total_log_likelihood - prev_log_likelihood
                if improvement < tol:
                    return {
                        "log_likelihoods": log_likelihoods,
                        "n_iter": iteration + 1,
                        "converged": True,
                    }

            prev_log_likelihood = total_log_likelihood

            # M-step: update parameters
            # Start probabilities
            total_start = np.sum(total_gamma_start)
            if total_start > 0:
                self.start_prob = total_gamma_start / total_start
            else:
                self.start_prob = np.ones(self.n_states) / self.n_states
            self.start_prob = self.start_prob + 1e-12  # Regularization
            self.start_prob = self.start_prob / np.sum(self.start_prob)
            self._log_start = np.log(self.start_prob)

            # Transition matrix
            for s in range(self.n_states):
                total_trans = np.sum(total_xi[s, :])
                if total_trans > 0:
                    self.trans_mat[s, :] = total_xi[s, :] / total_trans
                else:
                    self.trans_mat[s, :] = np.ones(self.n_states) / self.n_states
                self.trans_mat[s, :] = self.trans_mat[s, :] + 1e-12  # Regularization
                self.trans_mat[s, :] = self.trans_mat[s, :] / np.sum(self.trans_mat[s, :])
            self._log_trans = np.log(self.trans_mat + 1e-12)

            # Emission probabilities
            for s in range(self.n_states):
                total_emit = np.sum(total_gamma_obs[s, :])
                if total_emit > 0:
                    self.emission_prob[s, :] = total_gamma_obs[s, :] / total_emit
                else:
                    self.emission_prob[s, :] = np.ones(self.n_obs) / self.n_obs
                self.emission_prob[s, :] = self.emission_prob[s, :] + 1e-12  # Regularization
                self.emission_prob[s, :] = (
                    self.emission_prob[s, :] / np.sum(self.emission_prob[s, :])
                )

        return {
            "log_likelihoods": log_likelihoods,
            "n_iter": n_iter,
            "converged": False,
        }

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence (Viterbi).

        Args:
            obs_seq: Observation sequence, shape (T,).

        Returns:
            Most likely state sequence, shape (T,).
        """
        path, _ = self.viterbi(obs_seq)
        return path

