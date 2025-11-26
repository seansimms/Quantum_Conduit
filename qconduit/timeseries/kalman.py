"""Kalman filter and smoother for linear-Gaussian state-space models.

This module implements the standard Kalman filter (forward pass) and
Rauch-Tung-Striebel (RTS) smoother (backward pass) for linear state-space models.

References:
    - Durbin & Koopman (2012): Time Series Analysis by State Space Methods
    - Hamilton (1994): Time Series Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StateSpace:
    """Linear-Gaussian state-space model representation.

    State-space model:
        x_t = F * x_{t-1} + w_t    (state equation)
        y_t = H * x_t + v_t        (observation equation)

    where:
        w_t ~ N(0, Q)  (state noise)
        v_t ~ N(0, R)  (observation noise)
        x_0 ~ N(x0, P0) (initial state)

    Attributes:
        F: State transition matrix, shape (n_state, n_state).
        H: Observation matrix, shape (n_obs, n_state).
        Q: State noise covariance, shape (n_state, n_state).
        R: Observation noise covariance, shape (n_obs, n_obs).
        x0: Initial state mean, shape (n_state,).
        P0: Initial state covariance, shape (n_state, n_state).
    """

    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray

    def __post_init__(self) -> None:
        """Validate state-space model dimensions."""
        n_state = self.F.shape[0]
        n_obs = self.H.shape[0]

        # Validate shapes
        if self.F.shape[1] != n_state:
            raise ValueError(f"F must be square, got shape {self.F.shape}")
        if self.H.shape[1] != n_state:
            raise ValueError(
                f"H must have n_state={n_state} columns, got shape {self.H.shape}"
            )
        if self.Q.shape != (n_state, n_state):
            raise ValueError(f"Q must be shape ({n_state}, {n_state}), got {self.Q.shape}")
        if self.R.shape != (n_obs, n_obs):
            raise ValueError(f"R must be shape ({n_obs}, {n_obs}), got {self.R.shape}")
        if self.x0.shape != (n_state,):
            raise ValueError(f"x0 must be shape ({n_state},), got {self.x0.shape}")
        if self.P0.shape != (n_state, n_state):
            raise ValueError(
                f"P0 must be shape ({n_state}, {n_state}), got {self.P0.shape}"
            )


def kalman_filter(ss: StateSpace, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run Kalman filter (forward pass).

    Computes filtered state estimates x_{t|t} and covariances P_{t|t}
    for t = 1, ..., T.

    Args:
        ss: StateSpace model.
        y: Observations, shape (T, n_obs).

    Returns:
        Tuple of:
        - x_filtered: Filtered state means, shape (T, n_state).
        - P_filtered: Filtered state covariances, shape (T, n_state, n_state).
        - loglikelihood: Log-likelihood of observations.

    Raises:
        ValueError: If y is not 2D or has wrong number of columns.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

    T = len(y)
    n_state = ss.F.shape[0]
    n_obs = ss.H.shape[0]

    if y.shape[1] != n_obs:
        raise ValueError(
            f"y must have {n_obs} columns (observations per time), got {y.shape[1]}"
        )

    # Initialize
    x_filtered = np.zeros((T, n_state))
    P_filtered = np.zeros((T, n_state, n_state))

    # Prior: x_{0|0} = x0, P_{0|0} = P0
    x_pred = ss.x0.copy()
    P_pred = ss.P0.copy()

    loglik = 0.0

    for t in range(T):
        # Update step: x_{t|t} = x_{t|t-1} + K_t * (y_t - H * x_{t|t-1})
        #              P_{t|t} = (I - K_t * H) * P_{t|t-1}

        # Innovation
        y_pred = ss.H @ x_pred  # Predicted observation
        innovation = y[t] - y_pred
        S = ss.H @ P_pred @ ss.H.T + ss.R  # Innovation covariance

        # Kalman gain: K = P_pred * H' * S^(-1)
        try:
            K = P_pred @ ss.H.T @ np.linalg.solve(S, np.eye(len(S)))
        except np.linalg.LinAlgError:
            # Fallback: add small ridge for numerical stability
            S_ridge = S + 1e-12 * np.eye(len(S))
            K = P_pred @ ss.H.T @ np.linalg.solve(S_ridge, np.eye(len(S)))

        # Update filtered state
        x_filtered[t] = x_pred + K @ innovation
        P_filtered[t] = P_pred - K @ ss.H @ P_pred

        # Log-likelihood contribution (Gaussian)
        det_S = np.linalg.det(S)
        if det_S > 0:
            loglik -= 0.5 * (
                np.log(2 * np.pi) * n_obs
                + np.log(det_S)
                + innovation.T @ np.linalg.solve(S, innovation)
            )

        # Predict step for next iteration: x_{t+1|t} = F * x_{t|t}
        if t < T - 1:
            x_pred = ss.F @ x_filtered[t]
            P_pred = ss.F @ P_filtered[t] @ ss.F.T + ss.Q

    return x_filtered, P_filtered, loglik


def kalman_smoother(ss: StateSpace, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run Kalman smoother (Rauch-Tung-Striebel backward pass).

    Computes smoothed state estimates x_{t|T} and covariances P_{t|T}
    using all observations y_1, ..., y_T.

    Args:
        ss: StateSpace model.
        y: Observations, shape (T, n_obs).

    Returns:
        Tuple of:
        - x_smoothed: Smoothed state means, shape (T, n_state).
        - P_smoothed: Smoothed state covariances, shape (T, n_state, n_state).

    Raises:
        ValueError: If y is not 2D or has wrong number of columns.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

    T = len(y)
    n_state = ss.F.shape[0]

    # Forward pass: get filtered states
    x_filtered, P_filtered, _ = kalman_filter(ss, y)

    # Initialize smoothed states
    x_smoothed = np.zeros((T, n_state))
    P_smoothed = np.zeros((T, n_state, n_state))

    # Initialize: last smoothed state equals filtered state
    x_smoothed[-1] = x_filtered[-1]
    P_smoothed[-1] = P_filtered[-1]

    # Backward pass (RTS smoother)
    for t in range(T - 2, -1, -1):
        # Predicted state for t+1 given t
        x_pred_next = ss.F @ x_filtered[t]
        P_pred_next = ss.F @ P_filtered[t] @ ss.F.T + ss.Q

        # Smoothing gain
        try:
            J = P_filtered[t] @ ss.F.T @ np.linalg.solve(
                P_pred_next + 1e-12 * np.eye(n_state), np.eye(n_state)
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            J = P_filtered[t] @ ss.F.T @ np.linalg.pinv(P_pred_next)

        # Smoothed state
        x_smoothed[t] = x_filtered[t] + J @ (x_smoothed[t + 1] - x_pred_next)
        P_smoothed[t] = (
            P_filtered[t] + J @ (P_smoothed[t + 1] - P_pred_next) @ J.T
        )

    return x_smoothed, P_smoothed


def kalman_predict(
    ss: StateSpace, x_last: np.ndarray, P_last: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-step prediction from Kalman filter state.

    Predicts future states and observations starting from the last filtered state.

    Args:
        ss: StateSpace model.
        x_last: Last filtered/predicted state, shape (n_state,).
        P_last: Last filtered/predicted covariance, shape (n_state, n_state).
        steps: Number of steps ahead to predict.

    Returns:
        Tuple of:
        - x_pred: Predicted state means, shape (steps, n_state).
        - y_pred: Predicted observations, shape (steps, n_obs).
    """
    x_last = np.asarray(x_last)
    P_last = np.asarray(P_last)

    n_state = ss.F.shape[0]
    n_obs = ss.H.shape[0]

    if x_last.shape != (n_state,):
        raise ValueError(f"x_last must be shape ({n_state},), got {x_last.shape}")
    if P_last.shape != (n_state, n_state):
        raise ValueError(
            f"P_last must be shape ({n_state}, {n_state}), got {P_last.shape}"
        )

    x_pred = np.zeros((steps, n_state))
    y_pred = np.zeros((steps, n_obs))

    x = x_last.copy()
    P = P_last.copy()

    for h in range(steps):
        # Predict state
        x = ss.F @ x
        P = ss.F @ P @ ss.F.T + ss.Q

        x_pred[h] = x
        y_pred[h] = ss.H @ x

    return x_pred, y_pred

