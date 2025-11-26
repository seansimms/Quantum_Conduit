"""Tests for Kalman filter and smoother."""

from __future__ import annotations

import numpy as np
import pytest

from qconduit.timeseries.kalman import StateSpace, kalman_filter, kalman_predict, kalman_smoother


class TestStateSpace:
    """Tests for StateSpace dataclass."""

    def test_state_space_valid(self):
        """Test valid StateSpace construction."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        x0 = np.zeros(2)
        P0 = np.eye(2)

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        assert ss.F.shape == (2, 2)
        assert ss.H.shape == (1, 2)

    def test_state_space_invalid_shapes(self):
        """Test StateSpace with invalid shapes."""
        F = np.array([[1.0]])
        H = np.array([[1.0, 0.0]])  # Wrong number of columns
        Q = np.eye(1)
        R = np.array([[0.5]])
        x0 = np.zeros(1)
        P0 = np.eye(1)

        with pytest.raises(ValueError, match="must have n_state"):
            StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)


class TestKalmanFilter:
    """Tests for Kalman filter."""

    def test_kalman_filter_local_level(self):
        """Test Kalman filter on local level model (random walk + noise)."""
        np.random.seed(42)

        # Local level model: x_t = x_{t-1} + w_t, y_t = x_t + v_t
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])  # State noise variance
        R = np.array([[1.0]])  # Observation noise variance
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

        # Generate true state and observations
        T = 100
        true_state = np.zeros(T)
        y = np.zeros((T, 1))

        true_state[0] = 0.0
        for t in range(1, T):
            true_state[t] = true_state[t - 1] + np.random.normal(0, np.sqrt(Q[0, 0]))
            y[t, 0] = true_state[t] + np.random.normal(0, np.sqrt(R[0, 0]))

        # Run filter
        x_filtered, P_filtered, loglik = kalman_filter(ss, y)

        assert x_filtered.shape == (T, 1)
        assert P_filtered.shape == (T, 1, 1)
        assert isinstance(loglik, float)

        # Check that filtered states track true state reasonably
        # (correlation should be high)
        corr = np.corrcoef(x_filtered[:, 0], true_state)[0, 1]
        assert corr > 0.7

    def test_kalman_filter_1d_observations(self):
        """Test Kalman filter with 1D observation array."""
        np.random.seed(42)

        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

        T = 50
        y = np.random.randn(T)  # 1D array

        x_filtered, P_filtered, loglik = kalman_filter(ss, y)

        assert x_filtered.shape == (T, 1)
        assert P_filtered.shape == (T, 1, 1)


class TestKalmanSmoother:
    """Tests for Kalman smoother."""

    def test_kalman_smoother_local_level(self):
        """Test Kalman smoother on local level model."""
        np.random.seed(42)

        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

        T = 100
        true_state = np.zeros(T)
        y = np.zeros((T, 1))

        true_state[0] = 0.0
        for t in range(1, T):
            true_state[t] = true_state[t - 1] + np.random.normal(0, np.sqrt(Q[0, 0]))
            y[t, 0] = true_state[t] + np.random.normal(0, np.sqrt(R[0, 0]))

        # Run smoother
        x_smoothed, P_smoothed = kalman_smoother(ss, y)

        assert x_smoothed.shape == (T, 1)
        assert P_smoothed.shape == (T, 1, 1)

        # Smoothed states should have lower variance than filtered states
        x_filtered, P_filtered, _ = kalman_filter(ss, y)
        assert np.mean(np.diag(P_smoothed[-1])) <= np.mean(np.diag(P_filtered[-1]))


class TestKalmanPredict:
    """Tests for Kalman prediction."""

    def test_kalman_predict(self):
        """Test multi-step prediction."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])  # Constant velocity model
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=np.zeros(2), P0=np.eye(2))

        x_last = np.array([10.0, 1.0])  # Position=10, velocity=1
        P_last = np.eye(2) * 0.5

        x_pred, y_pred = kalman_predict(ss, x_last, P_last, steps=5)

        assert x_pred.shape == (5, 2)
        assert y_pred.shape == (5, 1)

        # Check that position increases (velocity is positive)
        assert x_pred[0, 0] > x_last[0]

    def test_kalman_predict_invalid(self):
        """Test prediction with invalid inputs."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])

        ss = StateSpace(F=F, H=H, Q=Q, R=R, x0=np.zeros(1), P0=np.eye(1))

        x_last = np.array([1.0, 2.0])  # Wrong dimension
        P_last = np.eye(1)

        with pytest.raises(ValueError, match="x_last must be shape"):
            kalman_predict(ss, x_last, P_last, steps=5)

