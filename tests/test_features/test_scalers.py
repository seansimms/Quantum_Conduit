"""Tests for classical scalers."""

import numpy as np

from qconduit.features import MinMaxScaler, RobustScaler, StandardScaler


def test_standard_scaler_roundtrip():
    X = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    restored = scaler.inverse_transform(transformed)
    assert np.allclose(restored, X)
    assert np.allclose(transformed[:, 1], 0.0)


def test_standard_scaler_zero_variance():
    X = np.ones((4, 1))
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    assert np.allclose(transformed, 0.0)


def test_robust_scaler_basic():
    X = np.array([[1.0], [2.0], [100.0]])
    scaler = RobustScaler().fit(X)
    transformed = scaler.transform(X)
    assert transformed[2] > transformed[0]
    restored = scaler.inverse_transform(transformed)
    assert np.allclose(restored, X)


def test_minmax_scaler_clip():
    X = np.array([[0.0], [5.0]])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0), clip=True).fit(X)
    transformed = scaler.transform([[10.0]])
    assert transformed.max() <= 1.0
    assert np.allclose(scaler.inverse_transform([[0.0]]), [[2.5]])


def test_scalers_deterministic():
    X = np.array([[0.0], [1.0], [2.0]])
    scaler_a = StandardScaler().fit(X)
    scaler_b = StandardScaler().fit(X)
    assert np.allclose(scaler_a.transform(X), scaler_b.transform(X))

