"""Tests for KBinsDiscretizer."""

import numpy as np

from qconduit.features import KBinsDiscretizer


def test_uniform_discretizer_roundtrip():
    X = np.linspace(0, 10, num=6).reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=3, strategy="uniform").fit(X)
    transformed = discretizer.transform(X)
    restored = discretizer.inverse_transform(transformed)
    assert restored.shape == X.shape


def test_quantile_discretizer_onehot():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    discretizer = KBinsDiscretizer(n_bins=2, strategy="quantile", encode="onehot").fit(X)
    transformed = discretizer.transform(X)
    assert transformed.shape == (4, 2)
    restored = discretizer.inverse_transform(transformed)
    assert restored.shape == (4, 1)

