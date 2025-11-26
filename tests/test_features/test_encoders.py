"""Tests for categorical encoders."""

import numpy as np
import pytest

from qconduit.features import OneHotEncoder, TargetEncoder


def test_one_hot_encoder_roundtrip():
    X = np.array([["red"], ["blue"], ["red"], ["green"]])
    encoder = OneHotEncoder().fit(X)
    transformed = encoder.transform(X)
    restored = encoder.inverse_transform(transformed)
    assert restored.shape == X.shape
    assert np.array_equal(restored, X)


def test_one_hot_encoder_drop_first():
    X = np.array([["cat"], ["dog"], ["cat"]])
    encoder = OneHotEncoder(drop="first").fit(X)
    transformed = encoder.transform(X)
    assert transformed.shape[1] == 1
    restored = encoder.inverse_transform(transformed)
    assert np.array_equal(restored, X)


def test_one_hot_encoder_unknown_category():
    X = np.array([["cat"], ["dog"]])
    encoder = OneHotEncoder().fit(X)
    with pytest.raises(ValueError):
        encoder.transform([["bird"]])


def test_target_encoder_basic():
    X = np.array(["a", "a", "b", "c"])
    y = np.array([0.0, 1.0, 1.0, 3.0])
    encoder = TargetEncoder(smoothing=1.0).fit(X, y)
    transformed = encoder.transform(["a", "b", "d"])
    assert transformed.shape == (3, 1)
    assert transformed[2, 0] == pytest.approx(encoder.global_mean_)

