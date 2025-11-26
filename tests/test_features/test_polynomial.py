"""Tests for polynomial features."""

import numpy as np

from qconduit.features import PolynomialFeatures


def test_polynomial_features_degree_two():
    X = np.array([[2.0, 3.0]])
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    transformed = poly.transform(X)
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    assert np.allclose(transformed, expected)
    names = poly.get_feature_names(["a", "b"])
    assert names == ["1", "a", "b", "a^2", "a b", "b^2"]


def test_polynomial_features_interaction_only():
    X = np.array([[1.0, 2.0, 3.0]])
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False).fit(X)
    transformed = poly.transform(X)
    assert transformed.shape[1] == 7

