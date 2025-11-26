"""Tests for PCA transformer."""

import numpy as np
import pytest

from qconduit.features import PCA


def test_pca_full_reconstruction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    pca = PCA().fit(X)
    transformed = pca.transform(X)
    reconstructed = pca.inverse_transform(transformed)
    assert np.allclose(reconstructed, X, atol=1e-10)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_pca_dimensionality_reduction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 5))
    pca = PCA(n_components=2).fit(X)
    transformed = pca.transform(X)
    assert transformed.shape == (50, 2)
    reconstructed = pca.inverse_transform(transformed)
    assert reconstructed.shape == X.shape


def test_pca_variance_threshold():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    pca = PCA(n_components=0.9).fit(X)
    assert 1 <= pca.n_components_ <= 3


def test_pca_requires_two_samples():
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        PCA().fit(X)


def test_pca_whiten_roundtrip():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 3))
    pca = PCA(n_components=2, whiten=True).fit(X)
    transformed = pca.transform(X)
    reconstructed = pca.inverse_transform(transformed)
    assert reconstructed.shape == X.shape


def test_pca_invalid_n_components():
    X = np.eye(3)
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(X)
    with pytest.raises(ValueError):
        PCA(n_components=1.2).fit(X)
    with pytest.raises(TypeError):
        PCA(n_components="two").fit(X)


def test_pca_zero_variance_fraction():
    X = np.ones((5, 4))
    pca = PCA(n_components=0.9).fit(X)
    assert pca.n_components_ == 1

