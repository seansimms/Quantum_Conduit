"""API consistency tests for feature transformers."""

import numpy as np
import pytest

from qconduit.features import (
    PCA,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
    Transformer,
)
from qconduit.features.base import check_is_fitted
from qconduit.features.utils import check_array, ensure_same_shape, validate_quantile_range


@pytest.mark.parametrize(
    ("transformer", "X"),
    [
        (StandardScaler(), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        (RobustScaler(), np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float32)),
        (MinMaxScaler(), np.array([[0.0], [1.0]], dtype=np.float32)),
        (PCA(n_components=1), np.array([[1.0, 0.0], [0.0, 1.0]])),
        (PolynomialFeatures(degree=2), np.array([[1.0, 2.0]])),
        (OneHotEncoder(), np.array([["a"], ["b"]], dtype=object)),
        (KBinsDiscretizer(n_bins=3), np.array([[0.0], [1.0], [2.0]])),
    ],
)
def test_fit_transform_matches(transformer, X):
    Xt_fit = transformer.fit_transform(X)
    Xt_separate = transformer.transform(X)
    assert np.allclose(Xt_fit, Xt_separate)


def test_target_encoder_fit_transform():
    X = np.array(["a", "b", "a"])
    y = np.array([0.0, 1.0, 1.0])
    encoder = TargetEncoder()
    transformed = encoder.fit_transform(X, y)
    assert transformed.shape == (3, 1)
    again = encoder.transform(X)
    assert np.allclose(transformed, again)


def test_check_array_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        check_array([[np.nan]])
    with pytest.raises(ValueError):
        check_array([1.0, 2.0])
    with pytest.raises(ValueError):
        check_array([[1.0 + 1.0j]])


def test_utils_helpers_raise():
    with pytest.raises(ValueError):
        ensure_same_shape(np.ones((2, 2)), expected_features=3)
    with pytest.raises(ValueError):
        validate_quantile_range((80.0, 20.0))


def test_transformer_base_behavior():
    class IdentityTransformer(Transformer):
        def transform(self, X: np.ndarray) -> np.ndarray:
            return check_array(X)

    transformer = IdentityTransformer()
    data = np.array([[1.0]])
    assert transformer.fit(data) is transformer
    assert np.array_equal(transformer.fit_transform(data), transformer.transform(data))
    with pytest.raises(NotImplementedError):
        transformer.inverse_transform(data)
    with pytest.raises(AttributeError):
        check_is_fitted(IdentityTransformer(), ("missing_",))


def test_dtype_promotion_to_float64():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    assert transformed.dtype == np.float64

