"""Classical feature scaling transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Transformer, check_is_fitted
from .utils import check_array, ensure_same_shape, validate_quantile_range


@dataclass
class StandardScaler(Transformer):
    """Standardize features by removing mean and scaling to unit variance.

    Examples
    --------
    >>> from qconduit.features import StandardScaler
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0], [2.0, 0.0]])
    >>> scaler = StandardScaler()
    >>> scaler.fit_transform(X)
    array([[...]])
    """

    with_mean: bool = True
    with_std: bool = True
    ddof: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "StandardScaler":
        del y
        X_checked = check_array(X)
        self.n_features_in_ = X_checked.shape[1]
        self.mean_ = X_checked.mean(axis=0) if self.with_mean else np.zeros(self.n_features_in_)
        variance = X_checked.var(axis=0, ddof=self.ddof)
        self.scale_ = np.where(variance == 0.0, 1.0, np.sqrt(variance))
        if not self.with_std:
            self.scale_ = np.ones_like(self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("mean_", "scale_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_transformed = X_checked.astype(np.float64, copy=True)
        if self.with_mean:
            X_transformed -= self.mean_
        if self.with_std:
            X_transformed /= self.scale_
        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("mean_", "scale_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_inversed = X_checked.astype(np.float64, copy=True)
        if self.with_std:
            X_inversed *= self.scale_
        if self.with_mean:
            X_inversed += self.mean_
        return X_inversed


@dataclass
class RobustScaler(Transformer):
    """Scale features using statistics robust to outliers.

    Examples
    --------
    >>> from qconduit.features import RobustScaler
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [10.0]])
    >>> RobustScaler().fit_transform(X)
    array([[...]])
    """

    with_centering: bool = True
    with_scaling: bool = True
    quantile_range: tuple[float, float] = (25.0, 75.0)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RobustScaler":
        del y
        X_checked = check_array(X)
        self.n_features_in_ = X_checked.shape[1]
        q_min, q_max = validate_quantile_range(self.quantile_range)
        self.center_ = np.median(X_checked, axis=0) if self.with_centering else np.zeros(
            self.n_features_in_
        )
        q1 = np.percentile(X_checked, q_min, axis=0, method="linear")
        q3 = np.percentile(X_checked, q_max, axis=0, method="linear")
        scale = q3 - q1
        scale[scale == 0.0] = 1.0
        self.scale_ = scale if self.with_scaling else np.ones_like(scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("center_", "scale_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_transformed = X_checked.astype(np.float64, copy=True)
        if self.with_centering:
            X_transformed -= self.center_
        if self.with_scaling:
            X_transformed /= self.scale_
        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("center_", "scale_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_inversed = X_checked.astype(np.float64, copy=True)
        if self.with_scaling:
            X_inversed *= self.scale_
        if self.with_centering:
            X_inversed += self.center_
        return X_inversed


@dataclass
class MinMaxScaler(Transformer):
    """Transform features to a given feature range.

    Examples
    --------
    >>> from qconduit.features import MinMaxScaler
    >>> import numpy as np
    >>> X = np.array([[0.0], [5.0]])
    >>> MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    array([[...]])
    """

    feature_range: tuple[float, float] = (0.0, 1.0)
    clip: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MinMaxScaler":
        del y
        X_checked = check_array(X)
        self.n_features_in_ = X_checked.shape[1]
        feature_min, feature_max = self.feature_range
        if feature_min >= feature_max:
            raise ValueError("feature_range min must be less than max.")
        self.data_min_ = X_checked.min(axis=0)
        self.data_max_ = X_checked.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0.0] = 1.0
        self.data_range_ = data_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        self.feature_range_ = (feature_min, feature_max)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("data_min_", "data_max_", "scale_", "min_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_transformed = X_checked * self.scale_ + self.min_
        if self.clip:
            feature_min, feature_max = self.feature_range_
            X_transformed = np.clip(X_transformed, feature_min, feature_max)
        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("data_min_", "scale_", "min_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        return (X_checked - self.min_) / self.scale_

