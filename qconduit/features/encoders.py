"""Categorical encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .base import Transformer, check_is_fitted
from .utils import check_array, rng_default


@dataclass
class OneHotEncoder(Transformer):
    """Dense one-hot encoder with sklearn-like semantics.

    Examples
    --------
    >>> from qconduit.features import OneHotEncoder
    >>> import numpy as np
    >>> X = np.array([["red"], ["blue"], ["red"]])
    >>> OneHotEncoder().fit_transform(X)
    array([[...]])
    """

    categories: Literal["auto"] | Sequence[Sequence[object]] = "auto"
    drop: Literal[None, "first"] = None
    sparse: bool = False  # retained for API parity

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OneHotEncoder":
        del y
        X_checked = check_array(X, dtype=None, allow_object=True)
        X_checked = np.asarray(X_checked, dtype=object)
        self.n_features_in_ = X_checked.shape[1]
        if self.categories == "auto":
            categories = [np.unique(X_checked[:, i]) for i in range(self.n_features_in_)]
        else:
            if len(self.categories) != self.n_features_in_:
                raise ValueError("categories length must match n_features.")
            categories = [np.array(cat, dtype=object) for cat in self.categories]
        for idx, cat in enumerate(categories):
            if cat.size == 0:
                raise ValueError(f"Feature {idx} has no categories.")
        self.categories_ = [np.array(sorted(cat.tolist())) for cat in categories]
        self.drop_idx_ = None
        if self.drop == "first":
            self.drop_idx_ = [0] * self.n_features_in_
        elif self.drop is not None:
            raise ValueError("drop must be None or 'first'.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("categories_", "n_features_in_"))
        X_checked = check_array(X, dtype=None, allow_object=True)
        X_checked = np.asarray(X_checked, dtype=object)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError("Feature mismatch during transform.")
        encoded_columns = []
        for feature_idx, categories in enumerate(self.categories_):
            column = X_checked[:, feature_idx]
            cat_to_index = {cat: idx for idx, cat in enumerate(categories)}
            try:
                indices = np.vectorize(cat_to_index.__getitem__)(column)
            except KeyError as exc:
                msg = f"Unknown category {exc.args[0]!r} in feature {feature_idx}."
                raise ValueError(msg) from exc
            n_categories = len(categories)
            drop_idx = None if self.drop_idx_ is None else self.drop_idx_[feature_idx]
            n_active = n_categories - (1 if drop_idx is not None else 0)
            one_hot = np.zeros((X_checked.shape[0], n_active), dtype=float)
            for sample_idx, cat_index in enumerate(indices):
                if drop_idx is not None and cat_index == drop_idx:
                    continue
                if drop_idx is None:
                    adjusted_index = cat_index
                else:
                    adjusted_index = cat_index if cat_index < drop_idx else cat_index - 1
                one_hot[sample_idx, adjusted_index] = 1.0
            encoded_columns.append(one_hot)
        return np.concatenate(encoded_columns, axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("categories_", "n_features_in_"))
        X_checked = check_array(X)
        n_samples = X_checked.shape[0]
        outputs = np.empty((n_samples, self.n_features_in_), dtype=object)
        start = 0
        for feature_idx, categories in enumerate(self.categories_):
            drop_idx = None if self.drop_idx_ is None else self.drop_idx_[feature_idx]
            width = len(categories) - (1 if drop_idx is not None else 0)
            feature_block = X_checked[:, start : start + width]
            start += width
            if drop_idx is None:
                category_indices = feature_block.argmax(axis=1)
            else:
                category_indices = np.full(n_samples, drop_idx, dtype=int)
                non_zero = feature_block.argmax(axis=1)
                has_signal = feature_block.max(axis=1) > 0
                category_indices[has_signal] = non_zero[has_signal]
                category_indices[has_signal & (non_zero >= drop_idx)] += 1
            outputs[:, feature_idx] = categories[category_indices]
        return outputs


@dataclass
class TargetEncoder(Transformer):
    """Mean target encoder with additive smoothing.

    Examples
    --------
    >>> from qconduit.features import TargetEncoder
    >>> import numpy as np
    >>> X = np.array(["a", "b", "a"])
    >>> y = np.array([0.0, 1.0, 1.0])
    >>> TargetEncoder().fit_transform(X, y)
    array([[...]])
    """

    smoothing: float = 1.0
    min_samples_leaf: int = 1
    noise: float = 0.0
    random_state: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TargetEncoder":
        if y is None:
            raise ValueError("TargetEncoder requires target values.")
        X_checked = check_array(X, ensure_2d=False, dtype=None, allow_object=True)
        X_flat = np.asarray(X_checked).reshape(-1)
        y_array = check_array(y, ensure_2d=False).astype(np.float64)
        if X_flat.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.categories_, counts = np.unique(X_flat, return_counts=True)
        self.global_mean_ = float(y_array.mean())
        cat_means = []
        for cat in self.categories_:
            mask = X_flat == cat
            cat_means.append(float(y_array[mask].mean()))
        cat_means = np.array(cat_means)
        effective_counts = np.maximum(counts, self.min_samples_leaf)
        numerator = effective_counts * cat_means + self.smoothing * self.global_mean_
        denominator = effective_counts + self.smoothing
        self.encoding_ = numerator / denominator
        self.n_features_in_ = 1
        self.rng_ = rng_default(self.random_state)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("categories_", "encoding_", "global_mean_"))
        X_checked = check_array(X, ensure_2d=False, dtype=None, allow_object=True)
        X_flat = np.asarray(X_checked).reshape(-1)
        encodings = np.full_like(X_flat, fill_value=self.global_mean_, dtype=np.float64)
        cat_to_value = dict(zip(self.categories_, self.encoding_))
        for idx, value in enumerate(X_flat):
            if value in cat_to_value:
                encodings[idx] = cat_to_value[value]
        if self.noise > 0.0:
            noise = self.rng_.normal(loc=0.0, scale=self.noise, size=encodings.shape)
            encodings = encodings + noise
        return encodings.reshape(-1, 1)

