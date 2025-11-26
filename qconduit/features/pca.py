"""Principal Component Analysis implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Transformer, check_is_fitted
from .utils import check_array, ensure_same_shape


@dataclass
class PCA(Transformer):
    """SVD-based PCA transformer.

    Examples
    --------
    >>> from qconduit.features import PCA, StandardScaler
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    >>> Xs = StandardScaler().fit_transform(X)
    >>> PCA(n_components=1).fit_transform(Xs)
    array([[...]])
    """

    n_components: int | float | None = None
    whiten: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCA":
        del y
        X_checked = check_array(X)
        n_samples, n_features = X_checked.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")
        self.mean_ = X_checked.mean(axis=0)
        X_centered = X_checked - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.n_samples_, self.n_features_in_ = n_samples, n_features
        self.components_ = Vt
        self.singular_values_ = S
        total_var = (S**2) / (n_samples - 1)
        self.explained_variance_ = total_var
        total_variance_sum = total_var.sum()
        self.explained_variance_ratio_ = np.divide(
            total_var,
            total_variance_sum if total_variance_sum > 0 else 1.0,
        )
        n_selected = self._resolve_n_components(total_var)
        self.n_components_ = n_selected
        self.components_ = self.components_[:n_selected]
        self.explained_variance_ = self.explained_variance_[:n_selected]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_selected]
        self.singular_values_ = self.singular_values_[:n_selected]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self, ("components_", "mean_", "explained_variance_", "n_features_in_")
        )
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        X_centered = X_checked - self.mean_
        X_transformed = X_centered @ self.components_.T
        if self.whiten:
            scaling = np.sqrt(self.explained_variance_)
            scaling[scaling == 0.0] = 1.0
            X_transformed /= scaling
        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self, ("components_", "mean_", "explained_variance_", "n_features_in_")
        )
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.components_.shape[0])
        X_reconstructed = X_checked.astype(np.float64, copy=True)
        if self.whiten:
            scaling = np.sqrt(self.explained_variance_)
            scaling[scaling == 0.0] = 1.0
            X_reconstructed *= scaling
        X_reconstructed = X_reconstructed @ self.components_
        return X_reconstructed + self.mean_

    def _resolve_n_components(self, explained_variance: np.ndarray) -> int:
        n_total = explained_variance.shape[0]
        if self.n_components is None:
            return n_total
        if isinstance(self.n_components, int):
            if not 1 <= self.n_components <= n_total:
                raise ValueError(
                    f"n_components={self.n_components} is out of bounds for {n_total} features."
                )
            return self.n_components
        if isinstance(self.n_components, float):
            if not 0.0 < self.n_components < 1.0:
                raise ValueError("Float n_components must be in (0, 1).")
            total = np.sum(explained_variance)
            if total == 0.0:
                return 1
            ratio = np.cumsum(explained_variance) / total
            return int(np.searchsorted(ratio, self.n_components) + 1)
        raise TypeError("n_components must be None, int, or float.")

