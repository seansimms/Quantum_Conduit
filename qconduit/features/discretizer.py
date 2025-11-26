"""Feature discretization transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import Transformer, check_is_fitted
from .utils import check_array, ensure_same_shape


@dataclass
class KBinsDiscretizer(Transformer):
    """Uniform or quantile based binning.

    Examples
    --------
    >>> from qconduit.features import KBinsDiscretizer
    >>> import numpy as np
    >>> X = np.linspace(0, 1, num=5).reshape(-1, 1)
    >>> KBinsDiscretizer(n_bins=3).fit_transform(X)
    array([[...]])
    """

    n_bins: int = 5
    strategy: Literal["uniform", "quantile"] = "uniform"
    encode: Literal["ordinal", "onehot"] = "ordinal"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KBinsDiscretizer":
        del y
        X_checked = check_array(X)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        if self.strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be 'uniform' or 'quantile'.")
        if self.encode not in {"ordinal", "onehot"}:
            raise ValueError("encode must be 'ordinal' or 'onehot'.")
        self.n_features_in_ = X_checked.shape[1]
        edges = []
        for feature_idx in range(self.n_features_in_):
            column = X_checked[:, feature_idx]
            if self.strategy == "uniform":
                min_val, max_val = column.min(), column.max()
                if min_val == max_val:
                    max_val = min_val + 1.0
                edges.append(np.linspace(min_val, max_val, self.n_bins + 1))
            else:
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges.append(np.percentile(column, quantiles, method="linear"))
        self.bin_edges_ = edges
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("bin_edges_", "n_features_in_"))
        X_checked = check_array(X)
        ensure_same_shape(X_checked, self.n_features_in_)
        bin_indices = np.zeros_like(X_checked, dtype=int)
        for feature_idx, edges in enumerate(self.bin_edges_):
            column = X_checked[:, feature_idx]
            inds = np.digitize(column, edges[1:-1], right=False)
            inds = np.clip(inds, 0, self.n_bins - 1)
            bin_indices[:, feature_idx] = inds
        if self.encode == "ordinal":
            return bin_indices
        encoded = []
        for feature_idx in range(self.n_features_in_):
            one_hot = np.zeros((X_checked.shape[0], self.n_bins), dtype=float)
            one_hot[np.arange(X_checked.shape[0]), bin_indices[:, feature_idx]] = 1.0
            encoded.append(one_hot)
        return np.concatenate(encoded, axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("bin_edges_", "n_features_in_"))
        X_checked = check_array(X)
        if self.encode == "ordinal":
            ensure_same_shape(X_checked, self.n_features_in_)
            bin_indices = X_checked.astype(int)
        else:
            n_samples = X_checked.shape[0]
            bin_indices = np.zeros((n_samples, self.n_features_in_), dtype=int)
            start = 0
            for feature_idx in range(self.n_features_in_):
                block = X_checked[:, start : start + self.n_bins]
                start += self.n_bins
                bin_indices[:, feature_idx] = block.argmax(axis=1)
        reconstructed = np.zeros_like(bin_indices, dtype=float)
        for feature_idx, edges in enumerate(self.bin_edges_):
            centers = 0.5 * (edges[:-1] + edges[1:])
            reconstructed[:, feature_idx] = centers[bin_indices[:, feature_idx]]
        return reconstructed

