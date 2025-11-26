"""Polynomial feature expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
from typing import Sequence

import numpy as np

from .base import Transformer, check_is_fitted
from .utils import check_array


@dataclass
class PolynomialFeatures(Transformer):
    """Generate polynomial and interaction features.

    Examples
    --------
    >>> from qconduit.features import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0]])
    >>> PolynomialFeatures(degree=2).fit_transform(X)
    array([[...]])
    """

    degree: int = 2
    include_bias: bool = True
    interaction_only: bool = False
    powers_: list[np.ndarray] = field(init=False, default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PolynomialFeatures":
        del y
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")
        X_checked = check_array(X)
        self.n_features_in_ = X_checked.shape[1]
        self.powers_ = self._generate_powers()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("powers_", "n_features_in_"))
        X_checked = check_array(X)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_checked.shape[1]}."
            )
        n_samples = X_checked.shape[0]
        X_out = np.ones((n_samples, len(self.powers_)), dtype=np.float64)
        for idx, power in enumerate(self.powers_):
            if np.all(power == 0):
                continue
            X_out[:, idx] = np.prod(np.power(X_checked, power), axis=1)
        return X_out

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        check_is_fitted(self, ("powers_", "n_features_in_"))
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if len(input_features) != self.n_features_in_:
            raise ValueError("input_features length mismatch.")
        names: list[str] = []
        for power in self.powers_:
            term_parts = []
            for feature_idx, exponent in enumerate(power):
                if exponent == 0:
                    continue
                name = input_features[feature_idx]
                if exponent > 1:
                    term_parts.append(f"{name}^{exponent}")
                else:
                    term_parts.append(name)
            if not term_parts:
                names.append("1")
            else:
                names.append(" ".join(term_parts))
        return names

    def _generate_powers(self) -> list[np.ndarray]:
        powers: list[np.ndarray] = []
        if self.include_bias:
            powers.append(np.zeros(self.n_features_in_, dtype=int))
        comb_func = combinations if self.interaction_only else combinations_with_replacement
        for degree in range(1, self.degree + 1):
            for combo in comb_func(range(self.n_features_in_), degree):
                power = np.zeros(self.n_features_in_, dtype=int)
                for idx in combo:
                    power[idx] += 1
                powers.append(power)
        return powers

