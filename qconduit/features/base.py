"""Base classes for feature transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Transformer(ABC):
    """Minimal sklearn-style transformer interface."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Transformer":
        """Fit transformer to data."""
        del X, y  # unused by default
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the transformation to X."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform in a single call."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transformation."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse_transform.")


def check_is_fitted(instance: Any, attributes: tuple[str, ...]) -> None:
    """Ensure transformer has been fitted."""
    missing = [attr for attr in attributes if not hasattr(instance, attr)]
    if missing:
        raise AttributeError(
            f"{instance.__class__.__name__} instance is not fitted yet. "
            f"Missing attributes: {missing}"
        )

