"""Gaussian Mixture Model (GMM) with Expectation-Maximization.

Implements EM algorithm for fitting GMMs with full or diagonal covariances.

References:
    Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
    Springer, Chapter 9.
"""

from typing import Optional

import numpy as np

from .utils import log_normal_pdf, logsumexp


class GaussianMixture:
    """Gaussian Mixture Model with EM parameter estimation.

    Supports full and diagonal covariance matrices. Uses log-domain computations
    for numerical stability.

    Attributes:
        n_components: Number of mixture components.
        covariance_type: "full" or "diag".
        means_: Component means, shape (n_components, n_features).
        covariances_: Component covariances, shape depends on covariance_type.
        weights_: Component weights, shape (n_components,).
    """

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        tol: float = 1e-6,
        max_iter: int = 100,
        reg_covar: float = 1e-6,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize GMM.

        Args:
            n_components: Number of mixture components.
            covariance_type: "full" or "diag".
            tol: Convergence tolerance for EM.
            max_iter: Maximum EM iterations.
            reg_covar: Regularization added to diagonal of covariances.
            rng: Random number generator for initialization. If None, uses default_rng(0).
        """
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type must be 'full' or 'diag', got {covariance_type}")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Parameters (set by fit)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.n_features_ = None

    def _logpdf_component(self, X: np.ndarray, k: int) -> np.ndarray:
        """Compute log PDF of each sample under component k.

        Args:
            X: Data, shape (n_samples, n_features).
            k: Component index.

        Returns:
            Log PDFs, shape (n_samples,).
        """
        mean = self.means_[k, :]
        cov = self.covariances_[k]

        if self.covariance_type == "diag":
            # Diagonal: efficient computation
            diff = X - mean
            log_det = np.sum(np.log(cov + 1e-12))
            quad_form = np.sum((diff**2) / (cov + 1e-12), axis=1)
            log_pdf = -0.5 * self.n_features_ * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form
        else:
            # Full covariance: use Cholesky-based log_normal_pdf
            log_pdf = np.array([log_normal_pdf(X[i, :], mean, cov) for i in range(len(X))])

        return log_pdf

        return log_pdf

    def _compute_log_responsibilities(self, X: np.ndarray) -> np.ndarray:
        """Compute log responsibilities (E-step).

        Args:
            X: Data, shape (n_samples, n_features).

        Returns:
            Log responsibilities, shape (n_samples, n_components).
        """
        n_samples = len(X)
        log_resp = np.zeros((n_samples, self.n_components))

        # Log weights
        log_weights = np.log(self.weights_ + 1e-12)

        # For each component, compute log PDF
        for k in range(self.n_components):
            log_resp[:, k] = log_weights[k] + self._logpdf_component(X, k)

        # Normalize: subtract log-sum-exp per sample
        log_resp_norm = logsumexp(log_resp, axis=1)
        log_resp = log_resp - log_resp_norm[:, np.newaxis]

        return log_resp

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize GMM parameters deterministically.

        Uses deterministic initialization: sort data lexicographically and
        take evenly spaced points as initial means.

        Args:
            X: Data, shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Deterministic initialization: sort and take evenly spaced points
        # Sort by first feature, then second, etc.
        sorted_indices = np.lexsort([X[:, i] for i in range(n_features - 1, -1, -1)])
        step = max(1, n_samples // self.n_components)
        init_indices = sorted_indices[::step][: self.n_components]

        # Initialize means
        self.means_ = X[init_indices, :].copy()

        # Initialize covariances: use global covariance scaled by component
        global_cov = np.cov(X.T)
        # Handle case where X is 1D (n_samples, 1) -> np.cov returns scalar or 1D
        if global_cov.ndim == 0:
            # Scalar case: single feature
            global_cov = np.array([[global_cov]])
        elif global_cov.ndim == 1:
            # 1D case: single feature
            global_cov = global_cov.reshape(1, 1)

        if self.covariance_type == "diag":
            # Extract diagonal from covariance matrix
            diag_vals = np.diag(global_cov)
            self.covariances_ = diag_vals * np.ones((self.n_components, n_features))
        else:
            self.covariances_ = np.tile(global_cov, (self.n_components, 1, 1))

        # Add regularization
        if self.covariance_type == "diag":
            self.covariances_ += self.reg_covar
        else:
            for k in range(self.n_components):
                self.covariances_[k] += self.reg_covar * np.eye(n_features)

        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        """Fit GMM to data using EM algorithm.

        Args:
            X: Data, shape (n_samples, n_features).

        Returns:
            self (for chaining).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if len(X) < self.n_components:
            raise ValueError(f"Need at least {self.n_components} samples, got {len(X)}")

        # Initialize
        self._initialize_parameters(X)

        prev_log_likelihood = -np.inf
        log_likelihood_history = []

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            log_resp = self._compute_log_responsibilities(X)
            resp = np.exp(log_resp)

            # Compute log-likelihood
            log_likelihood = np.sum(
                logsumexp(log_resp + np.log(self.weights_ + 1e-12)[np.newaxis, :], axis=1)
            )
            log_likelihood_history.append(log_likelihood)

            # Check convergence
            if iteration > 0:
                improvement = log_likelihood - prev_log_likelihood
                if improvement < self.tol:
                    break

            prev_log_likelihood = log_likelihood

            # M-step: update parameters
            # Weights
            resp_sum = np.sum(resp, axis=0)
            self.weights_ = resp_sum / len(X)
            self.weights_ = self.weights_ + 1e-12  # Regularization
            self.weights_ = self.weights_ / np.sum(self.weights_)

            # Means
            for k in range(self.n_components):
                self.means_[k, :] = np.sum(resp[:, k : k + 1] * X, axis=0) / (resp_sum[k] + 1e-12)

            # Covariances
            for k in range(self.n_components):
                diff = X - self.means_[k, :]
                resp_k = resp[:, k]

                if self.covariance_type == "diag":
                    # Diagonal: element-wise
                    cov_k = (
                        np.sum(resp_k[:, np.newaxis] * (diff**2), axis=0)
                        / (resp_sum[k] + 1e-12)
                    )
                    cov_k += self.reg_covar
                    self.covariances_[k, :] = cov_k
                else:
                    # Full: matrix
                    weighted_diff = np.sqrt(resp_k)[:, np.newaxis] * diff
                    cov_k = np.dot(weighted_diff.T, weighted_diff) / (resp_sum[k] + 1e-12)
                    cov_k += self.reg_covar * np.eye(self.n_features_)
                    # Ensure positive definite
                    eigvals = np.linalg.eigvals(cov_k)
                    if np.any(eigvals <= 0):
                        cov_k += (
                            np.abs(np.min(eigvals)) + self.reg_covar
                        ) * np.eye(self.n_features_)
                    self.covariances_[k, :, :] = cov_k

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute component assignment probabilities (responsibilities).

        Args:
            X: Data, shape (n_samples, n_features).

        Returns:
            Responsibilities, shape (n_samples, n_components).
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        log_resp = self._compute_log_responsibilities(X)
        return np.exp(log_resp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict component assignments (hard clustering).

        Args:
            X: Data, shape (n_samples, n_features).

        Returns:
            Component labels, shape (n_samples,).
        """
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)

    def score(self, X: np.ndarray) -> float:
        """Compute average log-likelihood per sample.

        Args:
            X: Data, shape (n_samples, n_features).

        Returns:
            Average log-likelihood.
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        log_resp = self._compute_log_responsibilities(X)
        log_likelihood = np.sum(
            logsumexp(log_resp + np.log(self.weights_ + 1e-12)[np.newaxis, :], axis=1)
        )
        return log_likelihood / len(X)

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from the fitted GMM.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator. If None, uses self.rng.

        Returns:
            Samples, shape (n_samples, n_features).

        Raises:
            ValueError: If model not fitted.
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if rng is None:
            rng = self.rng

        # Sample component assignments
        component_samples = rng.choice(self.n_components, size=n_samples, p=self.weights_)

        # Sample from each component
        samples = np.zeros((n_samples, self.n_features_))
        for k in range(self.n_components):
            mask = component_samples == k
            n_k = np.sum(mask)
            if n_k > 0:
                mean = self.means_[k, :]
                if self.covariance_type == "diag":
                    cov = self.covariances_[k, :]
                    samples[mask, :] = rng.multivariate_normal(mean, np.diag(cov), size=n_k)
                else:
                    cov = self.covariances_[k, :, :]
                    samples[mask, :] = rng.multivariate_normal(mean, cov, size=n_k)

        return samples

