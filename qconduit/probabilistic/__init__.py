"""Probabilistic inference tools: HMM, GMM, and particle filters.

This module provides textbook implementations of:
- Hidden Markov Models (HMM) with forward-backward, Viterbi, and Baum-Welch EM
- Gaussian Mixture Models (GMM) with EM
- Particle filters (Sequential Monte Carlo) for state-space models

All implementations are deterministic (given RNG seed), numerically stable,
and thoroughly tested.
"""

from .gmm import GaussianMixture
from .hmm import HiddenMarkovModel
from .particle import StateSpaceModel, bootstrap_particle_filter
from .utils import (
    effective_sample_size,
    log_normal_pdf,
    logsumexp,
    normal_pdf,
    normalize_log_weights,
    systematic_resample,
)

__all__ = [
    "HiddenMarkovModel",
    "GaussianMixture",
    "StateSpaceModel",
    "bootstrap_particle_filter",
    "logsumexp",
    "normal_pdf",
    "log_normal_pdf",
    "systematic_resample",
    "effective_sample_size",
    "normalize_log_weights",
]

