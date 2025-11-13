"""Noise models for quantum channels.

This package provides generic quantum noise models implemented via standard
Kraus operator channels. All implementations are textbook quantum computing
techniques with no proprietary features.
"""

from .base import NoiseModel
from .channels import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)

__all__ = [
    "NoiseModel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
]

