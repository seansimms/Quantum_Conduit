"""Noise models for quantum channels.

This package provides generic quantum noise models implemented via standard
Kraus operator channels. All implementations are textbook quantum computing
techniques with no proprietary features.
"""

from .base import NoiseModel
from .channels import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    PhaseDampingChannel,
    SingleQubitChannel,
    amplitude_damping_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
)
from .noisy_circuit import (
    NoiseConfig,
    sample_noisy_circuit_dm,
    simulate_noisy_circuit_dm,
)

__all__ = [
    "NoiseModel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    "SingleQubitChannel",
    "depolarizing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "identity_channel",
    "NoiseConfig",
    "simulate_noisy_circuit_dm",
    "sample_noisy_circuit_dm",
]

