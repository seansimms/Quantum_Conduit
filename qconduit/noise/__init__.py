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
    identity_channel,
)
from .density import (
    apply_kraus_channel_to_density_matrix,
    apply_kraus_channel_to_statevector,
    compose_kraus_channels,
    to_density_matrix,
)
from .kraus import (
    KrausChannel,
    amplitude_damping_channel,
    bit_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    generalized_amplitude_damping_channel,
    phase_damping_channel,
    phase_flip_channel,
    two_qubit_depolarizing_channel,
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
    "identity_channel",
    "NoiseConfig",
    "simulate_noisy_circuit_dm",
    "sample_noisy_circuit_dm",
    # New G11 APIs (KrausChannel-based)
    "KrausChannel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "depolarizing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "generalized_amplitude_damping_channel",
    "two_qubit_depolarizing_channel",
    "to_density_matrix",
    "apply_kraus_channel_to_density_matrix",
    "apply_kraus_channel_to_statevector",
    "compose_kraus_channels",
]

