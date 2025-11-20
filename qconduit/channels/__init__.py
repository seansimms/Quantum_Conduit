"""Textbook quantum noise channels and quantum channels API.

This module provides standard quantum noise channels (Kraus operators) and
utilities to apply them to states and circuits. All implementations are
textbook-only, deterministic, and CPU-aware.

Limitations
-----------
- **Multi-qubit channel extension**: The `KrausChannel.tensor_extend()` method
  currently only supports single-qubit channels. For channels acting on multiple
  qubits, construct the full-system Kraus operators manually or use the channel
  directly on a density matrix of the appropriate size.
"""

from .builtins import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    GeneralKraus,
    PhaseDampingChannel,
    PhaseFlipChannel,
)
from .circuit_integration import (
    NoisyCircuit,
    annotate_circuit_with_channels,
    apply_channel_schedule_to_state,
    apply_circuit_with_noise,
)
from .core import KrausChannel
from .utils import (
    density_from_statevector,
    is_density_matrix,
    statevector_from_density_sampling,
)

__all__ = [
    "KrausChannel",
    "DepolarizingChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "PhaseDampingChannel",
    "AmplitudeDampingChannel",
    "GeneralKraus",
    "density_from_statevector",
    "statevector_from_density_sampling",
    "is_density_matrix",
    "NoisyCircuit",
    "annotate_circuit_with_channels",
    "apply_circuit_with_noise",
    "apply_channel_schedule_to_state",
]

