"""Quantum Conduit - a PyTorch-native quantum statevector plumbing library."""

__version__ = "0.0.1"

# Core abstractions
from .core import Device, device, default_device, QuantumModule

# Gates
from .gates import (
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    RX,
    RY,
    RZ,
    is_unitary,
)

# Backend operations
from .backend import (
    zero_state,
    apply_gate,
    apply_two_qubit_gate,
    measure_expectation_z,
    measure_probs,
    zero_dm_state,
    dm_from_statevector,
    apply_kraus_single_qubit,
    measure_probs_dm,
    measure_expectation_z_dm,
)

# Operators
from .operators import (
    PauliTerm,
    PauliSum,
    expectation_pauli_term,
    expectation_pauli_sum,
    expectation_pauli_sum_dm,
)

# Noise models
from .noise import (
    NoiseModel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)

# Gradients
from .grad import param_shift_energy

# Circuit IR
from .circuit import GateOp, QuantumCircuit

# Layers
from .layers import HardwareEfficientAnsatz, ParametricAnsatz, QuantumBlock

# Algorithms
from .algorithms import VQE

# Diagnostics
from .diagnostics import (
    state_norm,
    assert_normalized,
    is_hermitian,
    assert_hermitian,
    fidelity,
    bloch_vector,
    is_debug_enabled,
    set_debug_enabled,
    debug_context,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Device",
    "device",
    "default_device",
    "QuantumModule",
    # Gates
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "CNOT",
    "RX",
    "RY",
    "RZ",
    "is_unitary",
    # Backend
    "zero_state",
    "apply_gate",
    "apply_two_qubit_gate",
    "measure_expectation_z",
    "measure_probs",
    "zero_dm_state",
    "dm_from_statevector",
    "apply_kraus_single_qubit",
    "measure_probs_dm",
    "measure_expectation_z_dm",
    # Operators
    "PauliTerm",
    "PauliSum",
    "expectation_pauli_term",
    "expectation_pauli_sum",
    "expectation_pauli_sum_dm",
    # Gradients
    "param_shift_energy",
    # Noise
    "NoiseModel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    # Circuit IR
    "GateOp",
    "QuantumCircuit",
    # Layers
    "ParametricAnsatz",
    "HardwareEfficientAnsatz",
    "QuantumBlock",
    # Algorithms
    "VQE",
    # Diagnostics
    "state_norm",
    "assert_normalized",
    "is_hermitian",
    "assert_hermitian",
    "fidelity",
    "bloch_vector",
    "is_debug_enabled",
    "set_debug_enabled",
    "debug_context",
]
