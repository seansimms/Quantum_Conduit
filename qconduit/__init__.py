"""Quantum Conduit - a PyTorch-native quantum statevector plumbing library."""

__version__ = "0.0.1"

# Core abstractions
# Algorithms
from .algorithms import VQE

# Backend operations
from .backend import (
    apply_gate,
    apply_kraus_single_qubit,
    apply_two_qubit_gate,
    dm_from_statevector,
    measure_expectation_z,
    measure_expectation_z_dm,
    measure_probs,
    measure_probs_dm,
    zero_dm_state,
    zero_state,
)

# Circuit IR
from .circuit import GateOp, QuantumCircuit
from .core import Device, QuantumModule, default_device, device

# Diagnostics
from .diagnostics import (
    assert_hermitian,
    assert_normalized,
    bloch_vector,
    debug_context,
    fidelity,
    is_debug_enabled,
    is_hermitian,
    set_debug_enabled,
    state_norm,
)

# Gates
from .gates import (
    CNOT,
    RX,
    RY,
    RZ,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    is_unitary,
)

# Gradients
from .grad import param_shift_energy

# Layers
from .layers import HardwareEfficientAnsatz, ParametricAnsatz, QuantumBlock

# Noise models
from .noise import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    NoiseModel,
    PhaseDampingChannel,
)

# Operators
from .operators import (
    PauliSum,
    PauliTerm,
    expectation_pauli_sum,
    expectation_pauli_sum_dm,
    expectation_pauli_term,
)

# Optimizers
from .optim import OptimConfig, create_optimizer

# Training
from .training import (
    EarlyStoppingConfig,
    TrainingCallback,
    TrainingHistory,
    TrainingStepInfo,
    VQETrainer,
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
    # Optimizers
    "OptimConfig",
    "create_optimizer",
    # Training
    "TrainingStepInfo",
    "TrainingHistory",
    "TrainingCallback",
    "EarlyStoppingConfig",
    "VQETrainer",
]
