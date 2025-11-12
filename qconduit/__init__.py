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
)

# Operators
from .operators import (
    PauliTerm,
    PauliSum,
    expectation_pauli_term,
    expectation_pauli_sum,
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
    # Operators
    "PauliTerm",
    "PauliSum",
    "expectation_pauli_term",
    "expectation_pauli_sum",
]
