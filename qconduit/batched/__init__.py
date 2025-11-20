"""Batched and vectorized simulation utilities.

This module provides utilities for evaluating many circuits, parameter sets,
or states in a batched/vectorized way, analogous to PyTorch batch semantics.

The main components are:

- BatchedState: Container for batched pure statevectors with shape (B, 2**n).
- apply_circuit_to_batched_states: Apply a circuit to many states.
- apply_ansatz_batch_to_state: Apply many parameterized circuits to the same state.
- evaluate_expectations_batched_via_states: Compute expectations for many states.
- evaluate_expectations_for_params_batched: Compute expectations for many parameter sets.

Examples
--------
Batched evaluation of ansatz parameters:

    >>> import torch
    >>> from qconduit.batched import evaluate_expectations_for_params_batched
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.operators import PauliSum
    >>>
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
    >>> params_batch = torch.linspace(0.0, 3.14, steps=10).unsqueeze(1).repeat(
    ...     1, ansatz.num_parameters
    ... )
    >>> H = PauliSum.from_label("ZZ")
    >>> energies = evaluate_expectations_for_params_batched(ansatz, params_batch, H)
    >>> print(energies.shape)  # (10,)

Batched circuit application:

    >>> import torch
    >>> from qconduit.circuit import QuantumCircuit
    >>> from qconduit.batched import BatchedState, apply_circuit_to_batched_states
    >>>
    >>> circuit = QuantumCircuit(n_qubits=1)
    >>> circuit.add_gate("H", [0])
    >>> states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    >>> batched = BatchedState(states, n_qubits=1)
    >>> result = apply_circuit_to_batched_states(circuit, batched)

Limitations
-----------
- Memory heuristic: Vectorized paths are used when B * dim * dim <= 1e8 elements.
  For larger batches or dimensions, the code falls back to per-row loops.
- CPU-only: All operations are CPU-only but device-aware using default_device().
- Deterministic: All operations are deterministic and reproducible.
"""

from .apply import (
    MAX_VECTORIZE_ELEMENTS,
    apply_ansatz_batch_to_state,
    apply_circuit_to_batched_states,
    apply_circuits_batch_to_states,
    batched_build_circuits_from_params,
)
from .evaluate import (
    evaluate_expectations_batched_via_states,
    evaluate_expectations_for_params_batched,
)
from .state import BatchedState

__all__ = [
    "BatchedState",
    "apply_circuit_to_batched_states",
    "apply_ansatz_batch_to_state",
    "batched_build_circuits_from_params",
    "apply_circuits_batch_to_states",
    "evaluate_expectations_batched_via_states",
    "evaluate_expectations_for_params_batched",
    "MAX_VECTORIZE_ELEMENTS",
]

