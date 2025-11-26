"""Batched circuit application utilities."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from qconduit.batched.state import BatchedState
from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device

# Maximum number of elements for vectorized unitary computation
# Conservative threshold: B * dim * dim <= MAX_VECTORIZE_ELEMENTS
MAX_VECTORIZE_ELEMENTS = int(1e8)


def _circuit_to_unitary(
    circuit: QuantumCircuit,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Compute the dense unitary matrix representing a QuantumCircuit.

    This is done by applying the circuit to each computational basis state
    and collecting the results as columns of the unitary matrix.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to convert to unitary.
    device : torch.device, optional
        Device for computation. If None, uses default_device().
    dtype : torch.dtype, optional
        Complex dtype for the unitary. Default is torch.complex128.

    Returns
    -------
    torch.Tensor
        Unitary matrix of shape (dim, dim) where dim = 2**n_qubits.
    """
    if device is None:
        qdevice = default_device()
        device = qdevice.as_torch_device()

    n_qubits = circuit.n_qubits
    dim = 2**n_qubits

    # Import here to avoid circular imports
    from qconduit.variational.vqe import _apply_circuit_to_statevector

    # Build unitary by applying circuit to each basis state
    unitary = torch.zeros((dim, dim), dtype=dtype, device=device)

    qdevice = Device(
        name="custom",
        torch_device=device,
        dtype=torch.float64,
        complex_dtype=dtype,
    )

    for i in range(dim):
        # Create basis state |i⟩
        basis_state = torch.zeros(dim, dtype=dtype, device=device)
        basis_state[i] = 1.0 + 0.0j

        # Apply circuit
        output_state = _apply_circuit_to_statevector(circuit, basis_state, device=qdevice)

        # Store as column i of unitary (U|i⟩ = output_state)
        unitary[:, i] = output_state

    return unitary


def apply_circuit_to_batched_states(
    circuit: QuantumCircuit, batched: BatchedState
) -> BatchedState:
    """
    Apply a QuantumCircuit to a batch of states.

    This function supports two execution paths:
    1. Vectorized path: If the circuit can be converted to a dense unitary
       matrix (when memory allows), uses efficient batched matrix multiplication.
    2. Fallback path: Loops over batch elements and applies the circuit
       gate-by-gate to each state.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to apply. Must act on batched.n_qubits qubits.
    batched : BatchedState
        Batch of input states with shape (B, dim).

    Returns
    -------
    BatchedState
        Batch of output states with shape (B, dim).

    Examples
    --------
    >>> import torch
    >>> from qconduit.circuit import QuantumCircuit
    >>> from qconduit.batched import BatchedState, apply_circuit_to_batched_states
    >>> # Create a simple H gate circuit
    >>> circuit = QuantumCircuit(n_qubits=1)
    >>> circuit.add_gate("H", [0])
    >>> # Create batch of states
    >>> states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    >>> batched = BatchedState(states, n_qubits=1)
    >>> # Apply circuit
    >>> result = apply_circuit_to_batched_states(circuit, batched)
    """
    if circuit.n_qubits != batched.n_qubits:
        raise ValueError(
            f"circuit.n_qubits={circuit.n_qubits} does not match "
            f"batched.n_qubits={batched.n_qubits}"
        )

    batch_size, dim = batched.states.shape
    device = batched.states.device
    dtype = batched.states.dtype

    # Check if we can use vectorized path
    memory_elements = batch_size * dim * dim
    use_vectorized = memory_elements <= MAX_VECTORIZE_ELEMENTS

    if use_vectorized:
        try:
            # Compute unitary matrix
            unitary = _circuit_to_unitary(circuit, device=device, dtype=dtype)

            # Apply unitary: states_out = states @ U.conj().T
            # For quantum circuits, U|psi⟩ means we compute U @ psi
            # But since states are row vectors, we do: states @ U.T
            # Actually, we want U|psi⟩, so for row vectors: (U @ psi.T).T = psi @ U.T
            # But U is column-major (U[:, i] = U|i⟩), so U.T gives row-major
            # For batched: (B, dim) @ (dim, dim) -> (B, dim)
            states_out = batched.states @ unitary.T

            return BatchedState(states=states_out, n_qubits=batched.n_qubits)
        except Exception:
            # Fallback to loop if vectorization fails
            use_vectorized = False

    if not use_vectorized:
        # Fallback: loop over batch
        from qconduit.variational.vqe import _apply_circuit_to_statevector

        qdevice = Device(
            name="custom",
            torch_device=device,
            dtype=torch.float64,
            complex_dtype=dtype,
        )

        out_rows = []
        for i in range(batch_size):
            psi = batched.states[i]
            psi_out = _apply_circuit_to_statevector(circuit, psi, device=qdevice)
            out_rows.append(psi_out)

        states_out = torch.stack(out_rows, dim=0)
        return BatchedState(states=states_out, n_qubits=batched.n_qubits)


def apply_ansatz_batch_to_state(
    ansatz_factory: Callable[[torch.Tensor], QuantumCircuit],
    params_batch: torch.Tensor,
    initial_state: torch.Tensor,
) -> BatchedState:
    """
    Apply many parameterized circuits (from an ansatz factory) to the same initial state.

    This builds a circuit for each parameter vector in params_batch and applies
    it to initial_state, returning a batch of output states.

    Parameters
    ----------
    ansatz_factory : Callable[[torch.Tensor], QuantumCircuit]
        Function that takes a 1D parameter tensor and returns a QuantumCircuit.
        Typically this is ansatz.build_circuit wrapped: lambda p: ansatz.build_circuit(p).
    params_batch : torch.Tensor
        Batch of parameter vectors with shape (B, d) where d is the number of
        parameters per circuit.
    initial_state : torch.Tensor
        Initial statevector of shape (dim,) where dim = 2**n_qubits.

    Returns
    -------
    BatchedState
        Batch of output states with shape (B, dim).

    Examples
    --------
    >>> import torch
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.batched import apply_ansatz_batch_to_state
    >>> from qconduit.backend.statevector import zero_state
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
    >>> params_batch = torch.randn(5, ansatz.num_parameters)
    >>> psi0 = zero_state(n_qubits=2, batch_shape=None)
    >>> result = apply_ansatz_batch_to_state(
    ...     lambda p: ansatz.build_circuit(p), params_batch, psi0
    ... )
    """
    if params_batch.ndim != 2:
        raise ValueError(
            f"params_batch must be 2D with shape (B, d), got shape {params_batch.shape}"
        )

    if initial_state.ndim != 1:
        raise ValueError(
            f"initial_state must be 1D with shape (dim,), got shape {initial_state.shape}"
        )

    batch_size, num_params = params_batch.shape
    dim = initial_state.shape[0]
    device = initial_state.device
    dtype = initial_state.dtype

    # Infer n_qubits from dimension
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"initial_state dimension {dim} is not a power of 2")

    # Check memory for vectorized path by estimating elements needed
    memory_elements = batch_size * dim * dim
    use_vectorized = memory_elements <= MAX_VECTORIZE_ELEMENTS

    if use_vectorized:
        try:
            # Build one circuit to check if we can compute unitaries
            test_params = params_batch[0]
            test_circuit = ansatz_factory(test_params)
            if test_circuit.n_qubits != n_qubits:
                raise ValueError(
                    f"ansatz_factory produces circuits with n_qubits={test_circuit.n_qubits}, "
                    f"but initial_state has dim={dim} (n_qubits={n_qubits})"
                )

            # Build all unitaries
            unitaries = []
            for i in range(batch_size):
                circuit = ansatz_factory(params_batch[i])
                unitary = _circuit_to_unitary(circuit, device=device, dtype=dtype)
                unitaries.append(unitary)

            # Stack unitaries: (B, dim, dim)
            unitary_batch = torch.stack(unitaries, dim=0)

            # Apply to initial_state: (B, dim, dim) @ (dim, 1) -> (B, dim, 1) -> (B, dim)
            psi_expanded = initial_state.unsqueeze(0).unsqueeze(-1)  # (1, dim, 1)
            states_out = (unitary_batch @ psi_expanded).squeeze(-1)

            return BatchedState(states=states_out, n_qubits=n_qubits)
        except Exception:
            # Fallback to loop
            use_vectorized = False

    if not use_vectorized:
        # Fallback: loop over batch
        from qconduit.variational.vqe import _apply_circuit_to_statevector

        qdevice = Device(
            name="custom",
            torch_device=device,
            dtype=torch.float64,
            complex_dtype=dtype,
        )

        out_rows = []
        for i in range(batch_size):
            circuit = ansatz_factory(params_batch[i])
            if circuit.n_qubits != n_qubits:
                raise ValueError(
                    f"Circuit at batch index {i} has n_qubits={circuit.n_qubits}, "
                    f"but initial_state has dim={dim} (n_qubits={n_qubits})"
                )
            psi_out = _apply_circuit_to_statevector(circuit, initial_state, device=qdevice)
            out_rows.append(psi_out)

        states_out = torch.stack(out_rows, dim=0)
        return BatchedState(states=states_out, n_qubits=n_qubits)


def batched_build_circuits_from_params(
    ansatz, params_batch: torch.Tensor
) -> Sequence[QuantumCircuit]:
    """
    Build a sequence of circuits, one per parameter vector in params_batch.

    Parameters
    ----------
    ansatz
        Ansatz object with a build_circuit(params) method.
    params_batch : torch.Tensor
        Batch of parameter vectors with shape (B, d).

    Returns
    -------
    Sequence[QuantumCircuit]
        List of B QuantumCircuit objects.

    Examples
    --------
    >>> import torch
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.batched import batched_build_circuits_from_params
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
    >>> params_batch = torch.randn(5, ansatz.num_parameters)
    >>> circuits = batched_build_circuits_from_params(ansatz, params_batch)
    >>> print(len(circuits))  # 5
    """
    if params_batch.ndim != 2:
        raise ValueError(
            f"params_batch must be 2D with shape (B, d), got shape {params_batch.shape}"
        )

    batch_size, num_params = params_batch.shape
    circuits = []
    for i in range(batch_size):
        params_row = params_batch[i]
        circuit = ansatz.build_circuit(params_row)
        circuits.append(circuit)

    return circuits


def apply_circuits_batch_to_states(
    circuits: Sequence[QuantumCircuit], states: torch.Tensor
) -> BatchedState:
    """
    Apply a batch of circuits to a batch of states.

    Each circuit[i] is applied to states[i], producing output states.

    Parameters
    ----------
    circuits : Sequence[QuantumCircuit]
        Sequence of B circuits.
    states : torch.Tensor
        Batch of states with shape (B, dim) or single state with shape (dim,).
        If single state, it is broadcast to all circuits.

    Returns
    -------
    BatchedState
        Batch of output states with shape (B, dim).

    Examples
    --------
    >>> import torch
    >>> from qconduit.circuit import QuantumCircuit
    >>> from qconduit.batched import apply_circuits_batch_to_states
    >>> circuits = [QuantumCircuit(1) for _ in range(3)]
    >>> for c in circuits:
    ...     c.add_gate("H", [0])
    >>> states = torch.randn(3, 2, dtype=torch.complex128)
    >>> result = apply_circuits_batch_to_states(circuits, states)
    """
    if len(circuits) == 0:
        raise ValueError("circuits must be non-empty")

    # Handle broadcasting: if states is 1D, broadcast to all circuits
    if states.ndim == 1:
        batch_size = len(circuits)
        dim = states.shape[0]
        states = states.unsqueeze(0).expand(batch_size, dim)
    elif states.ndim == 2:
        batch_size = states.shape[0]
        if batch_size != len(circuits):
            raise ValueError(
                f"states batch size {batch_size} does not match circuits length {len(circuits)}"
            )
    else:
        raise ValueError(
            f"states must be 1D (dim,) or 2D (B, dim), got shape {states.shape}"
        )

    dim = states.shape[1]
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"states dimension {dim} is not a power of 2")

    # Validate all circuits have same n_qubits
    for i, circuit in enumerate(circuits):
        if circuit.n_qubits != n_qubits:
            raise ValueError(
                f"circuit[{i}] has n_qubits={circuit.n_qubits}, "
                f"but states have dim={dim} (n_qubits={n_qubits})"
            )

    device = states.device
    dtype = states.dtype

    memory_elements = batch_size * dim * dim
    use_vectorized = memory_elements <= MAX_VECTORIZE_ELEMENTS

    if use_vectorized:
        try:
            # Build all unitaries
            unitaries = []
            for circuit in circuits:
                unitary = _circuit_to_unitary(circuit, device=device, dtype=dtype)
                unitaries.append(unitary)

            # Stack: (B, dim, dim)
            unitary_batch = torch.stack(unitaries, dim=0)

            # Apply: (B, dim, dim) @ (B, dim, 1)
            states_expanded = states.unsqueeze(-1)
            states_out = (unitary_batch @ states_expanded).squeeze(-1)

            return BatchedState(states=states_out, n_qubits=n_qubits)
        except Exception:
            # Fallback to loop
            use_vectorized = False

    if not use_vectorized:
        # Fallback: loop over batch
        from qconduit.variational.vqe import _apply_circuit_to_statevector

        qdevice = Device(
            name="custom",
            torch_device=device,
            dtype=torch.float64,
            complex_dtype=dtype,
        )

        out_rows = []
        for i in range(batch_size):
            circuit = circuits[i]
            psi = states[i]
            psi_out = _apply_circuit_to_statevector(circuit, psi, device=qdevice)
            out_rows.append(psi_out)

        states_out = torch.stack(out_rows, dim=0)
        return BatchedState(states=states_out, n_qubits=n_qubits)


__all__ = [
    "apply_circuit_to_batched_states",
    "apply_ansatz_batch_to_state",
    "batched_build_circuits_from_params",
    "apply_circuits_batch_to_states",
    "MAX_VECTORIZE_ELEMENTS",
]



