"""Batched expectation evaluation utilities."""

from __future__ import annotations

from typing import Optional

import torch

from qconduit.batched.state import BatchedState
from qconduit.core.device import default_device
from qconduit.exact import paulisum_to_dense
from qconduit.operators import PauliSum
from qconduit.operators.expectation import expectation_pauli_sum


def evaluate_expectations_batched_via_states(
    hamiltonian: PauliSum, batched_states: BatchedState
) -> torch.Tensor:
    """
    Evaluate expectation values ⟨ψ_i|H|ψ_i⟩ for a batch of states.

    This function computes expectations efficiently using vectorized operations
    when possible. If the Hamiltonian can be converted to a dense matrix, it
    uses batched matrix multiplication. Otherwise, it falls back to evaluating
    expectations term-by-term.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hamiltonian operator. Must act on batched_states.n_qubits qubits.
    batched_states : BatchedState
        Batch of states with shape (B, dim).

    Returns
    -------
    torch.Tensor
        Real tensor of shape (B,) containing expectation values.

    Examples
    --------
    >>> import torch
    >>> from qconduit.operators import PauliSum
    >>> from qconduit.batched import BatchedState, evaluate_expectations_batched_via_states
    >>> H = PauliSum.from_label("Z")  # Single-qubit Z operator
    >>> states = torch.randn(5, 2, dtype=torch.complex128)
    >>> states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    >>> batched = BatchedState(states, n_qubits=1)
    >>> expectations = evaluate_expectations_batched_via_states(H, batched)
    >>> print(expectations.shape)  # (5,)
    """
    if hamiltonian.n_qubits() != 0 and hamiltonian.n_qubits() != batched_states.n_qubits:
        raise ValueError(
            f"hamiltonian.n_qubits()={hamiltonian.n_qubits()} does not match "
            f"batched_states.n_qubits={batched_states.n_qubits}"
        )

    B, dim = batched_states.states.shape
    device = batched_states.states.device
    dtype = batched_states.states.dtype

    # Try vectorized path using dense Hamiltonian matrix
    try:
        H_dense = paulisum_to_dense(
            hamiltonian,
            num_qubits=batched_states.n_qubits,
            device=device,
            dtype=dtype,
        )

        # Compute expectations: ⟨ψ_i|H|ψ_i⟩ = ψ_i^† H ψ_i
        # psi shape: (B, dim)
        # H_dense shape: (dim, dim)
        # For each row psi[i], compute psi[i].conj() @ H_dense @ psi[i]
        # Vectorized: (psi.conj() * (H_dense @ psi.T).T).sum(dim=1).real

        # Compute H @ psi^T: (dim, dim) @ (dim, B) -> (dim, B)
        H_psi = H_dense @ batched_states.states.T  # (dim, B)

        # Compute psi^† @ (H @ psi): element-wise multiply and sum
        # psi.conj(): (B, dim)
        # H_psi.T: (B, dim)
        expectations = (batched_states.states.conj() * H_psi.T).sum(dim=1).real  # (B,)

        return expectations.to(dtype=torch.float64)
    except Exception:
        # Fallback: loop over batch and use expectation_pauli_sum
        expectations_list = []
        for i in range(B):
            state = batched_states.states[i]
            exp_val = expectation_pauli_sum(state, hamiltonian)
            if exp_val.ndim == 0:
                expectations_list.append(exp_val.item())
            else:
                expectations_list.append(exp_val.item())

        return torch.tensor(expectations_list, dtype=torch.float64, device=device)


def evaluate_expectations_for_params_batched(
    ansatz,
    params_batch: torch.Tensor,
    hamiltonian: PauliSum,
    initial_state: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Evaluate expectation values for many parameter vectors.

    This is the main helper for batched VQE-style evaluations: given a batch
    of parameter vectors, compute the expectation ⟨ψ(θ_i)|H|ψ(θ_i)⟩ for each
    parameter set θ_i.

    Parameters
    ----------
    ansatz
        Ansatz object with build_circuit(params) method and num_qubits property.
    params_batch : torch.Tensor
        Batch of parameter vectors with shape (B, d) where d = ansatz.num_parameters.
    hamiltonian : PauliSum
        Hamiltonian operator. Must act on ansatz.num_qubits qubits.
    initial_state : torch.Tensor, optional
        Initial statevector of shape (dim,). If None, uses |0...0⟩.
    device : torch.device, optional
        Device for computation. If None, uses default_device().

    Returns
    -------
    torch.Tensor
        Real tensor of shape (B,) containing expectation values.

    Examples
    --------
    >>> import torch
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.operators import PauliSum
    >>> from qconduit.batched import evaluate_expectations_for_params_batched
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=1)
    >>> params_batch = torch.randn(10, ansatz.num_parameters)
    >>> H = PauliSum.from_label("ZZ")
    >>> energies = evaluate_expectations_for_params_batched(ansatz, params_batch, H)
    >>> print(energies.shape)  # (10,)
    """
    if params_batch.ndim != 2:
        raise ValueError(
            f"params_batch must be 2D with shape (B, d), got shape {params_batch.shape}"
        )

    B, d = params_batch.shape
    n_qubits = ansatz.num_qubits

    if d != ansatz.num_parameters:
        raise ValueError(
            f"params_batch has d={d} parameters per row, but "
            f"ansatz.num_parameters={ansatz.num_parameters}"
        )

    if hamiltonian.n_qubits() != 0 and hamiltonian.n_qubits() != n_qubits:
        raise ValueError(
            f"hamiltonian.n_qubits()={hamiltonian.n_qubits()} does not match "
            f"ansatz.num_qubits={n_qubits}"
        )

    # Determine device
    if device is None:
        qdevice = default_device()
        device = qdevice.as_torch_device()

    # Prepare initial state
    if initial_state is None:
        from qconduit.backend.statevector import zero_state

        qdevice = default_device()
        initial_state = zero_state(
            n_qubits=n_qubits,
            batch_shape=None,
            device=qdevice,
            dtype=torch.complex128,
        )
    else:
        if initial_state.ndim != 1:
            raise ValueError(
                f"initial_state must be 1D with shape (dim,), got shape {initial_state.shape}"
            )
        dim = initial_state.shape[0]
        if dim != 2**n_qubits:
            raise ValueError(
                f"initial_state dimension {dim} does not match 2**n_qubits={2**n_qubits}"
            )
        # Normalize and ensure correct dtype/device
        norm = torch.linalg.norm(initial_state)
        if norm == 0.0:
            raise ValueError("initial_state has zero norm")
        initial_state = (initial_state / norm).to(device=device, dtype=torch.complex128)

    # Use apply_ansatz_batch_to_state to get batched states
    from qconduit.batched.apply import apply_ansatz_batch_to_state

    batched_states = apply_ansatz_batch_to_state(
        ansatz_factory=lambda p: ansatz.build_circuit(p),
        params_batch=params_batch.to(device=device),
        initial_state=initial_state,
    )

    # Evaluate expectations
    return evaluate_expectations_batched_via_states(hamiltonian, batched_states)


__all__ = [
    "evaluate_expectations_batched_via_states",
    "evaluate_expectations_for_params_batched",
]

