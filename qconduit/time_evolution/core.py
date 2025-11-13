"""Core time-evolution functions for statevector-based Trotterization."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch

from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate
from qconduit.core.device import Device, default_device
from qconduit.diagnostics import assert_normalized, is_debug_enabled
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum, PauliTerm

OrderLiteral = Literal[1, 2]


def _apply_pauli_term_evolution(
    state: torch.Tensor,
    term: PauliTerm,
    dt: float,
    n_qubits: int,
    device: Device,
) -> torch.Tensor:
    """
    Apply exp(-i * term.coeff * dt * P) to the given state, where P is
    the Pauli string encoded by `term`.

    This uses a standard decomposition:
    - Basis change so that all non-identity Paulis become Z.
    - Parity CNOT ladder into a pivot qubit.
    - Single RZ rotation by 2 * coeff * dt.
    - Undo CNOT ladder and basis changes.

    Parameters
    ----------
    state:
        Statevector tensor of shape (2**n_qubits,) on device.
    term:
        PauliTerm with real coefficient and pauli labels of length n_qubits.
    dt:
        Time step duration for this term.
    n_qubits:
        Number of qubits in the system.
    device:
        Device abstraction used to obtain torch.device and dtype.

    Returns
    -------
    torch.Tensor
        Updated statevector.
    """
    coeff = term.coeff
    # Require real coefficient for Hermitian time evolution.
    coeff_real = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
    if isinstance(coeff, complex) and abs(coeff.imag) > 1e-12:
        raise ValueError(
            "PauliTerm coefficient must be real for time evolution; "
            f"got {term.coeff}."
        )
    angle = 2.0 * coeff_real * dt

    paulis: Sequence[str] = term.paulis
    if len(paulis) != n_qubits:
        raise ValueError(
            f"PauliTerm length {len(paulis)} does not match n_qubits={n_qubits}."
        )

    torch_device = device.as_torch_device()
    dtype = state.dtype

    # Identify non-identity positions.
    non_identity_qubits: list[int] = []
    for q, p in enumerate(paulis):
        label = p.upper()
        if label not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli label {label!r} in term.")
        if label == "I":
            continue
        non_identity_qubits.append(q)

    # If all identity, evolution is trivial: global phase -> can skip.
    if not non_identity_qubits:
        return state

    # Track basis changes to undo later.
    basis_change: list[tuple[int, str]] = []
    for q in non_identity_qubits:
        label = paulis[q].upper()
        if label == "X":
            # H maps X to Z
            H = stdgates.H(dtype=dtype, device=torch_device)
            state = apply_gate(state, H, qubit=q, n_qubits=n_qubits)
            basis_change.append((q, "X"))
        elif label == "Y":
            # S† then H maps Y to Z
            # Implement S† as S^-1 = diag(1, -i).
            S = stdgates.S(dtype=dtype, device=torch_device)
            S_dag = S.conj().transpose(-2, -1)
            state = apply_gate(state, S_dag, qubit=q, n_qubits=n_qubits)
            H = stdgates.H(dtype=dtype, device=torch_device)
            state = apply_gate(state, H, qubit=q, n_qubits=n_qubits)
            basis_change.append((q, "Y"))
        elif label == "Z":
            basis_change.append((q, "Z"))
        else:
            # "I" already skipped
            basis_change.append((q, "I"))

    # Entangle into a pivot
    pivot = non_identity_qubits[0]
    # Build CNOT ladder: other qubits -> pivot
    # For parity accumulation, we want CNOT(q, pivot) where q is control and pivot is target
    ladder_qubits: list[int] = []
    for q in non_identity_qubits[1:]:
        # CNOT with q as control and pivot as target
        # The gate matrix depends on whether control < target
        control_first = q < pivot
        cnot_gate = stdgates.CNOT(dtype=dtype, device=torch_device, control_first=control_first)
        # When control_first=True, qubit1 is control and qubit2 is target
        # When control_first=False, qubit2 is control and qubit1 is target
        if control_first:
            state = apply_two_qubit_gate(
                state,
                cnot_gate,
                qubit1=q,
                qubit2=pivot,
                n_qubits=n_qubits,
            )
        else:
            state = apply_two_qubit_gate(
                state,
                cnot_gate,
                qubit1=pivot,
                qubit2=q,
                n_qubits=n_qubits,
            )
        ladder_qubits.append(q)

    # Apply RZ on pivot
    if abs(angle) > 0.0:
        rz = stdgates.RZ(angle, dtype=dtype, device=torch_device)
        state = apply_gate(state, rz, qubit=pivot, n_qubits=n_qubits)

    # Undo ladder in reverse order
    for q in reversed(ladder_qubits):
        control_first = q < pivot
        cnot_gate = stdgates.CNOT(dtype=dtype, device=torch_device, control_first=control_first)
        if control_first:
            state = apply_two_qubit_gate(
                state,
                cnot_gate,
                qubit1=q,
                qubit2=pivot,
                n_qubits=n_qubits,
            )
        else:
            state = apply_two_qubit_gate(
                state,
                cnot_gate,
                qubit1=pivot,
                qubit2=q,
                n_qubits=n_qubits,
            )

    # Undo basis changes (reverse order)
    for q, label in reversed(basis_change):
        if label == "X":
            # Undo H
            H = stdgates.H(dtype=dtype, device=torch_device)
            state = apply_gate(state, H, qubit=q, n_qubits=n_qubits)
        elif label == "Y":
            # Undo H then S† -> i.e., H then S
            H = stdgates.H(dtype=dtype, device=torch_device)
            state = apply_gate(state, H, qubit=q, n_qubits=n_qubits)
            S = stdgates.S(dtype=dtype, device=torch_device)
            state = apply_gate(state, S, qubit=q, n_qubits=n_qubits)
        # Z and I require no basis change undo

    if is_debug_enabled():
        assert_normalized(state, atol=1e-4)
    return state


def trotter_step_pauli_sum(
    state: torch.Tensor,
    hamiltonian: PauliSum,
    dt: float,
    n_qubits: int,
    order: OrderLiteral = 1,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Apply a single Trotter step for the evolution under a PauliSum
    Hamiltonian.

    For H = sum_k H_k (each H_k a PauliTerm), this approximates
    exp(-i H dt) |psi> using either:

    - First-order (Lie-Trotter):
        prod_k exp(-i H_k dt)
    - Second-order (symmetric Suzuki-Trotter):
        prod_k exp(-i H_k dt / 2) * prod_k' exp(-i H_k' dt / 2)
        where the second product is over the terms in reverse order.

    Parameters
    ----------
    state:
        Statevector of shape (2**n_qubits,) on the chosen device.
    hamiltonian:
        PauliSum Hamiltonian.
    dt:
        Time step duration.
    n_qubits:
        Number of qubits.
    order:
        1 for first-order, 2 for second-order symmetric Trotter.
    device:
        Optional Device. If None, default_device() is used.

    Returns
    -------
    torch.Tensor
        Updated statevector.
    """
    if device is None:
        dev = default_device()
    else:
        dev = device

    if order not in (1, 2):
        raise ValueError(f"Unsupported Trotter order {order}. Only 1 and 2 are supported.")

    terms: Sequence[PauliTerm] = hamiltonian.terms

    if order == 1:
        # First-order: prod_k exp(-i H_k dt)
        for term in terms:
            state = _apply_pauli_term_evolution(state, term, dt=dt, n_qubits=n_qubits, device=dev)
    else:
        # Second-order:
        # First sweep: dt/2
        half_dt = dt / 2.0
        for term in terms:
            state = _apply_pauli_term_evolution(
                state, term, dt=half_dt, n_qubits=n_qubits, device=dev
            )
        # Reverse sweep: dt/2
        for term in reversed(terms):
            state = _apply_pauli_term_evolution(
                state, term, dt=half_dt, n_qubits=n_qubits, device=dev
            )

    if is_debug_enabled():
        assert_normalized(state, atol=1e-4)
    return state


def time_evolve_state(
    state: torch.Tensor,
    hamiltonian: PauliSum,
    t: float,
    n_steps: int,
    n_qubits: int,
    order: OrderLiteral = 1,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Approximate time evolution under H = PauliSum for time t using
    Trotterization.

    This computes:
        |psi(t)> ≈ [U_dt]^n_steps |psi(0)>
    where U_dt is a single Trotter step for time dt = t / n_steps.

    Parameters
    ----------
    state:
        Initial statevector of shape (2**n_qubits,) on the chosen device.
    hamiltonian:
        PauliSum Hamiltonian.
    t:
        Total evolution time.
    n_steps:
        Number of Trotter steps. Must be >= 1.
    n_qubits:
        Number of qubits.
    order:
        1 or 2 for first- or second-order Trotter.
    device:
        Optional Device. If None, default_device() is used.

    Returns
    -------
    torch.Tensor
        Approximate time-evolved statevector at time t.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    dt = t / float(n_steps)

    if device is None:
        dev = default_device()
    else:
        dev = device

    # Ensure state is on the correct device
    if state.device != dev.as_torch_device():
        state = state.to(dev.as_torch_device())

    for _ in range(n_steps):
        state = trotter_step_pauli_sum(
            state=state,
            hamiltonian=hamiltonian,
            dt=dt,
            n_qubits=n_qubits,
            order=order,
            device=dev,
        )
    return state

