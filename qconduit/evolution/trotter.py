"""Trotter-Suzuki time evolution for PauliSum Hamiltonians."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch

from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate
from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum, PauliTerm

TrotterOrder = Literal[1, 2]


@dataclass(frozen=True)
class TrotterSchedule:
    """
    Description of a Trotter-Suzuki time-evolution schedule for H = ∑_j H_j.

    Attributes
    ----------
    num_steps:
        Number of Trotter steps m.
    total_time:
        Total evolution time T.
    order:
        Trotter order: 1 (Lie-Trotter) or 2 (symmetric Suzuki).
    """

    num_steps: int
    total_time: float
    order: TrotterOrder

    def __post_init__(self) -> None:
        """Validate TrotterSchedule parameters."""
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        if not math.isfinite(self.total_time):
            raise ValueError(
                f"total_time must be finite, got {self.total_time} "
                f"(NaN or ±inf not allowed)"
            )

    @property
    def step_time(self) -> float:
        """
        Time increment per Trotter step: dt = total_time / num_steps.
        """
        return self.total_time / float(self.num_steps)


def _paulisum_terms(hamiltonian: PauliSum) -> Sequence[PauliTerm]:
    """
    Return a sequence of PauliTerm objects that make up the PauliSum.

    This function assumes that `PauliSum` provides a `.terms` attribute.
    """
    return hamiltonian.terms


def _add_time_evolution_for_term(
    circuit: QuantumCircuit,
    term: PauliTerm,
    tau: float,
) -> None:
    """
    Append gates to circuit implementing exp(-i * term.coeff * tau * P),
    where P is the Pauli string encoded by term.

    This uses a standard textbook decomposition:
    - Basis change so that all non-identity Paulis become Z.
    - Parity CNOT ladder into a pivot qubit.
    - Single RZ rotation by 2 * coeff * tau.
    - Undo CNOT ladder and basis changes.

    Parameters
    ----------
    circuit:
        QuantumCircuit to append gates to.
    term:
        PauliTerm with real coefficient and pauli labels.
    tau:
        Time step duration for this term.
    """
    coeff = term.coeff
    coeff_real = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
    if isinstance(coeff, complex) and abs(coeff.imag) > 1e-12:
        raise ValueError(
            "PauliTerm coefficient must be real for time evolution; "
            f"got {term.coeff}."
        )
    angle = 2.0 * coeff_real * tau

    paulis: Sequence[str] = term.paulis
    n_qubits = circuit.n_qubits
    if len(paulis) != n_qubits:
        raise ValueError(
            f"PauliTerm length {len(paulis)} does not match circuit.n_qubits={n_qubits}."
        )

    # Identify non-identity positions
    non_identity_qubits: list[int] = []
    for q, p in enumerate(paulis):
        label = p.upper()
        if label not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli label {label!r} in term.")
        if label != "I":
            non_identity_qubits.append(q)

    # If all identity, evolution is trivial (global phase)
    if not non_identity_qubits:
        return

    # Track basis changes to undo later
    basis_change: list[tuple[int, str]] = []
    for q in non_identity_qubits:
        label = paulis[q].upper()
        if label == "X":
            # H maps X to Z
            circuit.add_gate("H", [q])
            basis_change.append((q, "X"))
        elif label == "Y":
            # S† then H maps Y to Z
            # Implement S† via RZ(-pi/2) up to global phase, then H
            circuit.add_gate("RZ", [q], params=[-math.pi / 2.0])
            circuit.add_gate("H", [q])
            basis_change.append((q, "Y"))
        elif label == "Z":
            basis_change.append((q, "Z"))
        else:
            basis_change.append((q, "I"))

    # Parity CNOT ladder
    pivot = non_identity_qubits[0]
    ladder_qubits: list[int] = []
    for q in non_identity_qubits[1:]:
        # CNOT with q as control and pivot as target
        circuit.add_gate("CNOT", [q, pivot])
        ladder_qubits.append(q)

    # Single RZ rotation on pivot
    if abs(angle) > 0.0:
        circuit.add_gate("RZ", [pivot], params=[angle])

    # Undo ladder in reverse order
    for q in reversed(ladder_qubits):
        circuit.add_gate("CNOT", [q, pivot])

    # Undo basis changes (reverse order)
    for q, label in reversed(basis_change):
        if label == "X":
            circuit.add_gate("H", [q])
        elif label == "Y":
            # Undo H then S† ≈ RZ(-pi/2)
            circuit.add_gate("H", [q])
            circuit.add_gate("RZ", [q], params=[math.pi / 2.0])
        # Z and I require no basis change undo


def build_trotter_step_circuit(
    hamiltonian: PauliSum,
    step_time: float,
    order: TrotterOrder,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Build a single Trotter step circuit U_step approximating exp(-i H dt),
    where H = ∑_j H_j and dt = step_time.

    For order 1 (Lie-Trotter):
        U_step ≈ ∏_j exp(-i H_j dt)

    For order 2 (symmetric Suzuki):
        U_step ≈ ∏_j exp(-i H_j dt/2) ∏_j^R exp(-i H_j dt/2),

    where ∏_j^R denotes the product in reverse order.

    Parameters
    ----------
    hamiltonian:
        PauliSum representing H.
    step_time:
        Time increment dt.
    order:
        Trotter order (1 or 2).
    num_qubits:
        Number of qubits the Hamiltonian acts upon.

    Returns
    -------
    QuantumCircuit
        Circuit implementing the single Trotter step.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    circuit = QuantumCircuit(num_qubits)
    terms = list(_paulisum_terms(hamiltonian))

    if order == 1:
        for term in terms:
            _add_time_evolution_for_term(circuit, term, step_time)
    else:
        # Order 2: symmetric Suzuki
        half_dt = step_time / 2.0
        # Forward half-steps
        for term in terms:
            _add_time_evolution_for_term(circuit, term, half_dt)
        # Reverse half-steps
        for term in reversed(terms):
            _add_time_evolution_for_term(circuit, term, half_dt)

    return circuit


def build_trotter_circuit(
    hamiltonian: PauliSum,
    schedule: TrotterSchedule,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Build a Trotterized circuit approximating U(T) = exp(-i H T) as a product
    of `schedule.num_steps` Trotter steps.

    Parameters
    ----------
    hamiltonian:
        PauliSum representing H.
    schedule:
        TrotterSchedule specifying total time, number of steps, and order.
    num_qubits:
        Number of qubits.

    Returns
    -------
    QuantumCircuit
        Trotterized circuit with `schedule.num_steps` repetitions of the
        appropriate step circuit.
    """
    full = QuantumCircuit(num_qubits)
    step = build_trotter_step_circuit(
        hamiltonian, schedule.step_time, schedule.order, num_qubits
    )

    # Append all gates from step, repeated num_steps times
    step_ops = list(step.ops)
    for _ in range(schedule.num_steps):
        for op in step_ops:
            full.add_gate(
                op.name,
                list(op.qubits),
                params=list(op.params) if op.params is not None else None,
            )

    return full


def _apply_circuit_to_statevector(
    circuit: QuantumCircuit,
    state: torch.Tensor,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Apply a QuantumCircuit to an arbitrary initial statevector.

    This is similar to circuit.simulate_state() but works with an arbitrary
    initial state instead of always starting from |0...0⟩.

    Parameters
    ----------
    circuit:
        QuantumCircuit to apply.
    state:
        Initial statevector of shape (2**n_qubits,) with complex dtype.
    device:
        Optional Device. If None, inferred from state.

    Returns
    -------
    torch.Tensor
        Final statevector after applying the circuit.
    """
    if device is None:
        # Try to infer device from state
        if state.device.type == "cpu":
            dev = default_device()
        elif state.device.type == "cuda":
            from qconduit.core.device import device as device_factory

            dev = device_factory("sv_cuda")
        else:
            dev = default_device()
    else:
        dev = device

    torch_device = dev.as_torch_device()
    dtype = state.dtype

    # Ensure state is on the correct device
    if state.device != torch_device:
        state = state.to(torch_device)

    # Apply each gate in the circuit
    for op in circuit.ops:
        name = op.name.upper()
        if len(op.qubits) == 1:
            q = op.qubits[0]
            gate = _resolve_single_qubit_gate(name, op.params, dtype, torch_device)
            state = apply_gate(state, gate, qubit=q, n_qubits=circuit.n_qubits)
        elif len(op.qubits) == 2:
            q0, q1 = op.qubits
            gate = _resolve_two_qubit_gate(name, q0, q1, dtype, torch_device)
            state = apply_two_qubit_gate(
                state, gate, qubit1=q0, qubit2=q1, n_qubits=circuit.n_qubits
            )
        else:
            raise ValueError(
                f"QuantumCircuit currently supports only 1- and 2-qubit gates; "
                f"got gate {op.name!r} on qubits {op.qubits}."
            )

    return state


def _resolve_single_qubit_gate(
    name: str,
    params: Optional[tuple[float, ...]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a single-qubit gate name to a matrix."""
    n = name.upper()
    if n == "I":
        return stdgates.I(dtype=dtype, device=device)
    if n == "X":
        return stdgates.X(dtype=dtype, device=device)
    if n == "Y":
        return stdgates.Y(dtype=dtype, device=device)
    if n == "Z":
        return stdgates.Z(dtype=dtype, device=device)
    if n == "H":
        return stdgates.H(dtype=dtype, device=device)
    if n == "S":
        return stdgates.S(dtype=dtype, device=device)
    if n == "T":
        return stdgates.T(dtype=dtype, device=device)

    if n in ("RX", "RY", "RZ"):
        if not params or len(params) != 1:
            raise ValueError(f"Gate {n} requires exactly one parameter.")
        theta = float(params[0])
        if n == "RX":
            return stdgates.RX(theta, dtype=dtype, device=device)
        if n == "RY":
            return stdgates.RY(theta, dtype=dtype, device=device)
        if n == "RZ":
            return stdgates.RZ(theta, dtype=dtype, device=device)

    raise ValueError(
        f"Unsupported single-qubit gate name {name!r}. "
        "Supported gates: I, X, Y, Z, H, S, T, RX, RY, RZ."
    )


def _resolve_two_qubit_gate(
    name: str,
    control: int,
    target: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a two-qubit gate name to a matrix."""
    n = name.upper()
    if n == "CNOT":
        # The circuit IR convention: first qubit in the list is control, second is target.
        # To match HardwareEfficientAnsatz behavior, we use control_first=True
        # when control < target.
        control_first = control < target
        return stdgates.CNOT(dtype=dtype, device=device, control_first=control_first)
    raise ValueError(
        f"Unsupported two-qubit gate name {name!r}. "
        "Currently only 'CNOT' is supported."
    )


def evolve_state_trotter(
    state: torch.Tensor,
    hamiltonian: PauliSum,
    schedule: TrotterSchedule,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Approximate |ψ(T)⟩ = exp(-i H T) |ψ(0)⟩ via Trotter-Suzuki decomposition
    and statevector simulation.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ(0)⟩.
    hamiltonian:
        PauliSum representing H.
    schedule:
        TrotterSchedule specifying (T, m, order).
    device:
        Optional device for computation. If None, inferred from state.

    Returns
    -------
    torch.Tensor
        1D complex tensor of shape (2**n,) representing the approximate
        evolved state |ψ(T)⟩.

    Raises
    ------
    ValueError:
        If state is not 1D, has zero length, or has length that is not a power of 2.
    """
    # Validate state shape
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")
    if state.numel() == 0:
        raise ValueError("state must have nonzero length")

    dim = state.shape[0]

    # Check if dimension is a power of 2
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"Statevector length {dim} must be a power of 2.")

    # Determine device
    if device is None:
        if state.device.type == "meta":
            device_obj = default_device()
        elif state.device.type == "cpu":
            device_obj = default_device()
        elif state.device.type == "cuda":
            from qconduit.core.device import device as device_factory

            device_obj = device_factory("sv_cuda")
        else:
            device_obj = default_device()
    else:
        if device.type == "cpu":
            device_obj = default_device()
        elif device.type == "cuda":
            from qconduit.core.device import device as device_factory

            device_obj = device_factory("sv_cuda")
        else:
            device_obj = default_device()

    # Move state to device
    if state.device != device_obj.as_torch_device():
        state = state.to(device_obj.as_torch_device())

    # Build Trotter circuit
    circ = build_trotter_circuit(hamiltonian, schedule, num_qubits=n_qubits)

    # Apply circuit to state
    state_out = _apply_circuit_to_statevector(circ, state, device=device_obj)

    return state_out


__all__ = [
    "TrotterOrder",
    "TrotterSchedule",
    "build_trotter_step_circuit",
    "build_trotter_circuit",
    "evolve_state_trotter",
]

