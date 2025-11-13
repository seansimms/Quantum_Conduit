"""Variational Quantum Eigensolver (VQE) driver and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from qconduit.backend.statevector import zero_state, apply_gate, apply_two_qubit_gate
from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device
from qconduit.exact import paulisum_to_dense
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum
from qconduit.operators.expectation import expectation_pauli_sum

if TYPE_CHECKING:
    from qconduit.variational.ansatz import VariationalAnsatz


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
    params: Optional[Tuple[float, ...]],
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
        control_first = control < target
        return stdgates.CNOT(dtype=dtype, device=device, control_first=control_first)
    raise ValueError(
        f"Unsupported two-qubit gate name {name!r}. "
        "Currently only 'CNOT' is supported."
    )


@dataclass(frozen=True)
class VQEResult:
    """
    Result of a VQE optimization run.

    Attributes
    ----------
    optimal_params:
        1D tensor of shape (num_parameters,) with the best-found parameters.
    optimal_value:
        Best-found expectation value (energy).
    history:
        List of (iteration_index, energy_value) tuples recording the energy
        at each evaluation in the optimization.
    n_evaluations:
        Total number of objective evaluations performed.
    converged:
        True if a stopping criterion other than max_iterations was triggered.
    """

    optimal_params: torch.Tensor
    optimal_value: float
    history: List[Tuple[int, float]]
    n_evaluations: int
    converged: bool


def evaluate_expectation_value(
    ansatz: "VariationalAnsatz",
    params: torch.Tensor,
    hamiltonian: PauliSum,
    initial_state: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate the expectation value ⟨ψ(θ)| H |ψ(θ)⟩ where |ψ(θ)⟩ is obtained by
    applying the ansatz circuit to either the default |0...0⟩ state or an
    optional initial state.

    Parameters
    ----------
    ansatz:
        Variational ansatz implementing build_circuit and num_qubits/parameters.
    params:
        1D tensor of shape (ansatz.num_parameters,) containing real parameters.
    hamiltonian:
        PauliSum representing the Hamiltonian H.
    initial_state:
        Optional statevector of shape (2**n,) for n == ansatz.num_qubits. If
        None, the all-zero state |0...0⟩ is used.
    device:
        Optional device on which to perform the computation.

    Returns
    -------
    float
        Real-valued expectation ⟨H⟩.
    """
    n_qubits = ansatz.num_qubits

    # Determine device
    if device is None:
        if initial_state is not None:
            torch_device = initial_state.device
        else:
            qdevice = default_device()
            torch_device = qdevice.as_torch_device()
    else:
        torch_device = device

    # Ensure params is float type on device
    params = params.to(device=torch_device, dtype=torch.float64)

    # Build circuit
    circuit = ansatz.build_circuit(params)

    # Prepare initial state
    if initial_state is not None:
        if initial_state.ndim != 1:
            raise ValueError(f"initial_state must be 1D, got shape {initial_state.shape}")
        if initial_state.shape[0] != 2**n_qubits:
            raise ValueError(
                f"initial_state length {initial_state.shape[0]} does not match "
                f"2**n_qubits = {2**n_qubits}"
            )
        # Normalize
        norm = torch.linalg.norm(initial_state)
        if norm == 0.0:
            raise ValueError("initial_state has zero norm")
        state = (initial_state / norm).to(device=torch_device, dtype=torch.complex128)
    else:
        # Create |0...0⟩
        qdevice = default_device() if device is None else Device(
            name="custom",
            torch_device=torch_device,
            dtype=torch.float64,
            complex_dtype=torch.complex128,
        )
        state = zero_state(
            n_qubits=n_qubits,
            batch_shape=None,
            device=qdevice,
            dtype=torch.complex128,
        )

    # Apply circuit
    qdevice = default_device() if device is None else Device(
        name="custom",
        torch_device=torch_device,
        dtype=torch.float64,
        complex_dtype=torch.complex128,
    )
    final_state = _apply_circuit_to_statevector(circuit, state, device=qdevice)

    # Compute expectation using existing helper
    expectation = expectation_pauli_sum(final_state, hamiltonian)

    # Convert to scalar float
    if expectation.ndim == 0:
        return float(expectation.item())
    else:
        return float(expectation.item())


def _build_statevector_autograd(
    ansatz: "VariationalAnsatz",
    params: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build statevector directly using PyTorch operations to preserve gradients.

    This is a fallback that works for specific ansatz types by building
    the statevector directly rather than going through circuit building.
    """
    from qconduit.variational.ansatz import (
        HardwareEfficientAnsatz,
        LayeredEntanglerAnsatz,
        QAOAAnsatz,
    )
    from qconduit.gates.standard import RX, RY, RZ, H, CNOT
    from qconduit.time_evolution.core import trotter_step_pauli_sum

    n_qubits = ansatz.num_qubits
    qdevice = Device(
        name="custom",
        torch_device=device,
        dtype=torch.float64,
        complex_dtype=torch.complex128,
    )

    # Prepare |0...0⟩ state
    state = zero_state(
        n_qubits=n_qubits,
        batch_shape=None,
        device=qdevice,
        dtype=torch.complex128,
    )

    if isinstance(ansatz, HardwareEfficientAnsatz):
        # Build HardwareEfficientAnsatz statevector directly
        for layer in range(ansatz.num_layers):
            for qubit in range(n_qubits):
                base = 2 * (layer * n_qubits + qubit)
                theta_rx = params[base]
                theta_rz = params[base + 1]

                gate_rx = RX(theta_rx, dtype=torch.complex128, device=device)
                state = apply_gate(state, gate_rx, qubit=qubit, n_qubits=n_qubits)

                gate_rz = RZ(theta_rz, dtype=torch.complex128, device=device)
                state = apply_gate(state, gate_rz, qubit=qubit, n_qubits=n_qubits)

            for qubit in range(n_qubits - 1):
                # CNOT with qubit as control, qubit+1 as target
                control_first = qubit < (qubit + 1)
                cnot_gate = CNOT(dtype=torch.complex128, device=device, control_first=control_first)
                state = apply_two_qubit_gate(
                    state, cnot_gate, qubit1=qubit, qubit2=qubit + 1, n_qubits=n_qubits
                )

    elif isinstance(ansatz, LayeredEntanglerAnsatz):
        # Build LayeredEntanglerAnsatz statevector directly
        for layer in range(ansatz.num_layers):
            for qubit in range(n_qubits):
                base = 2 * (layer * n_qubits + qubit)
                theta_ry = params[base]
                theta_rz = params[base + 1]

                gate_ry = RY(theta_ry, dtype=torch.complex128, device=device)
                state = apply_gate(state, gate_ry, qubit=qubit, n_qubits=n_qubits)

                gate_rz = RZ(theta_rz, dtype=torch.complex128, device=device)
                state = apply_gate(state, gate_rz, qubit=qubit, n_qubits=n_qubits)

            for qubit in range(n_qubits - 1):
                # CNOT with qubit as control, qubit+1 as target
                control_first = qubit < (qubit + 1)
                cnot_gate = CNOT(dtype=torch.complex128, device=device, control_first=control_first)
                state = apply_two_qubit_gate(
                    state, cnot_gate, qubit1=qubit, qubit2=qubit + 1, n_qubits=n_qubits
                )

            if ansatz.ring_entanglement and n_qubits > 2:
                # CNOT with last qubit as control, first qubit as target
                # Since n_qubits-1 > 0, control_first should be False
                control_first = (n_qubits - 1) < 0  # False since n_qubits-1 > 0
                cnot_gate = CNOT(dtype=torch.complex128, device=device, control_first=control_first)
                state = apply_two_qubit_gate(
                    state, cnot_gate, qubit1=n_qubits - 1, qubit2=0, n_qubits=n_qubits
                )

    elif isinstance(ansatz, QAOAAnsatz):
        # Build QAOAAnsatz statevector directly
        # Initialize in |+⟩^⊗n
        gate_h = H(dtype=torch.complex128, device=device)
        for qubit in range(n_qubits):
            state = apply_gate(state, gate_h, qubit=qubit, n_qubits=n_qubits)

        # Split parameters
        gammas = params[: ansatz.depth]
        betas = params[ansatz.depth :]

        # Apply QAOA layers
        for p in range(ansatz.depth):
            gamma_p = gammas[p]
            beta_p = betas[p]

            # Apply cost unitary exp(-i γ_p H_C) using Trotter step
            state = trotter_step_pauli_sum(
                state=state,
                hamiltonian=ansatz.cost_hamiltonian,
                dt=gamma_p,
                n_qubits=n_qubits,
                order=1,
                device=qdevice,
            )

            # Apply mixer unitary exp(-i β_p ∑_i X_i) = Rx(2*β_p) on each qubit
            for qubit in range(n_qubits):
                gate_rx = RX(2.0 * beta_p, dtype=torch.complex128, device=device)
                state = apply_gate(state, gate_rx, qubit=qubit, n_qubits=n_qubits)

    else:
        # Fallback: use circuit building (loses gradients)
        circuit = ansatz.build_circuit(params)
        state = _apply_circuit_to_statevector(circuit, state, device=qdevice)

    return state


def _vqe_energy_autograd(
    params: torch.Tensor,
    ansatz: "VariationalAnsatz",
    hamiltonian: PauliSum,
    device: torch.device,
) -> torch.Tensor:
    """
    Internal autograd-compatible energy function for VQE.

    Returns a scalar torch.Tensor with requires_grad=True.
    """
    n_qubits = ansatz.num_qubits

    # Build statevector directly to preserve gradients
    final_state = _build_statevector_autograd(ansatz, params, device)

    # Compute expectation via dense matrix (preserves gradients)
    H_dense = paulisum_to_dense(
        hamiltonian,
        num_qubits=n_qubits,
        device=device,
        dtype=torch.complex128,
    )

    # ⟨ψ|H|ψ⟩ = ψ† H ψ
    psi = final_state
    value = (psi.conj().unsqueeze(0) @ (H_dense @ psi.unsqueeze(1))).squeeze()

    # Return real part as scalar
    return value.real


def run_vqe(
    hamiltonian: PauliSum,
    ansatz: "VariationalAnsatz",
    initial_params: torch.Tensor,
    optimizer_name: str = "adam",
    max_iterations: int = 200,
    learning_rate: float = 0.05,
    tol_rel: float = 1e-6,
    device: Optional[torch.device] = None,
) -> VQEResult:
    """
    Run a simple VQE optimization loop to minimize ⟨H⟩ over ansatz parameters.

    This uses a first-order optimizer from torch.optim (SGD or Adam) on a
    differentiable cost function implemented via statevector simulation.

    Parameters
    ----------
    hamiltonian:
        PauliSum representing the Hamiltonian H to minimize.
    ansatz:
        VariationalAnsatz instance providing build_circuit and num_qubits.
    initial_params:
        1D tensor of shape (ansatz.num_parameters,) with initial parameter values.
    optimizer_name:
        Name of the optimizer to use: "sgd" or "adam" (case-insensitive).
    max_iterations:
        Maximum number of optimization iterations.
    learning_rate:
        Initial learning rate for the optimizer.
    tol_rel:
        Relative tolerance for convergence. Optimization stops if the relative
        change in energy between iterations falls below this threshold.
    device:
        Optional torch.device for computation. If None, `default_device()` is used.

    Returns
    -------
    VQEResult
        Summary of the optimization run.
    """
    # Validate inputs
    if initial_params.ndim != 1:
        raise ValueError(f"initial_params must be 1D, got shape {initial_params.shape}")
    if initial_params.shape[0] != ansatz.num_parameters:
        raise ValueError(
            f"initial_params length {initial_params.shape[0]} does not match "
            f"ansatz.num_parameters {ansatz.num_parameters}"
        )
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if tol_rel <= 0:
        raise ValueError(f"tol_rel must be > 0, got {tol_rel}")

    # Determine device
    if device is None:
        qdevice = default_device()
        device = qdevice.as_torch_device()
    else:
        device = device

    # Create parameter tensor with requires_grad=True
    params = (
        initial_params.to(device=device, dtype=torch.float64)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # Choose optimizer
    opt_name_lower = optimizer_name.lower()
    if opt_name_lower == "sgd":
        optimizer = torch.optim.SGD([params], lr=learning_rate)
    elif opt_name_lower == "adam":
        optimizer = torch.optim.Adam([params], lr=learning_rate)
    else:
        raise ValueError(
            f"Unsupported optimizer_name '{optimizer_name}'. "
            "Supported optimizers: 'sgd', 'adam'"
        )

    # Initialize tracking variables
    history: List[Tuple[int, float]] = []
    best_value = float("inf")
    best_params = params.detach().clone()
    converged = False
    n_evaluations = 0
    prev_value: Optional[float] = None

    # Optimization loop
    for it in range(max_iterations):
        optimizer.zero_grad()

        # Compute expectation as differentiable scalar
        energy = _vqe_energy_autograd(
            params=params,
            ansatz=ansatz,
            hamiltonian=hamiltonian,
            device=device,
        )

        n_evaluations += 1
        value = float(energy.item())
        history.append((it, value))

        if value < best_value:
            best_value = value
            best_params = params.detach().clone()

        # Check convergence if prev_value exists
        if prev_value is not None:
            denom = max(1.0, abs(prev_value))
            rel_change = abs(prev_value - value) / denom
            if rel_change < tol_rel:
                converged = True
                break

        # Backprop and step
        energy.backward()
        optimizer.step()

        prev_value = value

    # Construct result
    result = VQEResult(
        optimal_params=best_params.detach().cpu(),
        optimal_value=best_value,
        history=history,
        n_evaluations=n_evaluations,
        converged=converged,
    )

    return result


__all__ = [
    "VQEResult",
    "evaluate_expectation_value",
    "run_vqe",
]

