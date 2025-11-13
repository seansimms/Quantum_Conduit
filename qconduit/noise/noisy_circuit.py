"""Noisy circuit simulation using density-matrix backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..backend.density_matrix import (
    _kron,
    apply_kraus_single_qubit,
    dm_from_statevector,
    measure_probs_dm,
    zero_dm_state,
)
from ..backend.statevector import zero_state
from ..circuit import QuantumCircuit
from ..core.device import Device, default_device
from ..gates import standard as stdgates
from .channels import SingleQubitChannel


@dataclass(frozen=True)
class NoiseConfig:
    """
    Configuration for applying textbook noise channels to a quantum circuit.

    This is a purely classical description: it specifies which single-qubit
    channels to apply to which qubits, and when they are applied.

    Currently, the semantics are:

    - For each gate in the quantum circuit:
      - Apply the gate as usual.
      - For each qubit acted on by the gate:
        - If a channel is specified for that qubit, apply it immediately
          after the gate (in the chosen representation, e.g., density matrix).

    The mapping is defined by `per_qubit_channels`.
    """

    per_qubit_channels: Dict[int, SingleQubitChannel]

    def __post_init__(self) -> None:
        """Validate that all qubit indices are non-negative."""
        for qubit_idx in self.per_qubit_channels.keys():
            if qubit_idx < 0:
                raise ValueError(
                    f"Qubit index must be non-negative, got {qubit_idx}"
                )


def _apply_unitary_to_dm(
    rho: torch.Tensor,
    U: torch.Tensor,
    qubits: tuple[int, ...],
    n_qubits: int,
) -> torch.Tensor:
    """
    Apply a unitary gate U to a density matrix rho.

    For a unitary U, the evolution is: rho -> U rho U^\\dagger.

    Parameters
    ----------
    rho:
        Density matrix of shape (dim, dim) with dim = 2**n_qubits.
    U:
        Unitary gate matrix. For single-qubit gates, shape (2, 2).
        For two-qubit gates, shape (4, 4).
    qubits:
        Tuple of qubit indices that the gate acts on.
    n_qubits:
        Total number of qubits.

    Returns
    -------
    torch.Tensor
        New density matrix after applying the unitary.
    """
    if len(qubits) == 1:
        # Single-qubit gate - use the helper
        qubit = qubits[0]
        return _apply_single_qubit_unitary_to_dm(rho, U, qubit, n_qubits)
    elif len(qubits) == 2:
        # Two-qubit gate - use the helper
        q0, q1 = qubits
        return _apply_two_qubit_unitary_to_dm(rho, U, q0, q1, n_qubits)
    else:
        raise ValueError(
            f"Gates acting on {len(qubits)} qubits are not supported. "
            "Only 1- and 2-qubit gates are supported."
        )


def _apply_single_qubit_unitary_to_dm(
    rho: torch.Tensor,
    U: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """
    Apply a single-qubit unitary gate to a density matrix.

    Builds U_full = I ⊗ ... ⊗ U ⊗ ... ⊗ I and applies: rho -> U_full rho U_full^\\dagger.
    """
    dim = rho.shape[-1]
    dtype = rho.dtype
    device = rho.device

    # Build full-system unitary by tensoring with identity
    I_single = torch.eye(2, dtype=dtype, device=device)

    # Build U_full = I ⊗ ... ⊗ U ⊗ ... ⊗ I
    # Convention: qubit 0 is LSB, build from n_qubits-1 down to 0
    U_full = U if (n_qubits - 1) == qubit else I_single
    for q in range(n_qubits - 2, -1, -1):
        if q == qubit:
            op = U
        else:
            op = I_single
        U_full = _kron(U_full, op)

    # Apply unitary: rho -> U_full rho U_full^\dagger
    Urho = torch.matmul(U_full, rho)
    UrhoUdag = torch.matmul(Urho, U_full.conj().transpose(-2, -1))
    return UrhoUdag


def _apply_two_qubit_unitary_to_dm(
    rho: torch.Tensor,
    U: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int,
) -> torch.Tensor:
    """
    Apply a 2-qubit unitary gate to a density matrix.

    This builds the full-system unitary and applies it: rho -> U_full rho U_full^\\dagger.

    The gate U is assumed to act on qubits in the order (qubit1, qubit2) as they appear
    in the circuit, which may not be sorted. The implementation handles the embedding
    correctly by building the full matrix explicitly for small systems.
    """
    dim = rho.shape[-1]
    dtype = rho.dtype
    device = rho.device

    # Order qubits for embedding
    q_low, q_high = min(qubit1, qubit2), max(qubit1, qubit2)
    swapped = qubit1 > qubit2

    # Build the full unitary matrix explicitly
    # For a 2-qubit gate U on qubits q_low and q_high, we need to embed it into
    # the full 2**n_qubits dimensional space
    U_full = torch.zeros((dim, dim), dtype=dtype, device=device)

    # Iterate through all basis states
    for i in range(dim):
        # Decompose index i into bits
        bits_i = [(i >> q) & 1 for q in range(n_qubits)]

        # Extract the two relevant qubit values
        bit_low_i = bits_i[q_low]
        bit_high_i = bits_i[q_high]

        # Index into U: U is defined for |qubit1, qubit2> basis
        # If swapped, qubit1 > qubit2, so U acts on |q_high, q_low>
        # If not swapped, qubit1 < qubit2, so U acts on |q_low, q_high>
        if swapped:
            u_idx_in = bit_high_i * 2 + bit_low_i
        else:
            u_idx_in = bit_low_i * 2 + bit_high_i

        for j in range(dim):
            bits_j = [(j >> q) & 1 for q in range(n_qubits)]

            # Check if states i and j differ only in qubits q_low and q_high
            if not all(
                bits_i[q] == bits_j[q]
                for q in range(n_qubits)
                if q not in (q_low, q_high)
            ):
                continue

            # Extract the two relevant qubit values for j
            bit_low_j = bits_j[q_low]
            bit_high_j = bits_j[q_high]

            # Index into U for output
            if swapped:
                u_idx_out = bit_high_j * 2 + bit_low_j
            else:
                u_idx_out = bit_low_j * 2 + bit_high_j

            # Set the matrix element
            U_full[i, j] = U[u_idx_in, u_idx_out]

    # Apply: rho -> U_full rho U_full^\dagger
    Urho = torch.matmul(U_full, rho)
    UrhoUdag = torch.matmul(Urho, U_full.conj().transpose(-2, -1))
    return UrhoUdag


def _resolve_single_qubit_gate(
    name: str,
    params: Optional[tuple[float, ...]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a gate name to a 2x2 matrix."""
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
    """Resolve a 2-qubit gate name to a 4x4 matrix."""
    n = name.upper()
    if n == "CNOT":
        control_first = control < target
        return stdgates.CNOT(dtype=dtype, device=device, control_first=control_first)
    raise ValueError(
        f"Unsupported two-qubit gate name {name!r}. "
        "Currently only 'CNOT' is supported."
    )


def simulate_noisy_circuit_dm(
    circuit: QuantumCircuit,
    noise: NoiseConfig,
    initial_state: Optional[torch.Tensor] = None,
    use_statevector_backend: bool = True,
    device: Optional[Device | torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Simulate a QuantumCircuit under textbook single-qubit noise models,
    returning the final density matrix.

    Parameters
    ----------
    circuit:
        QuantumCircuit describing the ideal (noise-free) unitary evolution.
    noise:
        NoiseConfig specifying which single-qubit channels to apply on which
        qubits after each gate that acts on those qubits.
    initial_state:
        Optional initial state. If None, uses |0...0><0...0|. If provided:
        - If `use_statevector_backend` is True, `initial_state` is interpreted
          as a statevector of shape (2**n,).
        - Otherwise, it is interpreted as a density matrix of shape (2**n, 2**n).
    use_statevector_backend:
        If True, initialize from a statevector (either provided or |0...0>),
        then convert to a density matrix. If False, initialize directly as
        a density matrix.
    device:
        Optional torch.device. If None, use `default_device()`.
    dtype:
        Complex dtype to use for the simulation (default: complex64).

    Returns
    -------
    torch.Tensor
        Final density matrix of shape (2**n_qubits, 2**n_qubits) as a
        torch complex tensor on the chosen device.
    """
    n_qubits = circuit.n_qubits

    # Resolve device
    if device is None:
        qdevice = default_device()
    elif isinstance(device, Device):
        qdevice = device
    else:
        # torch.device - convert to Device
        if device.type == "cpu":
            from ..core.device import device as device_factory
            qdevice = device_factory("sv_cpu")
        else:
            raise ValueError(f"Unsupported device type: {device.type}")

    torch_device = qdevice.as_torch_device()

    # Initialize density matrix
    if use_statevector_backend:
        if initial_state is None:
            state = zero_state(n_qubits, device=qdevice, dtype=dtype)
            rho = dm_from_statevector(state)
        else:
            if initial_state.ndim != 1:
                raise ValueError(
                    f"When use_statevector_backend=True, initial_state must be 1D, "
                    f"got shape {initial_state.shape}"
                )
            if initial_state.shape[0] != 2**n_qubits:
                raise ValueError(
                    f"initial_state length {initial_state.shape[0]} does not match "
                    f"2**n_qubits = {2**n_qubits}"
                )
            state = initial_state.to(device=torch_device, dtype=dtype)
            rho = dm_from_statevector(state)
    else:
        if initial_state is None:
            rho = zero_dm_state(n_qubits, device=qdevice, dtype=dtype)
        else:
            if initial_state.ndim != 2:
                raise ValueError(
                    f"When use_statevector_backend=False, initial_state must be 2D, "
                    f"got shape {initial_state.shape}"
                )
            if initial_state.shape != (2**n_qubits, 2**n_qubits):
                raise ValueError(
                    f"initial_state shape {initial_state.shape} does not match "
                    f"(2**n_qubits, 2**n_qubits) = ({2**n_qubits}, {2**n_qubits})"
                )
            rho = initial_state.to(device=torch_device, dtype=dtype)

    # Iterate over gates and apply them with noise
    for op in circuit.ops:
        # Apply the gate
        if len(op.qubits) == 1:
            gate = _resolve_single_qubit_gate(
                op.name, op.params, dtype, torch_device
            )
            rho = _apply_unitary_to_dm(rho, gate, op.qubits, n_qubits)
        elif len(op.qubits) == 2:
            q0, q1 = op.qubits
            gate = _resolve_two_qubit_gate(op.name, q0, q1, dtype, torch_device)
            rho = _apply_two_qubit_unitary_to_dm(rho, gate, q0, q1, n_qubits)
        else:
            raise ValueError(
                f"Gates acting on {len(op.qubits)} qubits are not supported"
            )

        # Apply noise after the gate
        for qubit in op.qubits:
            if qubit in noise.per_qubit_channels:
                channel = noise.per_qubit_channels[qubit]
                rho = apply_kraus_single_qubit(
                    rho, channel.kraus_operators, qubit=qubit, n_qubits=n_qubits
                )

    return rho


def sample_noisy_circuit_dm(
    circuit: QuantumCircuit,
    noise: NoiseConfig,
    n_shots: int,
    initial_state: Optional[torch.Tensor] = None,
    device: Optional[Device | torch.device] = None,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample bitstrings from the output of a noisy circuit simulation
    using the density-matrix backend.

    Parameters
    ----------
    circuit:
        QuantumCircuit to simulate.
    noise:
        NoiseConfig specifying per-qubit channels.
    n_shots:
        Number of measurement shots to draw.
    initial_state:
        Optional initial state (statevector or density matrix, consistent with
        `simulate_noisy_circuit_dm` default behavior).
    device:
        Optional torch.device (default: `default_device()`).
    dtype:
        Complex dtype for simulation.
    generator:
        Optional torch.Generator to make sampling reproducible.

    Returns
    -------
    torch.Tensor
        Integer bitstrings of shape (n_shots, n_qubits), where each row
        is a measurement outcome in computational basis order.
    """
    # Get final density matrix
    rho = simulate_noisy_circuit_dm(
        circuit=circuit,
        noise=noise,
        initial_state=initial_state,
        device=device,
        dtype=dtype,
    )

    # Get measurement probabilities
    probs = measure_probs_dm(rho)

    # Sample using existing utility
    from ..sampling.bitstrings import sample_from_probs

    samples = sample_from_probs(
        probs=probs,
        n_qubits=circuit.n_qubits,
        n_shots=n_shots,
        generator=generator,
    )

    # sample_from_probs returns shape (n_shots, n_qubits) for non-batched probs
    return samples

