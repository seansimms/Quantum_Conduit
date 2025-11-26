"""Circuit-level integration for quantum channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from qconduit.backend.statevector import (
    apply_gate,
    apply_two_qubit_gate,
    zero_state,
)
from qconduit.channels.core import KrausChannel
from qconduit.channels.utils import density_from_statevector
from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device
from qconduit.gates import standard as stdgates


@dataclass(frozen=True)
class NoisyCircuit:
    """
    A wrapper around a QuantumCircuit with channel annotations.

    This non-invasively extends QuantumCircuit by storing a mapping of
    gate indices to channels that should be applied after those gates.

    Attributes
    ----------
    circuit: QuantumCircuit
        The underlying quantum circuit.
    channel_locations: Dict[int, Tuple[KrausChannel, Sequence[int]]]
        Mapping from gate index to (channel, target_qubits) tuple.
        Channels are applied after the gate at the specified index.
    """

    circuit: QuantumCircuit
    channel_locations: Dict[int, Tuple[KrausChannel, Sequence[int]]]

    def __post_init__(self) -> None:
        """Validate channel locations."""
        n_qubits = self.circuit.n_qubits
        for gate_idx, (channel, target_qubits) in self.channel_locations.items():
            if gate_idx < 0 or gate_idx >= len(self.circuit.ops):
                raise ValueError(
                    f"Gate index {gate_idx} out of range [0, {len(self.circuit.ops)})"
                )
            if any(q < 0 or q >= n_qubits for q in target_qubits):
                raise ValueError(
                    f"Target qubits {target_qubits} out of range [0, {n_qubits})"
                )


def annotate_circuit_with_channels(
    circuit: QuantumCircuit, channel_locations: Sequence[Tuple[int, KrausChannel]]
) -> NoisyCircuit:
    """
    Annotate a circuit with channels to be applied after specific gates.

    This is a non-invasive approach: it returns a NoisyCircuit wrapper that
    documents where channels should be applied, without modifying the original
    circuit.

    Parameters
    ----------
    circuit: QuantumCircuit
        The quantum circuit to annotate.
    channel_locations: Sequence[Tuple[int, KrausChannel]]
        List of (gate_index, channel) tuples. The channel will be applied
        after the gate at gate_index. For single-qubit channels, the channel
        acts on the qubit(s) that the gate at that index acts on.

    Returns
    -------
    NoisyCircuit
        A wrapper containing the circuit and channel annotations.

    Raises
    ------
    ValueError
        If gate indices are invalid or channels don't match gate qubits.
    """
    channel_dict: Dict[int, Tuple[KrausChannel, Sequence[int]]] = {}

    for gate_idx, channel in channel_locations:
        if gate_idx < 0 or gate_idx >= len(circuit.ops):
            raise ValueError(
                f"Gate index {gate_idx} out of range [0, {len(circuit.ops)})"
            )

        op = circuit.ops[gate_idx]
        # For single-qubit channels, use the gate's qubits
        # For multi-qubit channels, we'd need explicit target_qubits
        if channel.n_qubits == 1:
            if len(op.qubits) == 0:
                raise ValueError(f"Gate at index {gate_idx} has no qubits")
            # Apply channel to the first qubit (or all qubits for multi-qubit gates)
            # For simplicity, apply to first qubit
            target_qubits = [op.qubits[0]]
        else:
            # Multi-qubit channel: use all gate qubits
            target_qubits = list(op.qubits)
            if len(target_qubits) != channel.n_qubits:
                raise ValueError(
                    f"Channel n_qubits {channel.n_qubits} doesn't match "
                    f"gate qubits {len(target_qubits)}"
                )

        channel_dict[gate_idx] = (channel, target_qubits)

    return NoisyCircuit(circuit=circuit, channel_locations=channel_dict)


def apply_channel_schedule_to_state(
    state: torch.Tensor,
    channel_schedule: Sequence[Tuple[KrausChannel, Sequence[int]]],
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """
    Sequentially apply a schedule of channels to a density matrix.

    Each channel in the schedule is extended to the full system via tensor_extend
    and then applied to the density matrix.

    Parameters
    ----------
    state: torch.Tensor
        Input density matrix of shape (dim, dim) with dim = 2**n_qubits.
    channel_schedule: Sequence[Tuple[KrausChannel, Sequence[int]]]
        List of (channel, target_qubits) tuples to apply sequentially.
    n_qubits: Optional[int]
        Number of qubits. If None, inferred from state shape.

    Returns
    -------
    torch.Tensor
        Output density matrix after applying all channels.

    Raises
    ------
    ValueError
        If state is not a valid density matrix or channels are invalid.
    """
    if state.dim() != 2:
        raise ValueError(f"state must be 2D density matrix, got {state.dim()} dimensions")
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"state must be square, got shape {state.shape}")

    if n_qubits is None:
        dim = state.shape[0]
        n_qubits = int(dim.bit_length() - 1)  # log2(dim)
        if 1 << n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )

    rho = state
    for channel, target_qubits in channel_schedule:
        # Extend channel to full system
        extended_channel = channel.tensor_extend(n_qubits, target_qubits)
        # Apply to density matrix
        rho = extended_channel.apply_to_density(rho)

    return rho


def apply_circuit_with_noise(
    circuit: QuantumCircuit,
    channel_locations: Sequence[Tuple[int, KrausChannel]],
    psi0: Optional[torch.Tensor] = None,
    device: Optional[Device] = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Simulate a circuit with noise channels, returning a density matrix.

    This function simulates the circuit gate-by-gate. When noise channels are
    present, the simulation switches to density-matrix mode (since Kraus channels
    produce mixed states). If no channels are present, returns a pure statevector.

    Parameters
    ----------
    circuit: QuantumCircuit
        The quantum circuit to simulate.
    channel_locations: Sequence[Tuple[int, KrausChannel]]
        List of (gate_index, channel) tuples. Channels are applied after
        the gate at the specified index.
    psi0: Optional[torch.Tensor]
        Initial statevector. If None, starts from |0...0⟩.
    device: Optional[Device]
        Device for simulation. If None, uses default_device().
    dtype: torch.dtype
        Complex dtype for states. Defaults to torch.complex128.

    Returns
    -------
    torch.Tensor
        If no channels: final statevector of shape (dim,).
        If channels present: final density matrix of shape (dim, dim).

    Raises
    ------
    ValueError
        If circuit, channels, or initial state are invalid.
    """
    if device is None:
        dev = default_device()
    else:
        dev = device

    torch_device = dev.as_torch_device()
    n_qubits = circuit.n_qubits

    # Build channel mapping
    channel_map: Dict[int, Tuple[KrausChannel, Sequence[int]]] = {}
    for gate_idx, channel in channel_locations:
        if gate_idx < 0 or gate_idx >= len(circuit.ops):
            raise ValueError(
                f"Gate index {gate_idx} out of range [0, {len(circuit.ops)})"
            )
        op = circuit.ops[gate_idx]
        if channel.n_qubits == 1:
            target_qubits = [op.qubits[0]] if len(op.qubits) > 0 else []
        else:
            target_qubits = list(op.qubits)
        channel_map[gate_idx] = (channel, target_qubits)

    has_channels = len(channel_map) > 0

    # Initialize state
    if psi0 is None:
        if has_channels:
            # Start in density matrix form
            psi0_pure = zero_state(
                n_qubits=n_qubits,
                batch_shape=None,
                device=dev,
                dtype=dtype,
            )
            rho = density_from_statevector(psi0_pure)
        else:
            # Pure statevector simulation
            state = zero_state(
                n_qubits=n_qubits,
                batch_shape=None,
                device=dev,
                dtype=dtype,
            )
    else:
        if has_channels:
            rho = density_from_statevector(psi0.to(dtype=dtype, device=torch_device))
        else:
            state = psi0.to(dtype=dtype, device=torch_device)

    # Simulate circuit
    if has_channels:
        # Density matrix mode
        for i, op in enumerate(circuit.ops):
            # Apply gate as unitary: rho -> U rho U^†
            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate = _resolve_single_qubit_gate(name, op.params, dtype, torch_device)
                rho = _apply_single_qubit_unitary_to_dm(rho, gate, q, n_qubits)
            elif len(op.qubits) == 2:
                q0, q1 = op.qubits
                gate = _resolve_two_qubit_gate(name, q0, q1, dtype, torch_device)
                rho = _apply_two_qubit_unitary_to_dm(rho, gate, q0, q1, n_qubits)
            else:
                raise ValueError(
                    f"Unsupported gate {op.name} acting on {len(op.qubits)} qubits"
                )

            # Apply noise channel if present
            if i in channel_map:
                channel, target_qubits = channel_map[i]
                extended_channel = channel.tensor_extend(n_qubits, target_qubits)
                rho = extended_channel.apply_to_density(rho)

        return rho
    else:
        # Pure statevector mode
        for op in circuit.ops:
            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate = _resolve_single_qubit_gate(name, op.params, dtype, torch_device)
                state = apply_gate(state, gate, qubit=q, n_qubits=n_qubits)
            elif len(op.qubits) == 2:
                q0, q1 = op.qubits
                gate = _resolve_two_qubit_gate(name, q0, q1, dtype, torch_device)
                state = apply_two_qubit_gate(
                    state, gate, qubit1=q0, qubit2=q1, n_qubits=n_qubits
                )
            else:
                raise ValueError(
                    f"Unsupported gate {op.name} acting on {len(op.qubits)} qubits"
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


def _apply_single_qubit_unitary_to_dm(
    rho: torch.Tensor,
    unitary: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a single-qubit unitary to a density matrix."""
    dtype = rho.dtype
    device = rho.device

    identity_single = torch.eye(2, dtype=dtype, device=device)

    # Build embedded unitary
    # Convention: qubit 0 is LSB, build from n_qubits-1 down to 0
    full_unitary = unitary if (n_qubits - 1) == qubit else identity_single
    for q in range(n_qubits - 2, -1, -1):
        operand = unitary if q == qubit else identity_single
        full_unitary = torch.kron(full_unitary, operand)

    # Apply unitary: rho -> U_full rho U_full^\dagger
    tmp = torch.matmul(full_unitary, rho)
    result = torch.matmul(tmp, full_unitary.conj().transpose(-2, -1))
    return result


def _apply_two_qubit_unitary_to_dm(
    rho: torch.Tensor,
    unitary: torch.Tensor,
    qubit1: int,
    qubit2: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a two-qubit unitary to a density matrix."""
    dim = rho.shape[-1]
    dtype = rho.dtype
    device = rho.device

    # For small systems, build full matrix explicitly
    # This is simpler than trying to handle all qubit orderings

    # Build full 4x4 unitary matrix for the two qubits
    # We need to embed U into the full system
    # Strategy: build matrix element by element
    full_unitary = torch.zeros((dim, dim), dtype=dtype, device=device)

    # Iterate over all basis states
    for i in range(dim):
        for j in range(dim):
            # Extract bits for qubit1 and qubit2 from indices i and j
            # qubit 0 is LSB
            bits_i = [(i >> q) & 1 for q in range(n_qubits)]
            bits_j = [(j >> q) & 1 for q in range(n_qubits)]

            # Get the two-qubit state indices
            idx1_i = bits_i[qubit1]
            idx2_i = bits_i[qubit2]
            idx1_j = bits_j[qubit1]
            idx2_j = bits_j[qubit2]

            # Two-qubit index: (idx1, idx2) -> idx = idx1 * 2 + idx2
            twoq_i = idx1_i * 2 + idx2_i
            twoq_j = idx1_j * 2 + idx2_j

            # If other qubits match, contribute U[twoq_i, twoq_j]
            other_bits_match = all(
                bits_i[q] == bits_j[q] for q in range(n_qubits) if q not in (qubit1, qubit2)
            )

            if other_bits_match:
                # Build the full index contribution
                # We need to compute the full index from the bit pattern
                full_unitary[i, j] = unitary[twoq_i, twoq_j]

    # Apply unitary: rho -> U_full rho U_full^\dagger
    tmp = torch.matmul(full_unitary, rho)
    result = torch.matmul(tmp, full_unitary.conj().transpose(-2, -1))
    return result


__all__ = [
    "NoisyCircuit",
    "annotate_circuit_with_channels",
    "apply_channel_schedule_to_state",
    "apply_circuit_with_noise",
]

