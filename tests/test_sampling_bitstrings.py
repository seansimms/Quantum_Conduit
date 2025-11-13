"""Tests for bitstring sampling utilities."""

from __future__ import annotations

import torch

from qconduit.backend.statevector import zero_state
from qconduit.gates import standard as stdgates
from qconduit.backend.statevector import apply_gate
from qconduit.core.device import default_device
from qconduit.sampling import (
    sample_from_probs,
    sample_bitstrings_state,
    sample_bitstrings_dm,
    sample_bitstrings_circuit,
)
from qconduit.backend.density_matrix import dm_from_statevector
from qconduit.circuit import QuantumCircuit


def test_sample_from_probs_deterministic() -> None:
    """Test sampling from a deterministic distribution."""
    # Single qubit |0>
    n_qubits = 1
    probs = torch.tensor([1.0, 0.0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    samples = sample_from_probs(
        probs=probs,
        n_qubits=n_qubits,
        n_shots=50,
        qubits=None,
        generator=generator,
    )
    # Shape (50, 1)
    assert samples.shape == (50, 1)
    assert torch.all(samples == 0)


def test_sample_bitstrings_state_plus_state_balanced() -> None:
    """Test sampling from |+> state yields roughly balanced 0/1 outcomes."""
    dev = default_device()
    # Build |+> = H |0>
    state = zero_state(n_qubits=1, batch_shape=None, device=dev, dtype=torch.complex64)
    H = stdgates.H(dtype=torch.complex64, device=dev.as_torch_device())
    state = apply_gate(state, H, qubit=0, n_qubits=1)

    generator = torch.Generator().manual_seed(42)
    shots = 2000
    samples = sample_bitstrings_state(
        state=state,
        n_qubits=1,
        n_shots=shots,
        qubits=None,
        generator=generator,
    )
    # Expect roughly half 0 and half 1
    zeros = (samples == 0).sum().item()
    ones = (samples == 1).sum().item()
    assert abs(zeros - ones) < shots * 0.2  # 20% tolerance


def test_sample_bitstrings_dm_one_state() -> None:
    """Test sampling from density matrix |1><1| yields all ones."""
    dev = default_device()
    # Pure |1> state
    state = torch.tensor(
        [0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64, device=dev.as_torch_device()
    )
    rho = dm_from_statevector(state)

    generator = torch.Generator().manual_seed(123)
    samples = sample_bitstrings_dm(
        rho=rho,
        n_qubits=1,
        n_shots=100,
        qubits=None,
        generator=generator,
    )
    assert samples.shape == (100, 1)
    # All ones
    assert torch.all(samples == 1)


def test_sample_bitstrings_circuit_bell_state() -> None:
    """Test sampling from a circuit produces expected distribution."""
    # Build a 2-qubit circuit: H on q0, then CNOT(0->1)
    circuit = QuantumCircuit(n_qubits=2)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])

    generator = torch.Generator().manual_seed(7)
    shots = 2000
    samples = sample_bitstrings_circuit(
        circuit=circuit,
        n_shots=shots,
        qubits=None,
        generator=generator,
    )
    # Shape (shots, 2)
    assert samples.shape == (shots, 2)

    # Compute empirical frequencies
    # The circuit produces p(00)=0.5, p(01)=0.5
    # Bitstring representation uses [q0, q1] ordering, so:
    # - Index 0 = |00> (in |q1 q0> notation) -> [0, 0] (in [q0, q1] notation)
    # - Index 1 = |01> (in |q1 q0> notation) -> [1, 0] (in [q0, q1] notation)
    zeros_zero = (
        (samples[:, 0] == 0).logical_and(samples[:, 1] == 0).sum().item()
    )
    one_zero = (
        (samples[:, 0] == 1).logical_and(samples[:, 1] == 0).sum().item()
    )

    # Should be roughly 50/50 split between [0,0] and [1,0]
    assert abs(zeros_zero - one_zero) < shots * 0.2
    # Most outcomes should be [0,0] or [1,0]
    assert (zeros_zero + one_zero) > shots * 0.9


def test_sample_from_probs_qubit_subset() -> None:
    """Test sampling with a subset of qubits."""
    # 2-qubit uniform distribution
    n_qubits = 2
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    generator = torch.Generator().manual_seed(999)
    samples = sample_from_probs(
        probs=probs,
        n_qubits=n_qubits,
        n_shots=100,
        qubits=[0],  # Only keep qubit 0
        generator=generator,
    )
    assert samples.shape == (100, 1)
    # Should have both 0 and 1 values
    assert samples.min() == 0
    assert samples.max() == 1


def test_sample_bitstrings_state_batch() -> None:
    """Test sampling from batched states."""
    dev = default_device()
    # Batch of 2 states: |0> and |1>
    state0 = zero_state(n_qubits=1, batch_shape=None, device=dev, dtype=torch.complex64)
    state1 = torch.tensor(
        [0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64, device=dev.as_torch_device()
    )
    state_batch = torch.stack([state0, state1])  # Shape (2, 2)

    generator = torch.Generator().manual_seed(555)
    samples = sample_bitstrings_state(
        state=state_batch,
        n_qubits=1,
        n_shots=50,
        qubits=None,
        generator=generator,
    )
    # Shape (2, 50, 1)
    assert samples.shape == (2, 50, 1)
    # First batch should be all 0s
    assert torch.all(samples[0] == 0)
    # Second batch should be all 1s
    assert torch.all(samples[1] == 1)

