"""Bitstring sampling utilities for quantum states."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from qconduit.backend.statevector import measure_probs
from qconduit.backend.density_matrix import measure_probs_dm
from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device


def _indices_to_bitstrings(
    indices: torch.Tensor,
    n_qubits: int,
    qubits: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Convert integer outcome indices into bitstrings.

    Parameters
    ----------
    indices:
        Integer tensor of shape (..., n_shots) with values in [0, 2**n_qubits).
    n_qubits:
        Total number of qubits in the underlying system.
    qubits:
        Optional subsequence of qubit indices (0-based) to keep in the
        output bitstrings. If None, all qubits [0, ..., n_qubits-1] are kept.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (..., n_shots, len(qubits)) with bits
        in {0, 1}. By convention, qubit index 0 corresponds to the
        least-significant bit of the outcome index.
    """
    if qubits is None:
        qubits_tuple: Tuple[int, ...] = tuple(range(n_qubits))
    else:
        qubits_tuple = tuple(int(q) for q in qubits)
    if not qubits_tuple:
        raise ValueError("qubits must be non-empty if provided.")

    # Ensure integer type
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)

    # Shape: (..., n_shots)
    # We will broadcast a bit-mask per qubit.
    # For each qubit, bit = (index >> q) & 1.
    # Build a result tensor of shape (..., n_shots, len(qubits)).
    expanded_shape = indices.shape + (len(qubits_tuple),)
    result = torch.empty(expanded_shape, dtype=torch.int64, device=indices.device)

    for k, q in enumerate(qubits_tuple):
        if q < 0 or q >= n_qubits:
            raise ValueError(
                f"Requested qubit index {q} is out of bounds for n_qubits={n_qubits}."
            )
        # Compute this qubit's bit across all indices.
        bits = (indices >> q) & 1
        result[..., k] = bits

    return result


def sample_from_probs(
    probs: torch.Tensor,
    n_qubits: int,
    n_shots: int,
    qubits: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample bitstrings from a probability distribution over computational
    basis states.

    Parameters
    ----------
    probs:
        Tensor of shape (..., dim) with dim = 2**n_qubits containing
        non-negative values that sum to 1 along the last dimension.
        Typically obtained from measure_probs or measure_probs_dm.
    n_qubits:
        Total number of qubits for this distribution.
    n_shots:
        Number of measurement shots to draw per batch element.
    qubits:
        Optional subsequence of qubit indices to retain in the output.
        If None, all qubits [0, ..., n_qubits-1] are returned.
    generator:
        Optional torch.Generator to control randomness for reproducibility.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (..., n_shots, len(qubits)) with values
        in {0, 1}. The leading batch dimensions match those of `probs`.
    """
    if probs.dim() < 1:
        raise ValueError("probs must have at least one dimension.")
    dim = probs.shape[-1]
    if dim != 2 ** n_qubits:
        raise ValueError(
            f"probs last dimension {dim} does not match 2**n_qubits={2**n_qubits}."
        )
    if n_shots <= 0:
        raise ValueError("n_shots must be a positive integer.")

    # Normalize defensively in case of minor numerical drift.
    # Avoid division by zero: if sum is zero, raise.
    probs_flat = probs.reshape(-1, dim)
    sums = probs_flat.sum(dim=-1, keepdim=True)
    if not torch.all(sums > 0):
        raise ValueError("Probability distribution has zero total mass.")
    probs_norm = probs_flat / sums

    # Multinomial sampling per batch row.
    # We will sample with replacement n_shots outcomes per row.
    indices_flat = torch.multinomial(
        probs_norm,
        num_samples=n_shots,
        replacement=True,
        generator=generator,
    )
    # Reshape back to original batch shape: (..., n_shots)
    batch_shape = probs.shape[:-1]
    indices = indices_flat.reshape(batch_shape + (n_shots,))

    # Convert indices to bitstrings.
    return _indices_to_bitstrings(indices, n_qubits=n_qubits, qubits=qubits)


def sample_bitstrings_state(
    state: torch.Tensor,
    n_qubits: int,
    n_shots: int,
    qubits: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample bitstrings from a pure statevector via the Born rule.

    Parameters
    ----------
    state:
        Complex tensor representing the statevector, with shape
        (..., dim) and dim = 2**n_qubits.
    n_qubits:
        Number of qubits in the state.
    n_shots:
        Number of shots per batch element.
    qubits:
        Optional subset of qubits to include in the result. If None,
        all [0, ..., n_qubits-1] are included.
    generator:
        Optional torch.Generator for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (..., n_shots, len(qubits)) with bits
        in {0, 1}.
    """
    dim = state.shape[-1]
    if dim != 2 ** n_qubits:
        raise ValueError(
            f"State dimension {dim} does not match 2**n_qubits={2**n_qubits}."
        )
    probs = measure_probs(state, n_qubits=n_qubits)
    return sample_from_probs(
        probs=probs,
        n_qubits=n_qubits,
        n_shots=n_shots,
        qubits=qubits,
        generator=generator,
    )


def sample_bitstrings_dm(
    rho: torch.Tensor,
    n_qubits: int,
    n_shots: int,
    qubits: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample bitstrings from a density matrix via the Born rule.

    Parameters
    ----------
    rho:
        Complex density matrix tensor with shape (..., dim, dim) and
        dim = 2**n_qubits.
    n_qubits:
        Number of qubits in the state.
    n_shots:
        Number of shots per batch element.
    qubits:
        Optional subset of qubits to include in the result. If None,
        all [0, ..., n_qubits-1] are included.
    generator:
        Optional torch.Generator for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (..., n_shots, len(qubits)) with bits
        in {0, 1}.
    """
    if rho.dim() < 2 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho must be a square matrix in its last two dimensions.")
    dim = rho.shape[-1]
    if dim != 2 ** n_qubits:
        raise ValueError(
            f"Density matrix dimension {dim} does not match 2**n_qubits={2**n_qubits}."
        )

    probs = measure_probs_dm(rho)
    return sample_from_probs(
        probs=probs,
        n_qubits=n_qubits,
        n_shots=n_shots,
        qubits=qubits,
        generator=generator,
    )


def sample_bitstrings_circuit(
    circuit: QuantumCircuit,
    n_shots: int,
    qubits: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Simulate a QuantumCircuit and sample bitstrings from its output
    statevector.

    Parameters
    ----------
    circuit:
        QuantumCircuit to simulate from the |0...0> state.
    n_shots:
        Number of shots to sample.
    qubits:
        Optional subset of qubits to retain. If None, all qubits are used.
    device:
        Optional Device on which to run the simulation. If None, the
        default device is used.
    dtype:
        Complex dtype for state simulation (default complex64).
    generator:
        Optional torch.Generator for reproducibility.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (n_shots, len(qubits)) for non-batched
        circuits.
    """
    if device is None:
        dev = default_device()
    else:
        dev = device

    # For now, QuantumCircuit.simulate_state is assumed to return (dim,)
    # on the chosen device.
    state = circuit.simulate_state(device=dev, dtype=dtype)
    samples = sample_bitstrings_state(
        state=state,
        n_qubits=circuit.n_qubits,
        n_shots=n_shots,
        qubits=qubits,
        generator=generator,
    )
    # For a non-batched state, samples will have shape (n_shots, len(qubits))
    return samples


