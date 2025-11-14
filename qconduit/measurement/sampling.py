"""Sampling and measurement utilities for quantum statevectors."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from qconduit.core.device import default_device


def _infer_num_qubits_from_state(state: torch.Tensor) -> int:
    """
    Infer number of qubits n from a statevector of length 2**n.

    Raises ValueError if the state is not 1D or length is not a power of 2.

    Parameters
    ----------
    state:
        1D tensor representing a statevector.

    Returns
    -------
    int
        Number of qubits n such that state.shape[0] == 2**n.

    Raises
    ------
    ValueError
        If state is not 1D or length is not a power of 2.
    """
    if state.ndim != 1:
        raise ValueError("Statevector must be a 1D tensor.")
    
    dim = state.shape[0]
    if dim <= 0:
        raise ValueError(f"Statevector must have positive length, got {dim}.")
    
    # Check if dim is a power of 2: dim & (dim - 1) == 0 for powers of 2
    if dim & (dim - 1) != 0:
        raise ValueError(f"Statevector length must be a power of 2, got {dim}.")
    
    n_qubits = int(dim).bit_length() - 1
    return n_qubits


def basis_probabilities_from_statevector(
    state: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Compute computational-basis measurement probabilities from a pure statevector.

    The probabilities are computed using the Born rule:

        p(x) = |⟨x|ψ⟩|^2,

    where |ψ⟩ is given as a 1D statevector.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ⟩.
    device:
        Optional device for the returned probabilities. Defaults to the state's
        device or `default_device()`.
    dtype:
        Optional floating-point dtype for the probabilities. Defaults to
        torch.float64.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (2**n,) with non-negative entries summing to 1.0
        (up to numerical tolerance).

    Raises
    ------
    ValueError
        If state is not 1D, length is not a power of 2, or state has zero norm.
    """
    n_qubits = _infer_num_qubits_from_state(state)
    
    # Infer device
    if device is None:
        if state.device.type == "meta":
            device = default_device().as_torch_device()
        else:
            device = state.device
    else:
        device = device
    
    # Infer dtype
    if dtype is None:
        dtype = torch.float64
    
    # Cast state to complex dtype on the chosen device
    state_complex = state.to(dtype=torch.complex128, device=device)
    
    # Compute probabilities: |state|^2
    probs = state_complex.abs() ** 2
    
    # Normalize
    total = probs.sum()
    if total == 0:
        raise ValueError("Statevector has zero norm.")
    
    probs = (probs / total).to(device=device, dtype=dtype)
    
    return probs


def sample_bitstrings_from_probabilities(
    probs: torch.Tensor,
    num_qubits: int,
    n_shots: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample computational-basis bitstrings from a given probability distribution.

    Parameters
    ----------
    probs:
        1D tensor of shape (2**num_qubits,) with non-negative entries summing
        to ~1.
    num_qubits:
        Number of qubits (>= 1).
    n_shots:
        Number of measurement shots to draw (>= 1).
    generator:
        Optional torch.Generator for deterministic sampling. If None, a new
        generator on the default device is created with seed 0.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape (n_shots, num_qubits), where each row encodes
        a bitstring (b_0, ..., b_{num_qubits-1}) with bits 0 or 1.
        Column 0 corresponds to the most significant bit (qubit 0).

    Raises
    ------
    ValueError
        If num_qubits < 1, n_shots < 1, probs has wrong shape, or probabilities
        are invalid.
    """
    # Validate inputs
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if n_shots < 1:
        raise ValueError(f"n_shots must be >= 1, got {n_shots}")
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got shape {probs.shape}")
    
    expected_dim = 1 << num_qubits
    if probs.shape[0] != expected_dim:
        raise ValueError(
            f"probs.shape[0] must be 2**num_qubits = {expected_dim}, "
            f"got {probs.shape[0]}"
        )
    
    # Normalize probabilities defensively
    probs_f = probs.to(dtype=torch.float64)
    
    # Check for negative entries (with tolerance)
    if torch.any(probs_f < -1e-14):
        raise ValueError("Probabilities contain negative values.")
    
    # Clamp to non-negative
    probs_f = torch.clamp(probs_f, min=0.0)
    
    # Normalize
    total = probs_f.sum()
    if total <= 0:
        raise ValueError("Probabilities sum to zero or negative value.")
    
    probs_norm = probs_f / total
    
    # Choose device
    device = probs_norm.device
    
    # Create generator if needed
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
    
    # Sample indices using multinomial
    indices = torch.multinomial(
        probs_norm, n_shots, replacement=True, generator=generator
    )
    # indices.shape == (n_shots,)
    
    # Convert integer indices to bitstrings (MSB-first)
    # For index j, we want bitstring [b_0, ..., b_{n-1}] where
    # j = sum_{k=0}^{n-1} b_k * 2**(num_qubits-1-k)
    # So b_0 is the most significant bit
    bits = torch.zeros((n_shots, num_qubits), dtype=torch.int64, device=device)
    for q in range(num_qubits):
        bits[:, num_qubits - 1 - q] = (indices >> q) & 1
    
    return bits


def sample_bitstrings_from_statevector(
    state: torch.Tensor,
    n_shots: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample computational-basis bitstrings from a pure statevector.

    This uses the Born rule:

        p(x) = |⟨x|ψ⟩|^2,

    and returns both the sampled bitstrings and the underlying probabilities.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ⟩.
    n_shots:
        Number of samples to draw (>= 1).
    generator:
        Optional torch.Generator for deterministic sampling.

    Returns
    -------
    bitstrings:
        Integer tensor of shape (n_shots, n_qubits) with bits 0/1
        (MSB-first in the columns).
    probs:
        1D tensor of shape (2**n,) with the probabilities used for sampling.

    Raises
    ------
    ValueError
        If state is invalid or n_shots < 1.
    """
    n_qubits = _infer_num_qubits_from_state(state)
    
    # Compute probabilities
    probs = basis_probabilities_from_statevector(state)
    
    # Sample bitstrings
    bitstrings = sample_bitstrings_from_probabilities(
        probs, num_qubits=n_qubits, n_shots=n_shots, generator=generator
    )
    
    return bitstrings, probs


def bitstring_counts(
    bitstrings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute counts for each computational-basis bitstring from a sample array.

    Parameters
    ----------
    bitstrings:
        Integer tensor of shape (n_shots, n_qubits) with entries 0 or 1.

    Returns
    -------
    torch.Tensor
        1D tensor of length 2**n_qubits with integer counts. Index j corresponds
        to the bitstring encoding of j in binary (MSB-first as in sampling).

    Raises
    ------
    ValueError
        If bitstrings is not 2D or contains invalid entries (not 0 or 1).
    """
    # Validate
    if bitstrings.ndim != 2:
        raise ValueError(f"bitstrings must be 2D, got shape {bitstrings.shape}")
    
    # Check all entries are 0 or 1
    if torch.any((bitstrings != 0) & (bitstrings != 1)):
        raise ValueError("bitstrings must contain only 0 and 1")
    
    n_shots, n_qubits = bitstrings.shape
    
    # Convert each bitstring row to an integer index
    # Row [b_0, ..., b_{n-1}] -> j = sum_{k=0}^{n-1} b_k * 2**(n_qubits-1-k)
    indices = torch.zeros(n_shots, dtype=torch.int64, device=bitstrings.device)
    for q in range(n_qubits):
        indices += bitstrings[:, q].to(dtype=torch.int64) * (1 << (n_qubits - 1 - q))
    
    # Count occurrences
    max_index = (1 << n_qubits) - 1
    counts = torch.bincount(indices, minlength=1 << n_qubits)
    
    return counts


def empirical_probabilities_from_bitstrings(
    bitstrings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute empirical probabilities for each bitstring from sampled data.

    Parameters
    ----------
    bitstrings:
        Integer tensor of shape (n_shots, n_qubits) with entries 0 or 1.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (2**n_qubits,) with entries summing to 1.0.

    Raises
    ------
    ValueError
        If bitstrings is invalid or empty.
    """
    counts = bitstring_counts(bitstrings)
    
    total = counts.sum()
    if total == 0:
        raise ValueError("bitstrings is empty (no samples).")
    
    probs = counts.to(dtype=torch.float64) / total
    return probs


def estimate_pauli_z_expectation_from_samples(
    bitstrings: torch.Tensor,
    qubit_index: int,
) -> Tuple[float, float]:
    """
    Estimate the expectation value of Z on a single qubit from computational-basis samples.

    For each shot, the eigenvalue of Z on qubit i is:

        +1 if the bit is 0,
        -1 if the bit is 1.

    The estimator is the sample mean of these eigenvalues, with an associated
    (unbiased) standard error.

    Parameters
    ----------
    bitstrings:
        Integer tensor of shape (n_shots, n_qubits) with entries 0 or 1.
    qubit_index:
        Index of the qubit (0-based, 0 <= qubit_index < n_qubits).

    Returns
    -------
    mean:
        Estimated ⟨Z_i⟩ (float).
    stderr:
        Estimated standard error of the mean (float), or 0.0 if n_shots == 1.

    Raises
    ------
    ValueError
        If bitstrings is invalid or qubit_index is out of range.
    """
    # Validate
    if bitstrings.ndim != 2:
        raise ValueError(f"bitstrings must be 2D, got shape {bitstrings.shape}")
    
    n_shots, n_qubits = bitstrings.shape
    
    if qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError(
            f"qubit_index must be in [0, {n_qubits}), got {qubit_index}"
        )
    
    # Extract bits for the given qubit
    bits = bitstrings[:, qubit_index]
    
    # Ensure bits are 0/1
    if torch.any((bits != 0) & (bits != 1)):
        raise ValueError("bitstrings must contain only 0 and 1")
    
    # Map to eigenvalues: 0 -> +1, 1 -> -1
    vals = 1.0 - 2.0 * bits.to(dtype=torch.float64)
    
    # Compute mean
    mean = float(vals.mean())
    
    # Compute standard error
    if n_shots < 2:
        stderr = 0.0
    else:
        var = float(vals.var(unbiased=True))
        stderr = math.sqrt(var / n_shots)
    
    return mean, stderr


