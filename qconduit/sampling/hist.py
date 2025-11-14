"""Histogram and probability analysis utilities for bitstring samples."""

from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import torch


def _bits_to_str(bits: torch.Tensor) -> str:
    """
    Convert a 1D tensor of bits (0 or 1) to a bitstring like '0101'.

    Parameters
    ----------
    bits:
        1D tensor of bits (0 or 1).

    Returns
    -------
    str
        Bitstring representation, e.g. '0101'.
    """
    if bits.dim() != 1:
        raise ValueError("bits_to_str expects a 1D tensor of bits.")
    # Ensure on CPU for iteration
    cpu_bits = bits.to(torch.int64).detach().cpu()
    return "".join(str(int(b.item())) for b in cpu_bits)


def bitstring_counts(
    samples: torch.Tensor,
) -> Dict[str, int]:
    """
    Compute counts for each unique bitstring in a batch of samples.

    Parameters
    ----------
    samples:
        Integer tensor of shape (n_shots, n_bits) or (..., n_shots, n_bits)
        with entries in {0, 1}. If the tensor has batch dimensions, counts
        are aggregated across all batch elements.

    Returns
    -------
    Dict[str, int]
        Mapping from bitstring (e.g. '010') to the number of occurrences.
    """
    if samples.dim() < 2:
        raise ValueError("samples must have at least 2 dimensions (shots, bits).")
    # Flatten batch dimensions into one set of shots
    n_bits = samples.shape[-1]
    flat = samples.reshape(-1, n_bits)

    counts: Dict[str, int] = {}
    for i in range(flat.shape[0]):
        key = _bits_to_str(flat[i])
        counts[key] = counts.get(key, 0) + 1
    return counts


def counts_to_probs(
    counts: Mapping[str, int],
) -> Dict[str, float]:
    """
    Convert integer counts for bitstrings into a probability distribution.

    Parameters
    ----------
    counts:
        Mapping from bitstring to non-negative integer count.

    Returns
    -------
    Dict[str, float]
        Mapping from bitstring to probability, summing to 1.0.
    """
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Total count must be positive.")
    return {k: v / float(total) for k, v in counts.items()}


def kl_divergence(
    p: Mapping[str, float],
    q: Mapping[str, float],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute the Kullback-Leibler divergence KL(p || q) for discrete
    distributions over bitstrings.

    Parameters
    ----------
    p, q:
        Mappings from bitstring to probability. Keys not present in q
        are treated as having zero probability. p is normalized before
        computing KL; q is also renormalized defensively.
    epsilon:
        Small value to clamp probabilities away from zero to avoid
        log(0). This keeps the function numerically stable but means
        it is not an exact mathematical KL divergence if supports differ.

    Returns
    -------
    float
        KL(p || q) in natural units (nats).
    """
    # Normalize p and q defensively
    sum_p = sum(p.values())
    sum_q = sum(q.values())
    if sum_p <= 0 or sum_q <= 0:
        raise ValueError("Distributions p and q must have positive total probability.")

    kl = 0.0
    for key, p_val in p.items():
        p_prob = p_val / sum_p
        q_prob = q.get(key, 0.0) / sum_q
        # Clamp
        p_clamped = max(p_prob, epsilon)
        q_clamped = max(q_prob, epsilon)
        kl += p_clamped * math.log(p_clamped / q_clamped)
    return kl


def marginalize_probs(
    probs: torch.Tensor,
    n_qubits: int,
    qubits_to_keep: Sequence[int],
) -> torch.Tensor:
    """
    Compute the marginal probability distribution over a subset of qubits.

    Parameters
    ----------
    probs:
        Tensor of shape (..., dim) with dim = 2**n_qubits, representing
        probabilities over all computational basis states.
    n_qubits:
        Total number of qubits.
    qubits_to_keep:
        Sequence of qubit indices to retain. The output distribution
        will be over bitstrings ordered according to this sequence.

    Returns
    -------
    torch.Tensor
        Tensor of shape (..., 2**len(qubits_to_keep)) containing the
        marginal probabilities.

    Notes
    -----
    This implementation uses explicit integer indexing and is intended
    for small to moderate n_qubits. It is strictly textbook plumbing,
    not an optimized large-scale marginalization engine.
    """
    if probs.dim() < 1:
        raise ValueError("probs must have at least one dimension.")
    dim = probs.shape[-1]
    if dim != 2 ** n_qubits:
        raise ValueError(
            f"probs last dimension {dim} does not match 2**n_qubits={2**n_qubits}."
        )

    keep = tuple(int(q) for q in qubits_to_keep)
    if not keep:
        raise ValueError("qubits_to_keep must be non-empty.")

    for q in keep:
        if q < 0 or q >= n_qubits:
            raise ValueError(
                f"Qubit index {q} in qubits_to_keep is out of bounds for "
                f"n_qubits={n_qubits}."
            )

    batch_shape = probs.shape[:-1]
    k = len(keep)
    out_dim = 2 ** k

    # Initialize marginals to zero
    marg = probs.new_zeros(batch_shape + (out_dim,))

    # For each global basis index i, determine its reduced index
    # on the kept qubits and accumulate.
    # We iterate over all 2**n_qubits outcomes.
    # For small n_qubits this is acceptable and very clear.
    for i in range(dim):
        # Compute reduced index j for the kept qubits
        j = 0
        for bit_pos, q in enumerate(keep):
            bit = (i >> q) & 1
            j |= bit << bit_pos
        marg[..., j] += probs[..., i]
    return marg


