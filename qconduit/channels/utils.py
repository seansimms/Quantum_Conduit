"""Utility functions for working with quantum channels and density matrices."""

from __future__ import annotations

from typing import Optional

import torch


def density_from_statevector(psi: torch.Tensor) -> torch.Tensor:
    """
    Convert a pure statevector |ψ⟩ into a density matrix |ψ⟩⟨ψ|.

    Parameters
    ----------
    psi: torch.Tensor
        Statevector tensor of shape (..., dim) with complex dtype.

    Returns
    -------
    torch.Tensor
        Density matrix tensor of shape (..., dim, dim) with complex dtype.

    Raises
    ------
    ValueError
        If psi is not complex dtype.
    """
    if not torch.is_complex(psi):
        raise ValueError(f"psi must be complex dtype, got {psi.dtype}")

    # Compute outer product: |ψ⟩⟨ψ|
    # psi: (..., dim)
    # psi.unsqueeze(-1): (..., dim, 1)
    # psi.conj().unsqueeze(-2): (..., 1, dim)
    # Result: (..., dim, dim)
    rho = psi.unsqueeze(-1) @ psi.conj().unsqueeze(-2)

    # Ensure complex128 dtype
    if rho.dtype != torch.complex128:
        rho = rho.to(dtype=torch.complex128)

    return rho


def statevector_from_density_sampling(
    rho: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample a pure statevector from a mixed density matrix.

    Uses spectral decomposition: diagonalize rho = V diag(p) V^†, sample
    an eigenvector according to probabilities p, and return the normalized
    eigenvector.

    Parameters
    ----------
    rho: torch.Tensor
        Density matrix tensor of shape (dim, dim) with complex dtype.
    generator: Optional[torch.Generator]
        Random number generator for deterministic sampling. If None, creates
        one with seed 0.

    Returns
    -------
    torch.Tensor
        Sampled pure statevector of shape (dim,) with complex dtype.

    Raises
    ------
    ValueError
        If rho is not square, not Hermitian, or has invalid trace.
    """
    if rho.dim() != 2:
        raise ValueError(f"rho must be 2D, got {rho.dim()} dimensions")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be square, got shape {rho.shape}")
    if not torch.is_complex(rho):
        raise ValueError(f"rho must be complex dtype, got {rho.dtype}")

    device = rho.device

    # Ensure Hermitian (rho should be Hermitian, but numerical errors may occur)
    rho_herm = (rho + rho.conj().T) / 2.0

    # Diagonalize: rho = V @ diag(eigenvals) @ V^†
    eigenvals, eigenvecs = torch.linalg.eigh(rho_herm)

    # Eigenvals should be probabilities (non-negative, sum to ~1)
    eigenvals = torch.clamp(eigenvals.real, min=0.0)
    # Normalize probabilities
    prob_sum = eigenvals.sum()
    if prob_sum < 1e-12:
        raise ValueError("Density matrix has zero trace")
    probs = eigenvals / prob_sum

    # Sample eigenvector index
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

    idx = torch.multinomial(probs, num_samples=1, generator=generator).item()

    # Return normalized eigenvector
    psi = eigenvecs[:, idx]
    norm = torch.norm(psi)
    if norm < 1e-12:
        raise ValueError(f"Sampled eigenvector {idx} has zero norm")
    psi = psi / norm

    return psi


def is_density_matrix(rho: torch.Tensor, atol: float = 1e-10) -> bool:
    """
    Check if a tensor represents a valid density matrix.

    Validates:
        - Square matrix
        - Hermitian (within tolerance)
        - Trace ≈ 1 (within tolerance)
        - Positive semidefinite (eigenvalues >= 0)

    Parameters
    ----------
    rho: torch.Tensor
        Tensor to check, shape (dim, dim) with complex dtype.
    atol: float
        Absolute tolerance for checks.

    Returns
    -------
    bool
        True if rho is a valid density matrix within tolerance.
    """
    if rho.dim() != 2:
        return False
    if rho.shape[0] != rho.shape[1]:
        return False
    if not torch.is_complex(rho):
        return False

    # Check Hermiticity: rho ≈ rho^†
    rho_herm = (rho + rho.conj().T) / 2.0
    diff_herm = torch.abs(rho - rho_herm)
    if torch.max(diff_herm).item() > atol:
        return False

    # Check trace ≈ 1
    trace = torch.trace(rho).real
    if abs(trace - 1.0) > atol:
        return False

    # Check positive semidefinite: all eigenvalues >= -atol
    eigenvals = torch.linalg.eigvalsh(rho_herm)
    if torch.min(eigenvals).item() < -atol:
        return False

    return True


__all__ = [
    "density_from_statevector",
    "statevector_from_density_sampling",
    "is_density_matrix",
]

