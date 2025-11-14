"""Core diagnostic functions for quantum states and operators."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def state_norm(state: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm of a quantum state tensor.

    This is a generic helper for both statevectors and flattened
    states. It assumes the last dimension corresponds to the
    Hilbert-space components.

    Parameters
    ----------
    state:
        Complex tensor with shape (..., dim).

    Returns
    -------
    torch.Tensor
        Real tensor with shape (...) giving the norm for each batch
        element.

    Raises
    ------
    ValueError
        If state has fewer than 1 dimension.
    """
    if state.dim() < 1:
        raise ValueError("state_norm expects a tensor with at least 1 dimension.")

    # Norm = sqrt(<psi|psi>)
    norm_sq = (state.conj() * state).sum(dim=-1).real
    return torch.sqrt(norm_sq)


def assert_normalized(
    state: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    """
    Assert that a statevector has norm ~1 within a tolerance.

    Parameters
    ----------
    state:
        Complex statevector tensor (..., dim).
    atol:
        Absolute tolerance for |norm - 1|.

    Raises
    ------
    ValueError
        If the state is not normalized within the tolerance.
    """
    norms = state_norm(state)
    if not torch.all(torch.isfinite(norms)):
        raise ValueError("State norm contains non-finite values.")

    if not torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=0.0):
        raise ValueError(
            f"State is not normalized within tolerance {atol}. "
            f"Norms found: {norms.detach().cpu().tolist()}"
        )


def is_hermitian(
    mat: torch.Tensor,
    atol: float = 1e-6,
) -> bool:
    """
    Check whether a matrix (or batch of matrices) is Hermitian.

    Parameters
    ----------
    mat:
        Complex tensor with shape (..., n, n).
    atol:
        Absolute tolerance for checking equality.

    Returns
    -------
    bool
        True if mat is Hermitian within the tolerance, False otherwise.
    """
    if mat.dim() < 2 or mat.shape[-1] != mat.shape[-2]:
        return False

    diff = mat - mat.conj().transpose(-2, -1)
    max_dev = diff.abs().max()
    if not torch.isfinite(max_dev):
        return False

    return bool(max_dev <= atol)


def assert_hermitian(
    mat: torch.Tensor,
    atol: float = 1e-6,
) -> None:
    """
    Assert that a matrix (or batch of matrices) is Hermitian.

    Parameters
    ----------
    mat:
        Complex tensor with shape (..., n, n).
    atol:
        Absolute tolerance for checking equality.

    Raises
    ------
    ValueError
        If the matrix is not Hermitian within the tolerance.
    """
    if not is_hermitian(mat, atol=atol):
        raise ValueError(
            f"Matrix is not Hermitian within tolerance {atol}."
        )


def fidelity(
    state_a: torch.Tensor,
    state_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fidelity between two *pure* states or two *density matrices*
    in simple textbook cases.

    For pure statevectors |psi>, |phi> (shape (..., dim)):

        F = |<psi|phi>|^2.

    For density matrices rho, sigma that commute and are diagonal in the
    same basis (shape (..., dim, dim)):

        F = (sum_i sqrt(p_i q_i))^2 where p_i, q_i are diagonal entries.

    This helper intentionally supports only these simple cases, which
    are easy to re-derive and reason about. It is not a general
    implementation of Uhlmann fidelity.

    Parameters
    ----------
    state_a:
        First state (pure statevector or diagonal density matrix).
    state_b:
        Second state (pure statevector or diagonal density matrix).

    Returns
    -------
    torch.Tensor
        Real tensor with shape (...) containing fidelity values.

    Raises
    ------
    ValueError
        If state shapes do not match or states are invalid.
    """
    if state_a.shape != state_b.shape:
        raise ValueError("fidelity expects tensors with the same shape.")

    if state_a.dim() >= 2 and state_a.shape[-1] == state_a.shape[-2]:
        # Density-matrix path (diagonal commuting assumption)
        diag_a = state_a.diagonal(dim1=-2, dim2=-1).real
        diag_b = state_b.diagonal(dim1=-2, dim2=-1).real

        if diag_a.shape != diag_b.shape:
            raise ValueError("fidelity: diagonal shapes do not match.")

        # Clamp small negatives due to numerical noise
        diag_a_clamped = torch.clamp(diag_a, min=0.0)
        diag_b_clamped = torch.clamp(diag_b, min=0.0)

        overlap = torch.sqrt(diag_a_clamped) * torch.sqrt(diag_b_clamped)
        return (overlap.sum(dim=-1)) ** 2

    if state_a.dim() < 1:
        raise ValueError("fidelity expects at least 1D tensors for pure states.")

    # Pure statevector path
    inner = (state_a.conj() * state_b).sum(dim=-1)
    return (inner.abs()) ** 2


def bloch_vector(state: torch.Tensor) -> torch.Tensor:
    """
    Compute the Bloch vector (x, y, z) for a single-qubit state.

    For a pure statevector |psi> = [a, b]^T:

        x = 2 Re(a* conj(b))
        y = 2 Im(a* conj(b))
        z = |a|^2 - |b|^2

    Parameters
    ----------
    state:
        Complex tensor with shape (..., 2) representing a single-qubit
        statevector. The last dimension must be 2.

    Returns
    -------
    torch.Tensor
        Real tensor of shape (..., 3) with Bloch components (x, y, z).

    Raises
    ------
    ValueError
        If state does not have last dimension of size 2.
    """
    if state.shape[-1] != 2:
        raise ValueError(
            "bloch_vector requires a single-qubit state with last dimension 2."
        )

    a = state[..., 0]
    b = state[..., 1]
    a_conj = a.conj()

    # x = 2 Re(a* b)
    x = 2.0 * (a_conj * b).real

    # y = 2 Im(a* b)
    y = 2.0 * (a_conj * b).imag

    # z = |a|^2 - |b|^2
    z = (a.abs() ** 2) - (b.abs() ** 2)

    return torch.stack([x, y, z], dim=-1)


