"""Density matrix utilities for applying quantum channels.

This module provides functions to work with pure states and density matrices,
and to apply Kraus channels to them.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from qconduit.core.device import default_device
from qconduit.noise.kraus import KrausChannel


def to_density_matrix(
    state: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Construct a density matrix ρ = |ψ><ψ| from a pure statevector |ψ>.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ>.
    device:
        Optional device for the output. Defaults to the state's device or
        `default_device()`.
    dtype:
        Optional complex dtype for the output. Defaults to the state's dtype
        or torch.complex128.

    Returns
    -------
    torch.Tensor
        2D tensor of shape (dim, dim) with ρ = |ψ><ψ|.

    Raises
    ------
    ValueError
        If state is not 1D, or if dimension is not a power of 2.
    """
    # Validate state is 1D
    if state.ndim != 1:
        raise ValueError(f"state must be 1D, got shape {state.shape}")

    # Determine dimension
    dim = state.shape[0]

    # Validate dimension is a power of 2
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(
            f"state dimension {dim} is not a power of 2. "
            "Expected shape (2**n,) for some n >= 1."
        )

    # Determine target device and dtype
    if device is None:
        device = state.device if hasattr(state, "device") else default_device().as_torch_device()
    if dtype is None:
        dtype = state.dtype if torch.is_complex(state) else torch.complex128

    # Move/cast state to desired device/dtype
    state = state.to(device=device, dtype=dtype)

    # Compute outer product: ρ = |ψ><ψ|
    # state: (dim,)
    # state.unsqueeze(1): (dim, 1)
    # state.conj().unsqueeze(0): (1, dim)
    # Result: (dim, dim)
    rho = state.unsqueeze(1) @ state.conj().unsqueeze(0)

    return rho


def apply_kraus_channel_to_density_matrix(
    rho: torch.Tensor,
    channel: KrausChannel,
) -> torch.Tensor:
    """
    Apply a KrausChannel to a density matrix ρ:

        ρ' = ∑_i K_i ρ K_i†.

    This helper assumes the channel acts on the full system, i.e.

        channel.num_qubits == n_qubits,

    where n_qubits is inferred from the dimension of ρ.

    Parameters
    ----------
    rho:
        2D complex tensor of shape (dim, dim), dim = 2**n.
    channel:
        KrausChannel with num_qubits == n.

    Returns
    -------
    torch.Tensor
        The updated density matrix ρ' with the same shape and dtype as ρ.

    Raises
    ------
    ValueError
        If rho is not 2D square, dimension is not a power of 2, or channel
        qubit count does not match density matrix dimension.
    """
    # Validate rho is 2D and square
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2D, got shape {rho.shape}")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be square, got shape {rho.shape}")

    # Determine dimension and infer n_qubits
    dim = rho.shape[0]
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(
            f"rho dimension {dim} is not a power of 2. "
            "Expected shape (2**n, 2**n) for some n >= 1."
        )

    # Ensure channel.num_qubits matches
    if channel.num_qubits != n_qubits:
        raise ValueError(
            f"Channel qubit count ({channel.num_qubits}) does not match "
            f"density matrix dimension (2**{n_qubits} = {dim})."
        )

    # Ensure rho is complex
    if not torch.is_complex(rho):
        rho = rho.to(dtype=torch.complex128)

    # Ensure channel Kraus operators are on same device/dtype as rho
    device = rho.device
    dtype = rho.dtype
    channel = channel.to(device=device, dtype=dtype)

    # Apply channel: ρ' = ∑_i K_i ρ K_i†
    rho_out = torch.zeros_like(rho)
    for K in channel.kraus_ops:
        # K @ rho @ K.conj().T
        K_rho = K @ rho
        K_rho_Kdag = K_rho @ K.conj().T
        rho_out = rho_out + K_rho_Kdag

    return rho_out


def apply_kraus_channel_to_statevector(
    state: torch.Tensor,
    channel: KrausChannel,
) -> torch.Tensor:
    """
    Apply a KrausChannel to a pure statevector |ψ> by:

        ρ = |ψ><ψ|,
        ρ' = E(ρ) = ∑_i K_i ρ K_i†,

    and returning the resulting density matrix ρ'.

    Generic noise channels map pure states to mixed states, so this function
    always returns a density matrix, not a statevector.

    Parameters
    ----------
    state:
        1D complex tensor of shape (2**n,) representing |ψ>.
    channel:
        KrausChannel with num_qubits == n.

    Returns
    -------
    torch.Tensor
        Density matrix ρ' of shape (dim, dim).
    """
    # Convert statevector to density matrix
    rho = to_density_matrix(state)

    # Apply channel
    rho_out = apply_kraus_channel_to_density_matrix(rho, channel)

    return rho_out


def compose_kraus_channels(
    first: KrausChannel,
    second: KrausChannel,
) -> KrausChannel:
    """
    Compose two Kraus channels E2 ∘ E1 acting on the same system size.

    If E1 has Kraus {K_i} and E2 has {L_j}, then the composite channel has
    Kraus operators {L_j K_i}:

        (E2 ∘ E1)(ρ) = ∑_{i,j} L_j K_i ρ K_i† L_j†.

    Parameters
    ----------
    first:
        Channel E1 applied first.
    second:
        Channel E2 applied after E1.

    Returns
    -------
    KrausChannel
        Composite channel E2 ∘ E1.

    Raises
    ------
    ValueError
        If the channels act on different numbers of qubits.
    """
    # Require same number of qubits
    if first.num_qubits != second.num_qubits:
        raise ValueError(
            f"Cannot compose channels with different qubit counts: "
            f"first has {first.num_qubits} qubits, second has {second.num_qubits} qubits."
        )

    # Ensure both channels are on same device/dtype
    # Use first channel's device/dtype as reference
    device = first.kraus_ops[0].device
    dtype = first.kraus_ops[0].dtype
    first = first.to(device=device, dtype=dtype)
    second = second.to(device=device, dtype=dtype)

    # Build composite Kraus operators: L_j K_i for all pairs
    composite_ops = []
    for K in first.kraus_ops:
        for L in second.kraus_ops:
            # L @ K
            composite_op = L @ K
            composite_ops.append(composite_op)

    # Create composite channel
    composite_name = f"{second.name}∘{first.name}"
    return KrausChannel(
        name=composite_name,
        kraus_ops=tuple(composite_ops),
        num_qubits=first.num_qubits,
    )


__all__ = [
    "to_density_matrix",
    "apply_kraus_channel_to_density_matrix",
    "apply_kraus_channel_to_statevector",
    "compose_kraus_channels",
]


