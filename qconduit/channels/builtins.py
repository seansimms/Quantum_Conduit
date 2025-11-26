"""Built-in textbook quantum noise channels."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from qconduit.channels.core import KrausChannel
from qconduit.core.device import default_device
from qconduit.gates.standard import I, X, Y, Z


def DepolarizingChannel(
    p: float, device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Single-qubit depolarizing channel with depolarizing probability p.

    Kraus operators:
        K_0 = sqrt(1 - p) I
        K_1 = sqrt(p/3) X
        K_2 = sqrt(p/3) Y
        K_3 = sqrt(p/3) Z

    Parameters
    ----------
    p: float
        Depolarizing probability in [0, 1].
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().

    Returns
    -------
    KrausChannel
        A single-qubit depolarizing channel (n_qubits=1).

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Depolarizing probability p must be in [0, 1], got {p}")

    if device is None:
        device = default_device().as_torch_device()

    dtype = torch.complex128

    i_mat = I(dtype=dtype, device=device)
    x_mat = X(dtype=dtype, device=device)
    y_mat = Y(dtype=dtype, device=device)
    z_mat = Z(dtype=dtype, device=device)

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p_over_3 = torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float64)).to(
        dtype=dtype
    )

    k0 = sqrt_one_minus_p * i_mat
    k1 = sqrt_p_over_3 * x_mat
    k2 = sqrt_p_over_3 * y_mat
    k3 = sqrt_p_over_3 * z_mat

    return KrausChannel(kraus_ops=(k0, k1, k2, k3), n_qubits=1)


def BitFlipChannel(
    p: float, device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Single-qubit bit-flip channel with flip probability p.

    Kraus operators:
        K_0 = sqrt(1 - p) I
        K_1 = sqrt(p) X

    Parameters
    ----------
    p: float
        Bit-flip probability in [0, 1].
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().

    Returns
    -------
    KrausChannel
        A single-qubit bit-flip channel (n_qubits=1).

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Bit-flip probability p must be in [0, 1], got {p}")

    if device is None:
        device = default_device().as_torch_device()

    dtype = torch.complex128

    i_mat = I(dtype=dtype, device=device)
    x_mat = X(dtype=dtype, device=device)

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float64)).to(dtype=dtype)

    k0 = sqrt_one_minus_p * i_mat
    k1 = sqrt_p * x_mat

    return KrausChannel(kraus_ops=(k0, k1), n_qubits=1)


def PhaseFlipChannel(
    p: float, device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Single-qubit phase-flip channel with flip probability p.

    Kraus operators:
        K_0 = sqrt(1 - p) I
        K_1 = sqrt(p) Z

    Parameters
    ----------
    p: float
        Phase-flip probability in [0, 1].
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().

    Returns
    -------
    KrausChannel
        A single-qubit phase-flip channel (n_qubits=1).

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Phase-flip probability p must be in [0, 1], got {p}")

    if device is None:
        device = default_device().as_torch_device()

    dtype = torch.complex128

    i_mat = I(dtype=dtype, device=device)
    z_mat = Z(dtype=dtype, device=device)

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float64)).to(dtype=dtype)

    k0 = sqrt_one_minus_p * i_mat
    k1 = sqrt_p * z_mat

    return KrausChannel(kraus_ops=(k0, k1), n_qubits=1)


def PhaseDampingChannel(
    p: float, device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Single-qubit phase-damping (dephasing) channel with probability p.

    Kraus operators:
        K_0 = [[1, 0],
               [0, sqrt(1 - p)]]
        K_1 = [[0, 0],
               [0, sqrt(p)]]

    This damps off-diagonal coherence without changing populations.

    Parameters
    ----------
    p: float
        Phase damping probability in [0, 1].
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().

    Returns
    -------
    KrausChannel
        A single-qubit phase-damping channel (n_qubits=1).

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Phase damping probability p must be in [0, 1], got {p}")

    if device is None:
        device = default_device().as_torch_device()

    dtype = torch.complex128

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float64)).to(dtype=dtype)

    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = 1.0
    k0[1, 1] = sqrt_one_minus_p

    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[1, 1] = sqrt_p

    return KrausChannel(kraus_ops=(k0, k1), n_qubits=1)


def AmplitudeDampingChannel(
    gamma: float, device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Single-qubit amplitude-damping channel with relaxation probability gamma.

    Kraus operators:
        K_0 = [[1, 0],
               [0, sqrt(1 - gamma)]]
        K_1 = [[0, sqrt(gamma)],
               [0, 0]]

    This models decay |1⟩ -> |0⟩ with probability gamma.

    Parameters
    ----------
    gamma: float
        Amplitude damping parameter in [0, 1].
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().

    Returns
    -------
    KrausChannel
        A single-qubit amplitude-damping channel (n_qubits=1).

    Raises
    ------
    ValueError
        If gamma is not in [0, 1].
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(
            f"Amplitude damping parameter gamma must be in [0, 1], got {gamma}"
        )

    if device is None:
        device = default_device().as_torch_device()

    dtype = torch.complex128

    sqrt_one_minus_gamma = torch.sqrt(
        torch.tensor(1.0 - gamma, dtype=torch.float64)
    ).to(dtype=dtype)
    sqrt_gamma = torch.sqrt(torch.tensor(gamma, dtype=torch.float64)).to(dtype=dtype)

    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = 1.0
    k0[1, 1] = sqrt_one_minus_gamma

    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[0, 1] = sqrt_gamma

    return KrausChannel(kraus_ops=(k0, k1), n_qubits=1)


def GeneralKraus(
    kraus_list: Sequence[torch.Tensor], device: Optional[torch.device] = None
) -> KrausChannel:
    """
    Construct a KrausChannel from a list of Kraus operators.

    This is a convenience wrapper that validates inputs and constructs
    a KrausChannel. The Kraus operators must all have the same shape
    (dim, dim) where dim is a power of 2.

    Parameters
    ----------
    kraus_list: Sequence[torch.Tensor]
        List of Kraus operator tensors, each of shape (dim, dim) where
        dim = 2**n_qubits for some n_qubits >= 1.
    device: Optional[torch.device]
        Device for the Kraus operators. If None, uses default_device().
        Operators will be moved to this device.

    Returns
    -------
    KrausChannel
        A channel with the given Kraus operators.

    Raises
    ------
    ValueError
        If kraus_list is empty, operators have invalid shapes, or
        operators don't all have the same shape.
    """
    if len(kraus_list) == 0:
        raise ValueError("kraus_list must contain at least one operator")

    # Check all operators have the same shape
    first_shape = kraus_list[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError(
            f"All Kraus operators must be square matrices, "
            f"got shape {first_shape}"
        )

    dim = first_shape[0]
    # Check dim is a power of 2
    n_qubits = int(math.log2(dim))
    if 1 << n_qubits != dim:
        raise ValueError(
            f"Kraus operator dimension {dim} is not a power of 2. "
            f"Expected 2**n for some n >= 1."
        )

    for i, kraus_op in enumerate(kraus_list):
        if kraus_op.shape != first_shape:
            raise ValueError(
                f"Kraus operator {i} has shape {kraus_op.shape}, "
                f"but expected {first_shape}"
            )

    # Convert to tuple and construct channel
    kraus_tuple = tuple(kraus_list)
    return KrausChannel(kraus_ops=kraus_tuple, n_qubits=n_qubits)


__all__ = [
    "DepolarizingChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "PhaseDampingChannel",
    "AmplitudeDampingChannel",
    "GeneralKraus",
]



