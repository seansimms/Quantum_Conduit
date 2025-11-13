"""Standard quantum noise channels implemented via Kraus operators.

All channels are textbook implementations from standard quantum computing references.
No proprietary features or optimizations are included.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..backend.density_matrix import apply_kraus_single_qubit
from ..gates.standard import I, X, Y, Z
from .base import NoiseModel


def _apply_iid_single_qubit_channel(
    rho: torch.Tensor,
    n_qubits: int,
    kraus_ops: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Apply the same single-qubit Kraus channel to each qubit independently.

    This is a helper function for implementing iid noise models.

    Args:
        rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
        n_qubits: Number of qubits.
        kraus_ops: Tuple of 2x2 complex tensors representing single-qubit Kraus operators.

    Returns:
        A new density matrix tensor of the same shape as rho.
    """
    out = rho
    for q in range(n_qubits):
        out = apply_kraus_single_qubit(
            out, kraus_ops=kraus_ops, qubit=q, n_qubits=n_qubits
        )
    return out


class DepolarizingChannel(NoiseModel):
    """
    Standard single-qubit depolarizing channel applied iid to each qubit.

    The depolarizing channel is defined as:
        E(rho) = (1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z)

    Implemented via Kraus operators:
        E0 = sqrt(1 - p) I
        E1 = sqrt(p/3) X
        E2 = sqrt(p/3) Y
        E3 = sqrt(p/3) Z

    This is the standard textbook depolarizing channel with no modifications.

    Args:
        p: Depolarizing probability in [0, 1]. At p=1, the channel produces
            the completely mixed state.

    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, p: float) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("Depolarizing probability p must be in [0, 1].")
        self.p = float(p)

    def _kraus_ops(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the Kraus operators for this channel."""
        p = self.p

        i = I(dtype=dtype, device=device)
        x = X(dtype=dtype, device=device)
        y = Y(dtype=dtype, device=device)
        z = Z(dtype=dtype, device=device)

        e0 = torch.sqrt(torch.tensor(1.0 - p, dtype=dtype, device=device)) * i
        scale = torch.sqrt(torch.tensor(p / 3.0, dtype=dtype, device=device))
        e1 = scale * x
        e2 = scale * y
        e3 = scale * z

        return (e0, e1, e2, e3)

    def apply_density_matrix(self, rho: torch.Tensor, n_qubits: int) -> torch.Tensor:
        """
        Apply the depolarizing channel to the density matrix.

        Args:
            rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
            n_qubits: Number of qubits.

        Returns:
            A new density matrix tensor of the same shape as rho.
        """
        dim = rho.shape[-1]
        if dim != 2**n_qubits:
            raise ValueError(
                f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
                "in DepolarizingChannel."
            )

        dtype = rho.dtype
        device = rho.device

        kraus_ops = self._kraus_ops(dtype=dtype, device=device)
        return _apply_iid_single_qubit_channel(rho, n_qubits=n_qubits, kraus_ops=kraus_ops)


class AmplitudeDampingChannel(NoiseModel):
    """
    Standard single-qubit amplitude damping channel with parameter gamma in [0, 1].

    The amplitude damping channel models energy dissipation, where |1> decays to |0>.
    Kraus operators:
        E0 = [[1, 0],
              [0, sqrt(1 - gamma)]]
        E1 = [[0, sqrt(gamma)],
              [0, 0]]

    Applied iid to each qubit.

    This is the standard textbook amplitude damping channel.

    Args:
        gamma: Damping parameter in [0, 1]. At gamma=1, |1> fully decays to |0>.

    Raises:
        ValueError: If gamma is not in [0, 1].
    """

    def __init__(self, gamma: float) -> None:
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("Amplitude damping gamma must be in [0, 1].")
        self.gamma = float(gamma)

    def _kraus_ops(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the Kraus operators for this channel."""
        g = self.gamma

        e0 = torch.zeros((2, 2), dtype=dtype, device=device)
        e1 = torch.zeros((2, 2), dtype=dtype, device=device)

        e0[0, 0] = 1.0
        e0[1, 1] = torch.sqrt(torch.tensor(1.0 - g, dtype=dtype, device=device))

        e1[0, 1] = torch.sqrt(torch.tensor(g, dtype=dtype, device=device))

        return (e0, e1)

    def apply_density_matrix(self, rho: torch.Tensor, n_qubits: int) -> torch.Tensor:
        """
        Apply the amplitude damping channel to the density matrix.

        Args:
            rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
            n_qubits: Number of qubits.

        Returns:
            A new density matrix tensor of the same shape as rho.
        """
        dim = rho.shape[-1]
        if dim != 2**n_qubits:
            raise ValueError(
                f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
                "in AmplitudeDampingChannel."
            )

        dtype = rho.dtype
        device = rho.device

        kraus_ops = self._kraus_ops(dtype=dtype, device=device)
        return _apply_iid_single_qubit_channel(rho, n_qubits=n_qubits, kraus_ops=kraus_ops)


class PhaseDampingChannel(NoiseModel):
    """
    Standard single-qubit phase damping (dephasing) channel with parameter gamma in [0, 1].

    The phase damping channel models loss of phase coherence without energy loss.
    One common Kraus representation:
        E0 = [[1, 0],
              [0, sqrt(1 - gamma)]]
        E1 = [[0, 0],
              [0, sqrt(gamma)]]

    Applied iid to each qubit.

    This is the standard textbook phase damping channel.

    Args:
        gamma: Damping parameter in [0, 1]. At gamma=1, all off-diagonal elements vanish.

    Raises:
        ValueError: If gamma is not in [0, 1].
    """

    def __init__(self, gamma: float) -> None:
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("Phase damping gamma must be in [0, 1].")
        self.gamma = float(gamma)

    def _kraus_ops(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the Kraus operators for this channel."""
        g = self.gamma

        e0 = torch.zeros((2, 2), dtype=dtype, device=device)
        e1 = torch.zeros((2, 2), dtype=dtype, device=device)

        e0[0, 0] = 1.0
        e0[1, 1] = torch.sqrt(torch.tensor(1.0 - g, dtype=dtype, device=device))

        e1[1, 1] = torch.sqrt(torch.tensor(g, dtype=dtype, device=device))

        return (e0, e1)

    def apply_density_matrix(self, rho: torch.Tensor, n_qubits: int) -> torch.Tensor:
        """
        Apply the phase damping channel to the density matrix.

        Args:
            rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
            n_qubits: Number of qubits.

        Returns:
            A new density matrix tensor of the same shape as rho.
        """
        dim = rho.shape[-1]
        if dim != 2**n_qubits:
            raise ValueError(
                f"rho dimension {dim} does not match 2**n_qubits = {2**n_qubits} "
                "in PhaseDampingChannel."
            )

        dtype = rho.dtype
        device = rho.device

        kraus_ops = self._kraus_ops(dtype=dtype, device=device)
        return _apply_iid_single_qubit_channel(rho, n_qubits=n_qubits, kraus_ops=kraus_ops)

