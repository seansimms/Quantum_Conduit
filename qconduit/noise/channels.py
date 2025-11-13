"""Standard quantum noise channels implemented via Kraus operators.

All channels are textbook implementations from standard quantum computing references.
No proprietary features or optimizations are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ..backend.density_matrix import apply_kraus_single_qubit
from ..gates.standard import I, X, Y, Z
from .base import NoiseModel


@dataclass(frozen=True)
class SingleQubitChannel:
    """
    Textbook single-qubit quantum channel represented via Kraus operators.

    The channel acts on a single qubit, with Kraus operators {K_i} satisfying:

        sum_i K_i^\\dagger K_i = I.

    This class is purely a container: it does not know *when* or *how*
    it is applied in a circuit. Application is handled by backend utilities.
    """

    name: str
    kraus_operators: Tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        """Validate that Kraus operators satisfy trace-preserving condition."""
        if len(self.kraus_operators) == 0:
            raise ValueError("kraus_operators must be non-empty")

        # Validate all operators are 2x2 complex tensors
        for i, K in enumerate(self.kraus_operators):
            if not isinstance(K, torch.Tensor):
                raise ValueError(f"Kraus operator {i} must be a torch.Tensor")
            if K.shape != (2, 2):
                raise ValueError(
                    f"Kraus operator {i} must have shape (2, 2), got {K.shape}"
                )
            if not torch.is_complex(K):
                raise ValueError(
                    f"Kraus operator {i} must be complex dtype, got {K.dtype}"
                )

        # Check trace-preserving condition: sum_i K_i^\dagger K_i = I
        # Get dtype and device from first operator
        dtype = self.kraus_operators[0].dtype
        device = self.kraus_operators[0].device
        I_ref = torch.eye(2, dtype=dtype, device=device)

        sum_kdag_k = torch.zeros((2, 2), dtype=dtype, device=device)
        for K in self.kraus_operators:
            # Ensure same dtype and device
            K = K.to(dtype=dtype, device=device)
            Kdag = K.conj().transpose(-2, -1)
            sum_kdag_k = sum_kdag_k + Kdag @ K

        # Check if sum_kdag_k ≈ I within tolerance
        diff = torch.abs(sum_kdag_k - I_ref)
        max_diff = torch.max(diff).item()
        if max_diff > 1e-6:
            raise ValueError(
                f"Kraus operators do not satisfy trace-preserving condition. "
                f"sum_i K_i^dagger K_i should equal I, but max difference is {max_diff:.2e}"
            )


def depolarizing_channel(p: float) -> SingleQubitChannel:
    """
    Construct a textbook single-qubit depolarizing channel.

    The depolarizing channel with parameter p is defined via Kraus operators:
        K_0 = sqrt(1 - p) I
        K_1 = sqrt(p/3) X
        K_2 = sqrt(p/3) Y
        K_3 = sqrt(p/3) Z

    At p=0, the channel is identity. At p=1, it fully depolarizes to the
    maximally mixed state.

    Parameters
    ----------
    p:
        Depolarizing probability in [0, 1].

    Returns
    -------
    SingleQubitChannel
        A channel with the specified depolarizing parameter.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Depolarizing parameter p must be in [0, 1], got {p}")

    # Use complex64 as default dtype
    dtype = torch.complex64
    device = torch.device("cpu")

    i = I(dtype=dtype, device=device)
    x = X(dtype=dtype, device=device)
    y = Y(dtype=dtype, device=device)
    z = Z(dtype=dtype, device=device)

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float32)).to(
        dtype=dtype
    )
    sqrt_p_over_3 = torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float32)).to(
        dtype=dtype
    )

    k0 = sqrt_one_minus_p * i
    k1 = sqrt_p_over_3 * x
    k2 = sqrt_p_over_3 * y
    k3 = sqrt_p_over_3 * z

    name = f"depolarizing(p={p})"
    return SingleQubitChannel(name=name, kraus_operators=(k0, k1, k2, k3))


def phase_damping_channel(p: float) -> SingleQubitChannel:
    """
    Construct a textbook single-qubit phase damping (dephasing) channel.

    The phase damping channel with parameter p is defined via Kraus operators:
        K_0 = sqrt(1 - p) I
        K_1 = sqrt(p) |0><0|
        K_2 = sqrt(p) |1><1|

    This channel models loss of phase coherence without energy loss.

    Parameters
    ----------
    p:
        Phase damping parameter in [0, 1]. At p=1, all off-diagonal elements vanish.

    Returns
    -------
    SingleQubitChannel
        A channel with the specified phase damping parameter.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Phase damping parameter p must be in [0, 1], got {p}")

    dtype = torch.complex64
    device = torch.device("cpu")

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float32)).to(
        dtype=dtype
    )
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float32)).to(dtype=dtype)

    # K_0 = sqrt(1 - p) I
    k0 = sqrt_one_minus_p * I(dtype=dtype, device=device)

    # K_1 = sqrt(p) |0><0| = sqrt(p) [[1, 0], [0, 0]]
    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[0, 0] = sqrt_p

    # K_2 = sqrt(p) |1><1| = sqrt(p) [[0, 0], [0, 1]]
    k2 = torch.zeros((2, 2), dtype=dtype, device=device)
    k2[1, 1] = sqrt_p

    name = f"phase_damping(p={p})"
    return SingleQubitChannel(name=name, kraus_operators=(k0, k1, k2))


def amplitude_damping_channel(gamma: float) -> SingleQubitChannel:
    """
    Construct a textbook single-qubit amplitude damping channel.

    The amplitude damping channel with parameter gamma models energy dissipation,
    where |1> decays to |0>. Kraus operators:
        K_0 = [[1, 0],
               [0, sqrt(1 - gamma)]]
        K_1 = [[0, sqrt(gamma)],
               [0, 0]]

    Parameters
    ----------
    gamma:
        Damping parameter in [0, 1]. At gamma=1, |1> fully decays to |0>.

    Returns
    -------
    SingleQubitChannel
        A channel with the specified amplitude damping parameter.

    Raises
    ------
    ValueError
        If gamma is not in [0, 1].
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(
            f"Amplitude damping parameter gamma must be in [0, 1], got {gamma}"
        )

    dtype = torch.complex64
    device = torch.device("cpu")

    sqrt_one_minus_gamma = torch.sqrt(
        torch.tensor(1.0 - gamma, dtype=torch.float32)
    ).to(dtype=dtype)
    sqrt_gamma = torch.sqrt(torch.tensor(gamma, dtype=torch.float32)).to(dtype=dtype)

    # K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = 1.0
    k0[1, 1] = sqrt_one_minus_gamma

    # K_1 = [[0, sqrt(gamma)], [0, 0]]
    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[0, 1] = sqrt_gamma

    name = f"amplitude_damping(gamma={gamma})"
    return SingleQubitChannel(name=name, kraus_operators=(k0, k1))


def identity_channel() -> SingleQubitChannel:
    """
    Construct a trivial single-qubit channel that leaves the state unchanged.

    This is useful as a default when no noise is desired.

    Returns
    -------
    SingleQubitChannel
        The identity channel (K_0 = I).
    """
    dtype = torch.complex64
    device = torch.device("cpu")
    k0 = I(dtype=dtype, device=device)
    return SingleQubitChannel(name="identity", kraus_operators=(k0,))


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


__all__ = [
    "SingleQubitChannel",
    "depolarizing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "identity_channel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
]
