"""Kraus operator channels for quantum noise models.

This module provides a textbook implementation of quantum channels via
Kraus operators. All channels are standard, non-proprietary implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from qconduit.core.device import default_device
from qconduit.gates.standard import I, X, Y, Z


@dataclass(frozen=True)
class KrausChannel:
    """
    Trace-preserving quantum channel represented in Kraus form:

        E(ρ) = ∑_i K_i ρ K_i†,

    where the Kraus operators K_i act on a Hilbert space of dimension
    d = 2**num_qubits.

    This class assumes a fixed number of qubits and enforces basic
    trace-preserving checks at construction time.
    """

    name: str
    kraus_ops: tuple[torch.Tensor, ...]
    num_qubits: int

    def __post_init__(self) -> None:
        """Validate KrausChannel invariants."""
        # Validate num_qubits
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")

        # Validate kraus_ops is non-empty
        if len(self.kraus_ops) == 0:
            raise ValueError("kraus_ops must contain at least one operator.")

        # Determine dimension
        dim = 1 << self.num_qubits

        # Validate and normalize each Kraus operator
        normalized_ops = []
        dtype = None
        device = None

        for i, K in enumerate(self.kraus_ops):
            # Ensure it is a torch.Tensor
            if not isinstance(K, torch.Tensor):
                raise ValueError(
                    f"Kraus operator {i} must be a torch.Tensor, got {type(K)}"
                )

            # Ensure 2D and correct shape
            if K.dim() != 2:
                raise ValueError(
                    f"Kraus operator {i} must be 2D, got {K.dim()} dimensions"
                )
            if K.shape != (dim, dim):
                raise ValueError(
                    f"Kraus operator {i} must have shape ({dim}, {dim}), got {K.shape}"
                )

            # Enforce complex dtype - convert to complex128 if not already complex
            if not torch.is_complex(K):
                K_complex = K.to(dtype=torch.complex128)
            else:
                K_complex = K
            normalized_ops.append(K_complex)

            # Track dtype and device from first operator
            if dtype is None:
                dtype = K_complex.dtype
                device = K_complex.device

        # Ensure all operators are on same device and dtype
        normalized_ops = [
            K.to(dtype=dtype, device=device) for K in normalized_ops
        ]

        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(self, "kraus_ops", tuple(normalized_ops))

        # Compute trace-preserving check: ∑ K_i† K_i = I
        identity = torch.eye(dim, dtype=dtype, device=device)
        sum_kdag_k = torch.zeros((dim, dim), dtype=dtype, device=device)

        for K in self.kraus_ops:
            Kdag = K.conj().T
            sum_kdag_k = sum_kdag_k + Kdag @ K

        # Check if sum_kdag_k ≈ I
        diff = torch.abs(sum_kdag_k - identity)
        max_diff = torch.max(diff).item()
        if max_diff > 1e-7:
            raise ValueError(
                f"Kraus operators do not define a trace-preserving channel "
                f"(∑K†K ≠ I). Max difference: {max_diff:.2e}"
            )

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "KrausChannel":
        """
        Return a new KrausChannel with Kraus operators moved to the given
        device and/or dtype.

        Parameters
        ----------
        device:
            Optional device for the output. If None, keeps existing device.
        dtype:
            Optional dtype for the output. If None, keeps existing dtype.

        Returns
        -------
        KrausChannel
            A new channel with transformed Kraus operators.
        """
        # Determine target device and dtype
        target_device = device if device is not None else self.kraus_ops[0].device
        target_dtype = dtype if dtype is not None else self.kraus_ops[0].dtype

        # Transform all Kraus operators
        # Use .to() with both device and dtype to ensure conversion happens
        new_ops = tuple(K.to(device=target_device, dtype=target_dtype) for K in self.kraus_ops)

        # Create new channel (validation will run in __post_init__)
        return KrausChannel(
            name=self.name, kraus_ops=new_ops, num_qubits=self.num_qubits
        )

    @property
    def kraus_operators(self) -> tuple[torch.Tensor, ...]:
        """
        Backward compatibility property: alias for kraus_ops.

        Returns
        -------
        tuple[torch.Tensor, ...]
            The Kraus operators.
        """
        return self.kraus_ops

    def is_trace_preserving(self, atol: float = 1e-7) -> bool:
        """
        Check whether the channel is trace-preserving, i.e. ∑ K_i† K_i ≈ I.

        Parameters
        ----------
        atol:
            Absolute tolerance for the check.

        Returns
        -------
        bool
            True if the channel is trace-preserving within tolerance.
        """
        dim = 1 << self.num_qubits
        dtype = self.kraus_ops[0].dtype
        device = self.kraus_ops[0].device

        identity = torch.eye(dim, dtype=dtype, device=device)
        sum_kdag_k = torch.zeros((dim, dim), dtype=dtype, device=device)

        for K in self.kraus_ops:
            Kdag = K.conj().T
            sum_kdag_k = sum_kdag_k + Kdag @ K

        diff = torch.abs(sum_kdag_k - identity)
        max_diff = torch.max(diff).item()
        return max_diff <= atol


def bit_flip_channel(p: float) -> KrausChannel:
    """
    Single-qubit bit-flip channel with flip probability p:

        E(ρ) = (1 - p) ρ + p X ρ X,

    implemented with Kraus operators:

        K0 = sqrt(1 - p) I,
        K1 = sqrt(p) X,

    for 0 <= p <= 1.

    Parameters
    ----------
    p:
        Bit-flip probability in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit bit-flip channel.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Bit-flip probability p must be in [0, 1], got {p}")

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

    return KrausChannel(
        name=f"bit_flip(p={p})", kraus_ops=(k0, k1), num_qubits=1
    )


def phase_flip_channel(p: float) -> KrausChannel:
    """
    Single-qubit phase-flip (Z) channel with probability p:

        E(ρ) = (1 - p) ρ + p Z ρ Z.

    Kraus operators:

        K0 = sqrt(1 - p) I,
        K1 = sqrt(p) Z.

    Parameters
    ----------
    p:
        Phase-flip probability in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit phase-flip channel.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Phase-flip probability p must be in [0, 1], got {p}")

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

    return KrausChannel(
        name=f"phase_flip(p={p})", kraus_ops=(k0, k1), num_qubits=1
    )


def bit_phase_flip_channel(p: float) -> KrausChannel:
    """
    Single-qubit bit-phase-flip (Y) channel with probability p:

        E(ρ) = (1 - p) ρ + p Y ρ Y.

    Kraus operators:

        K0 = sqrt(1 - p) I,
        K1 = sqrt(p) Y.

    Parameters
    ----------
    p:
        Bit-phase-flip probability in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit bit-phase-flip channel.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Bit-phase-flip probability p must be in [0, 1], got {p}")

    device = default_device().as_torch_device()
    dtype = torch.complex128

    i_mat = I(dtype=dtype, device=device)
    y_mat = Y(dtype=dtype, device=device)

    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float64)).to(dtype=dtype)

    k0 = sqrt_one_minus_p * i_mat
    k1 = sqrt_p * y_mat

    return KrausChannel(
        name=f"bit_phase_flip(p={p})", kraus_ops=(k0, k1), num_qubits=1
    )


def depolarizing_channel(p: float) -> KrausChannel:
    """
    Single-qubit depolarizing channel with depolarizing probability p:

        E(ρ) = (1 - p) ρ + (p / 3) (X ρ X + Y ρ Y + Z ρ Z),

    with Kraus operators:

        K0 = sqrt(1 - p) I,
        K1 = sqrt(p / 3) X,
        K2 = sqrt(p / 3) Y,
        K3 = sqrt(p / 3) Z,

    for 0 <= p <= 1.

    Parameters
    ----------
    p:
        Depolarizing probability in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit depolarizing channel.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Depolarizing probability p must be in [0, 1], got {p}")

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

    return KrausChannel(
        name=f"depolarizing(p={p})", kraus_ops=(k0, k1, k2, k3), num_qubits=1
    )


def phase_damping_channel(gamma: Optional[float] = None, p: Optional[float] = None) -> KrausChannel:
    """
    Single-qubit phase-damping (dephasing) channel with parameter gamma:

        K0 = [[1, 0],
              [0, sqrt(1 - gamma)]],
        K1 = [[0, 0],
              [0, sqrt(gamma)]],

    where 0 <= gamma <= 1.

    This damps the off-diagonal coherence without changing populations.

    Parameters
    ----------
    gamma:
        Phase damping parameter in [0, 1]. If None, uses p instead.
    p:
        Phase damping parameter in [0, 1] (backward compatibility alias for gamma).
        If gamma is provided, p is ignored.

    Returns
    -------
    KrausChannel
        A single-qubit phase-damping channel.

    Raises
    ------
    ValueError
        If gamma or p is not in [0, 1], or if neither is provided.
    """
    # Backward compatibility: accept 'p' parameter
    if gamma is None:
        if p is None:
            raise ValueError("Either gamma or p must be provided")
        gamma = p
    
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"Phase damping parameter gamma must be in [0, 1], got {gamma}")

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
    k1[1, 1] = sqrt_gamma

    return KrausChannel(
        name=f"phase_damping(gamma={gamma})", kraus_ops=(k0, k1), num_qubits=1
    )


def amplitude_damping_channel(gamma: float) -> KrausChannel:
    """
    Single-qubit amplitude-damping channel with relaxation probability gamma:

        K0 = [[1, 0],
              [0, sqrt(1 - gamma)]],
        K1 = [[0, sqrt(gamma)],
              [0, 0]],

    where 0 <= gamma <= 1.

    This models decay |1> -> |0> with probability gamma.

    Parameters
    ----------
    gamma:
        Amplitude damping parameter in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit amplitude-damping channel.

    Raises
    ------
    ValueError
        If gamma is not in [0, 1].
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(
            f"Amplitude damping parameter gamma must be in [0, 1], got {gamma}"
        )

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

    return KrausChannel(
        name=f"amplitude_damping(gamma={gamma})", kraus_ops=(k0, k1), num_qubits=1
    )


def generalized_amplitude_damping_channel(
    gamma: float,
    p_excited: float,
) -> KrausChannel:
    """
    Single-qubit generalized amplitude-damping channel with parameters:

        gamma: relaxation probability (0 <= gamma <= 1),
        p_excited: equilibrium excited-state population (0 <= p_excited <= 1).

    Kraus operators:

        K0 = sqrt(p) * [[1, 0],
                        [0, sqrt(1 - gamma)]]
        K1 = sqrt(p) * [[0, sqrt(gamma)],
                        [0, 0]]
        K2 = sqrt(1 - p) * [[sqrt(1 - gamma), 0],
                            [0, 1]]
        K3 = sqrt(1 - p) * [[0, 0],
                            [sqrt(gamma), 0]]

    with p = p_excited.

    Parameters
    ----------
    gamma:
        Relaxation probability in [0, 1].
    p_excited:
        Equilibrium excited-state population in [0, 1].

    Returns
    -------
    KrausChannel
        A single-qubit generalized amplitude-damping channel.

    Raises
    ------
    ValueError
        If gamma or p_excited is not in [0, 1].
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if p_excited < 0.0 or p_excited > 1.0:
        raise ValueError(f"p_excited must be in [0, 1], got {p_excited}")

    device = default_device().as_torch_device()
    dtype = torch.complex128

    p = p_excited
    sqrt_p = torch.sqrt(torch.tensor(p, dtype=torch.float64)).to(dtype=dtype)
    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_one_minus_gamma = torch.sqrt(
        torch.tensor(1.0 - gamma, dtype=torch.float64)
    ).to(dtype=dtype)
    sqrt_gamma = torch.sqrt(torch.tensor(gamma, dtype=torch.float64)).to(dtype=dtype)

    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = sqrt_p
    k0[1, 1] = sqrt_p * sqrt_one_minus_gamma

    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[0, 1] = sqrt_p * sqrt_gamma

    k2 = torch.zeros((2, 2), dtype=dtype, device=device)
    k2[0, 0] = sqrt_one_minus_p * sqrt_one_minus_gamma
    k2[1, 1] = sqrt_one_minus_p

    k3 = torch.zeros((2, 2), dtype=dtype, device=device)
    k3[1, 0] = sqrt_one_minus_p * sqrt_gamma

    return KrausChannel(
        name=f"generalized_amplitude_damping(gamma={gamma}, p_excited={p_excited})",
        kraus_ops=(k0, k1, k2, k3),
        num_qubits=1,
    )


def two_qubit_depolarizing_channel(p: float) -> KrausChannel:
    """
    Two-qubit depolarizing channel with depolarizing probability p:

        E(ρ) = (1 - p) ρ + (p / 15) ∑_{P != I⊗I} P ρ P,

    where P runs over the 16 tensor products {I, X, Y, Z} ⊗ {I, X, Y, Z},
    excluding I⊗I.

    Kraus operators:

        K0 = sqrt(1 - p) I⊗I,
        K_i = sqrt(p / 15) P_i,  for the 15 non-identity Paulis.

    This is defined only for 2 qubits.

    Parameters
    ----------
    p:
        Depolarizing probability in [0, 1].

    Returns
    -------
    KrausChannel
        A two-qubit depolarizing channel.

    Raises
    ------
    ValueError
        If p is not in [0, 1].
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Depolarizing probability p must be in [0, 1], got {p}")

    device = default_device().as_torch_device()
    dtype = torch.complex128

    # Build single-qubit Pauli matrices
    i_single = I(dtype=dtype, device=device)
    x_single = X(dtype=dtype, device=device)
    y_single = Y(dtype=dtype, device=device)
    z_single = Z(dtype=dtype, device=device)

    paulis = [i_single, x_single, y_single, z_single]

    # Build all 16 tensor products
    kraus_ops = []
    sqrt_one_minus_p = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(
        dtype=dtype
    )
    sqrt_p_over_15 = torch.sqrt(torch.tensor(p / 15.0, dtype=torch.float64)).to(
        dtype=dtype
    )

    for p1 in paulis:
        for p2 in paulis:
            # Compute tensor product: p1 ⊗ p2
            # Use torch.kron for Kronecker product
            pauli_product = torch.kron(p1, p2)

            if p1 is i_single and p2 is i_single:
                # Identity term
                k0 = sqrt_one_minus_p * pauli_product
                kraus_ops.append(k0)
            else:
                # Non-identity Pauli term
                ki = sqrt_p_over_15 * pauli_product
                kraus_ops.append(ki)

    return KrausChannel(
        name=f"two_qubit_depolarizing(p={p})", kraus_ops=tuple(kraus_ops), num_qubits=2
    )


__all__ = [
    "KrausChannel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "depolarizing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "generalized_amplitude_damping_channel",
    "two_qubit_depolarizing_channel",
]

