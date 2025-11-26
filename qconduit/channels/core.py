"""Core quantum channel implementation via Kraus operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from qconduit.core.device import default_device


@dataclass(frozen=True)
class KrausChannel:
    """
    A quantum channel represented by a list of Kraus operators {K_i}.

    The channel acts on density matrices as:
        E(ρ) = ∑_k K_k ρ K_k^†

    Attributes
    ----------
    kraus_ops: Tuple[torch.Tensor, ...]
        Sequence of 2D complex tensors, each shape (2**n_target_qubits, 2**n_target_qubits).
    n_qubits: int
        Number of qubits the channel acts on (dimension = 2**n_qubits).

    Notes
    -----
    Qubit ordering convention: follows qconduit.backend.statevector convention,
    where qubit 0 is the least significant bit (LSB) in the computational basis index.

    Limitations
    -----------
    - **Multi-qubit channel extension**: The `tensor_extend()` method currently only
      supports single-qubit channels (n_qubits=1). For channels acting on multiple
      qubits, you must construct the full-system Kraus operators manually or use
      the channel directly on a density matrix of the appropriate size.
    """

    kraus_ops: Tuple[torch.Tensor, ...]
    n_qubits: int

    def __post_init__(self) -> None:
        """Validate KrausChannel invariants."""
        # Validate n_qubits
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")

        # Validate kraus_ops is non-empty
        if len(self.kraus_ops) == 0:
            raise ValueError("kraus_ops must contain at least one operator")

        dim = 1 << self.n_qubits  # 2**n_qubits

        # Validate and normalize each Kraus operator
        normalized_ops = []
        dtype = None
        device = None

        for i, kraus_op in enumerate(self.kraus_ops):
            # Ensure it is a torch.Tensor
            if not isinstance(kraus_op, torch.Tensor):
                raise ValueError(
                    f"Kraus operator {i} must be a torch.Tensor, got {type(kraus_op)}"
                )

            # Ensure 2D and square
            if kraus_op.dim() != 2:
                raise ValueError(
                    f"Kraus operator {i} must be 2D, got {kraus_op.dim()} dimensions"
                )
            if kraus_op.shape[0] != kraus_op.shape[1]:
                raise ValueError(
                    f"Kraus operator {i} must be square, got shape {kraus_op.shape}"
                )
            if kraus_op.shape[0] != dim:
                raise ValueError(
                    f"Kraus operator {i} must have shape ({dim}, {dim}), got {kraus_op.shape}"
                )

            # Enforce complex128 dtype and default device
            target_device = default_device().as_torch_device()
            if not torch.is_complex(kraus_op):
                kraus_complex = kraus_op.to(dtype=torch.complex128, device=target_device)
            else:
                kraus_complex = kraus_op.to(dtype=torch.complex128, device=target_device)
            normalized_ops.append(kraus_complex)

            # Track dtype and device from first operator
            if dtype is None:
                dtype = kraus_complex.dtype
                device = kraus_complex.device

        # Ensure all operators are on same device and dtype
        normalized_ops = [
            op.to(dtype=dtype, device=device) for op in normalized_ops
        ]

        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(self, "kraus_ops", tuple(normalized_ops))

        # Check CPTP property: ∑_k K_k^† K_k = I
        identity = torch.eye(dim, dtype=dtype, device=device)
        sum_kdag_k = torch.zeros((dim, dim), dtype=dtype, device=device)

        for kraus_op in self.kraus_ops:
            kraus_dag = kraus_op.conj().T
            sum_kdag_k = sum_kdag_k + kraus_dag @ kraus_op

        # Check if sum_kdag_k ≈ I within tolerance
        # Use relative tolerance for better numerical stability with accumulated errors
        diff = torch.abs(sum_kdag_k - identity)
        max_diff = torch.max(diff).item()
        # Use more lenient tolerance: 1e-8 absolute or 1e-6 relative
        # This accounts for floating-point accumulation errors in channel construction
        rel_tol = 1e-6
        abs_tol = 1e-8
        max_identity = torch.max(torch.abs(identity)).item()
        if max_diff > abs_tol and max_diff > rel_tol * max_identity:
            raise ValueError(
                f"Kraus operators do not define a CPTP channel "
                f"(∑K^†K ≠ I). Max difference: {max_diff:.2e} "
                f"(tolerance: abs={abs_tol:.1e}, rel={rel_tol:.1e})"
            )

    def apply_to_density(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Apply the channel to a density matrix: E(ρ) = ∑_k K_k ρ K_k^†.

        Parameters
        ----------
        rho: torch.Tensor
            Density matrix of shape (dim, dim) with dim = 2**n_qubits, complex dtype.

        Returns
        -------
        torch.Tensor
            Output density matrix of same shape and dtype as rho.

        Raises
        ------
        ValueError
            If rho shape or dtype is invalid.
        """
        if rho.dim() != 2:
            raise ValueError(f"rho must be 2D, got {rho.dim()} dimensions")
        if rho.shape[0] != rho.shape[1]:
            raise ValueError(f"rho must be square, got shape {rho.shape}")
        dim = 1 << self.n_qubits
        if rho.shape[0] != dim:
            raise ValueError(
                f"rho dimension {rho.shape[0]} does not match channel dimension {dim}"
            )
        if not torch.is_complex(rho):
            raise ValueError(f"rho must be complex dtype, got {rho.dtype}")

        # Ensure rho is on same device as Kraus operators
        device = self.kraus_ops[0].device
        dtype = self.kraus_ops[0].dtype
        rho = rho.to(dtype=dtype, device=device)

        # Apply channel: E(ρ) = ∑_k K_k ρ K_k^†
        new_rho = torch.zeros_like(rho)
        for kraus_op in self.kraus_ops:
            kraus_rho = kraus_op @ rho
            kraus_rho_kdag = kraus_rho @ kraus_op.conj().T
            new_rho = new_rho + kraus_rho_kdag

        return new_rho

    def apply_to_statevector(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply the channel to a pure statevector, returning a density matrix.

        This converts the pure state |ψ⟩ to density matrix |ψ⟩⟨ψ|, applies the channel,
        and returns the resulting density matrix.

        Parameters
        ----------
        psi: torch.Tensor
            Statevector of shape (dim,) with dim = 2**n_qubits, complex dtype.

        Returns
        -------
        torch.Tensor
            Output density matrix of shape (dim, dim), complex dtype.

        Raises
        ------
        ValueError
            If psi shape or dtype is invalid.
        """
        if psi.dim() != 1:
            raise ValueError(f"psi must be 1D, got {psi.dim()} dimensions")
        dim = 1 << self.n_qubits
        if psi.shape[0] != dim:
            raise ValueError(
                f"psi dimension {psi.shape[0]} does not match channel dimension {dim}"
            )
        if not torch.is_complex(psi):
            raise ValueError(f"psi must be complex dtype, got {psi.dtype}")

        # Convert to density matrix: ρ = |ψ⟩⟨ψ|
        rho = psi.unsqueeze(1) @ psi.conj().unsqueeze(0)

        # Apply channel
        return self.apply_to_density(rho)

    def tensor_extend(
        self, total_n_qubits: int, target_qubits: Sequence[int]
    ) -> KrausChannel:
        """
        Extend the channel to act on a larger system.

        Constructs a channel acting on total_n_qubits by embedding Kraus ops
        with identity on other qubits.

        Parameters
        ----------
        total_n_qubits: int
            Total number of qubits in the extended system.
        target_qubits: Sequence[int]
            Indices of qubits the channel acts on (must have length n_qubits).
            For single-qubit channels (n_qubits=1), this should be a single qubit index.

        Returns
        -------
        KrausChannel
            New channel acting on total_n_qubits qubits.

        Raises
        ------
        ValueError
            If target_qubits length doesn't match n_qubits, or indices are invalid.
        NotImplementedError
            If n_qubits > 1 (multi-qubit channel extension not yet implemented).
            See class docstring for limitations and workarounds.

        Notes
        -----
        **Current Limitation**: This method only supports single-qubit channels
        (n_qubits=1). For multi-qubit channels, construct the full-system Kraus
        operators manually using tensor products.
        """
        if len(target_qubits) != self.n_qubits:
            raise ValueError(
                f"target_qubits length {len(target_qubits)} must equal "
                f"channel n_qubits {self.n_qubits}"
            )
        if total_n_qubits < self.n_qubits:
            raise ValueError(
                f"total_n_qubits {total_n_qubits} must be >= n_qubits {self.n_qubits}"
            )
        if any(q < 0 or q >= total_n_qubits for q in target_qubits):
            raise ValueError(
                f"All target_qubits must be in [0, {total_n_qubits}), "
                f"got {target_qubits}"
            )

        device = self.kraus_ops[0].device
        dtype = self.kraus_ops[0].dtype
        identity_single = torch.eye(2, dtype=dtype, device=device)

        extended_kraus_ops = []
        for kraus_op in self.kraus_ops:
            # Build tensor product: P_{n-1} ⊗ P_{n-2} ⊗ ... ⊗ P_0
            # where P_i = K if i in target_qubits, else I
            # Convention: qubit 0 is LSB, so we build from MSB (n-1) down to LSB (0)
            # This matches the convention used in qconduit.backend.density_matrix
            # For single-qubit channels: target_qubits has one element
            if self.n_qubits == 1:
                target_qubit = target_qubits[0]
                # Build tensor_op = I ⊗ ... ⊗ K ⊗ ... ⊗ I
                # Start with the operator for the highest qubit
                tensor_op = (
                    kraus_op if (total_n_qubits - 1) == target_qubit else identity_single
                )
                for q in range(total_n_qubits - 2, -1, -1):
                    operand = kraus_op if q == target_qubit else identity_single
                    tensor_op = torch.kron(tensor_op, operand)
            else:
                # Multi-qubit extension: need to map target_qubits to positions in K
                # This is more complex and not needed for built-in channels
                #
                # LIMITATION: Multi-qubit channel extension (n_qubits > 1) is not
                # currently implemented. This means you cannot use tensor_extend()
                # with channels that act on more than one qubit simultaneously.
                #
                # Workaround: For multi-qubit channels, construct the full-system
                # Kraus operators manually using tensor products, or use the channel
                # directly on a density matrix of the appropriate size.
                raise NotImplementedError(
                    "Multi-qubit tensor extension for n_qubits > 1 is not yet implemented. "
                    "This limitation affects channels that act on multiple qubits simultaneously. "
                    "For single-qubit channels, tensor_extend() works correctly. "
                    "See the module documentation for workarounds."
                )

            extended_kraus_ops.append(tensor_op)

        return KrausChannel(
            kraus_ops=tuple(extended_kraus_ops), n_qubits=total_n_qubits
        )

    def compose(self, other: KrausChannel) -> KrausChannel:
        """
        Compose two channels: E₂ ∘ E₁.

        The composition E₂ ∘ E₁ has Kraus operators {K^{(2)}_a K^{(1)}_b}
        for all pairs (a, b).

        Parameters
        ----------
        other: KrausChannel
            The channel to compose with (applied second).

        Returns
        -------
        KrausChannel
            Composed channel E₂ ∘ E₁.

        Raises
        ------
        ValueError
            If n_qubits don't match.
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Cannot compose channels with different n_qubits: "
                f"{self.n_qubits} vs {other.n_qubits}"
            )

        composed_ops = []
        for op_second in other.kraus_ops:
            for op_first in self.kraus_ops:
                composed_ops.append(op_second @ op_first)

        return KrausChannel(kraus_ops=tuple(composed_ops), n_qubits=self.n_qubits)

    def as_superoperator(self) -> torch.Tensor:
        """
        Return the superoperator matrix representation of the channel.

        The superoperator S satisfies vec(ρ_out) = S @ vec(ρ_in), where vec
        is the vectorization operator (column-stacking).

        Returns
        -------
        torch.Tensor
            Superoperator matrix of shape (dim*dim, dim*dim), complex dtype.
        """
        dim = 1 << self.n_qubits
        device = self.kraus_ops[0].device
        dtype = self.kraus_ops[0].dtype

        # Build superoperator: S = ∑_k K_k ⊗ K_k^*
        superop = torch.zeros((dim * dim, dim * dim), dtype=dtype, device=device)
        for kraus_op in self.kraus_ops:
            kraus_conj = kraus_op.conj()
            superop = superop + torch.kron(kraus_op, kraus_conj)

        return superop

    def is_cptp(self, atol: float = 1e-8, rtol: float = 1e-6) -> bool:
        """
        Check if the channel is completely positive and trace-preserving (CPTP).

        Checks that ∑_k K_k^† K_k ≈ I within tolerance. Uses both absolute and
        relative tolerance for better numerical stability.

        Parameters
        ----------
        atol: float
            Absolute tolerance for the check. Default is 1e-8 (more lenient than
            the constructor's validation to account for accumulated errors).
        rtol: float
            Relative tolerance for the check. Default is 1e-6.

        Returns
        -------
        bool
            True if channel is CPTP within tolerance.
        """
        dim = 1 << self.n_qubits
        dtype = self.kraus_ops[0].dtype
        device = self.kraus_ops[0].device

        identity = torch.eye(dim, dtype=dtype, device=device)
        sum_kdag_k = torch.zeros((dim, dim), dtype=dtype, device=device)

        for kraus_op in self.kraus_ops:
            kraus_dag = kraus_op.conj().T
            sum_kdag_k = sum_kdag_k + kraus_dag @ kraus_op

        diff = torch.abs(sum_kdag_k - identity)
        max_diff = torch.max(diff).item()
        max_identity = torch.max(torch.abs(identity)).item()
        # Use both absolute and relative tolerance
        return max_diff <= atol or max_diff <= rtol * max_identity

    def sample_state_after_channel(
        self, psi: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample a post-measurement statevector after applying the channel.

        Uses stochastic Kraus sampling: compute probabilities p_k = ⟨ψ|K_k^† K_k|ψ⟩,
        sample index j according to these probabilities, and return normalized
        statevector K_j |ψ⟩ / ||K_j |ψ⟩||.

        Parameters
        ----------
        psi: torch.Tensor
            Input statevector of shape (dim,), complex dtype.
        generator: Optional[torch.Generator]
            Random number generator for deterministic sampling. If None, creates
            one with seed 0.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Tuple of (normalized output statevector, sampled Kraus operator index).

        Raises
        ------
        ValueError
            If psi shape/dtype is invalid, or if sampled Kraus operator produces
            zero-norm statevector.
        """
        if psi.dim() != 1:
            raise ValueError(f"psi must be 1D, got {psi.dim()} dimensions")
        dim = 1 << self.n_qubits
        if psi.shape[0] != dim:
            raise ValueError(
                f"psi dimension {psi.shape[0]} does not match channel dimension {dim}"
            )
        if not torch.is_complex(psi):
            raise ValueError(f"psi must be complex dtype, got {psi.dtype}")

        device = self.kraus_ops[0].device
        dtype = self.kraus_ops[0].dtype
        psi = psi.to(dtype=dtype, device=device)

        # Compute probabilities: p_k = ⟨ψ|K_k^† K_k|ψ⟩
        probs = []
        for kraus_op in self.kraus_ops:
            op_dag_op = kraus_op.conj().T @ kraus_op
            prob_k = (psi.conj() @ op_dag_op @ psi).real
            if prob_k < 0:
                prob_k = 0.0  # Numerical errors can produce tiny negatives
            probs.append(prob_k)

        probs_tensor = torch.tensor(probs, dtype=torch.float64, device=device)
        probs_tensor = torch.clamp(probs_tensor, min=0.0)
        # Normalize probabilities
        prob_sum = probs_tensor.sum()
        if prob_sum < 1e-12:
            raise ValueError("All Kraus operator probabilities are zero")
        probs_tensor = probs_tensor / prob_sum

        # Sample index
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(0)

        j = torch.multinomial(probs_tensor, num_samples=1, generator=generator).item()

        # Compute output state for the sampled Kraus operator
        sampled_op = self.kraus_ops[j]
        psi_out = sampled_op @ psi

        # Normalize
        norm = torch.norm(psi_out)
        if norm < 1e-12:
            raise ValueError(
                f"Sampled Kraus operator {j} produces zero-norm statevector"
            )
        psi_out = psi_out / norm

        return psi_out, j


def _kron(*mats: torch.Tensor) -> torch.Tensor:
    """Compute Kronecker product of multiple matrices."""
    result = mats[0]
    for mat in mats[1:]:
        result = torch.kron(result, mat)
    return result


def _eye(n_qubits: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return identity matrix of size 2**n_qubits."""
    dim = 1 << n_qubits
    return torch.eye(dim, dtype=dtype, device=device)


__all__ = ["KrausChannel", "_kron", "_eye"]

