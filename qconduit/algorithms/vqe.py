"""Variational Quantum Eigensolver (VQE) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..core.device import Device
from ..layers.ansatzes import ParametricAnsatz
from ..operators.pauli import PauliSum
from ..operators.expectation import expectation_pauli_sum

if TYPE_CHECKING:
    pass


def ensure_hamiltonian_diag(
    hamiltonian_diag: torch.Tensor, n_qubits: int, device: Device
) -> torch.Tensor:
    """
    Validate and prepare a diagonal Hamiltonian representation.

    Args:
        hamiltonian_diag: Diagonal Hamiltonian tensor of shape (2**n_qubits,).
            Must be real-valued (float32 or float64).
        n_qubits: Number of qubits. Must be >= 1.
        device: Device to move the tensor to.

    Returns:
        Cleaned tensor of shape (2**n_qubits,) with dtype torch.float32,
        moved to the specified device.

    Raises:
        ValueError: If hamiltonian_diag is not 1D or has incorrect length.
    """
    if hamiltonian_diag.ndim != 1:
        raise ValueError(
            f"hamiltonian_diag must be 1D, got shape {hamiltonian_diag.shape}"
        )

    expected_length = 2**n_qubits
    if hamiltonian_diag.shape[0] != expected_length:
        raise ValueError(
            f"hamiltonian_diag must have length 2**n_qubits = {expected_length}, "
            f"got {hamiltonian_diag.shape[0]}"
        )

    # Convert to float32 if needed (ensure it's a real floating-point type)
    if not torch.is_floating_point(hamiltonian_diag):
        hamiltonian_diag = hamiltonian_diag.to(dtype=torch.float32)
    else:
        hamiltonian_diag = hamiltonian_diag.to(dtype=torch.float32)

    # Move to device
    hamiltonian_diag = hamiltonian_diag.to(device=device.as_torch_device())

    return hamiltonian_diag


class VQE(nn.Module):
    """
    Variational Quantum Eigensolver for diagonal or Pauli-sum Hamiltonians.

    This class provides a minimal VQE implementation that works with a parametric
    ansatz and either a diagonal Hamiltonian (represented as a vector of eigenvalues
    over the computational basis) or a Pauli-sum Hamiltonian (represented as a sum
    of Pauli operators). The energy computation is fully differentiable via PyTorch
    autograd, enabling gradient-based optimization.

    For v0.1+, VQE accepts either a diagonal Hamiltonian or a Pauli-sum Hamiltonian
    via PauliSum, using standard Pauli expectation evaluation. This is still "dumb
    plumbing"; it does not introduce any novel algorithm.

    Args:
        ansatz: ParametricAnsatz instance that maps parameters to quantum states.
        hamiltonian: Either a diagonal Hamiltonian tensor of shape (2**n_qubits,)
            containing eigenvalues (must be real-valued), or a PauliSum object
            representing a Pauli-sum Hamiltonian.
        device: Device specification. If None, uses the ansatz's device.

    Attributes:
        ansatz: ParametricAnsatz instance.
        hamiltonian_diag: Diagonal Hamiltonian tensor, or None if using PauliSum.
        hamiltonian_pauli: PauliSum Hamiltonian, or None if using diagonal.
        device: Quantum device instance.

    Example:
        >>> from qconduit.layers.ansatzes import HardwareEfficientAnsatz
        >>> from qconduit.operators import PauliTerm, PauliSum
        >>> ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        >>> # Diagonal Hamiltonian
        >>> hamiltonian = torch.tensor([0.0, 1.0])  # |1⟩ has energy 1
        >>> vqe = VQE(ansatz, hamiltonian)
        >>> # Or Pauli-sum Hamiltonian
        >>> pauli_ham = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        >>> vqe = VQE(ansatz, pauli_ham)
        >>> params = torch.tensor([0.0])
        >>> energy = vqe.energy(params)
    """

    def __init__(
        self,
        ansatz: ParametricAnsatz,
        hamiltonian: torch.Tensor | PauliSum,
        device: Device | None = None,
    ) -> None:
        """Initialize a VQE instance."""
        super().__init__()

        if ansatz.n_qubits < 1:
            raise ValueError(f"ansatz.n_qubits must be >= 1, got {ansatz.n_qubits}")

        self.ansatz = ansatz

        # Determine device
        if device is None:
            self.device = ansatz.device
        else:
            self.device = device
            # Move ansatz to the specified device
            ansatz.to(device)

        n_qubits = ansatz.n_qubits

        # Distinguish between diagonal tensor and PauliSum
        if isinstance(hamiltonian, PauliSum):
            # Validate n_qubits match
            if hamiltonian.n_qubits() != n_qubits:
                raise ValueError(
                    f"hamiltonian.n_qubits() = {hamiltonian.n_qubits()} does not "
                    f"match ansatz.n_qubits = {n_qubits}"
                )
            self.hamiltonian_pauli = hamiltonian
            self.hamiltonian_diag = None
        else:
            # Treat as diagonal tensor
            self.hamiltonian_diag = ensure_hamiltonian_diag(
                hamiltonian, n_qubits, self.device
            )
            self.hamiltonian_pauli = None

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute energy (alias for energy()).

        This method allows VQE to be used as a standard PyTorch module.

        Args:
            params: Parameter tensor for the ansatz.

        Returns:
            Energy value(s) as a real tensor.
        """
        return self.energy(params)

    def energy(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation value of the Hamiltonian.

        The energy is computed as ⟨ψ(θ)|H|ψ(θ)⟩, where |ψ(θ)⟩ is the state
        produced by the ansatz with parameters θ, and H is either a diagonal
        Hamiltonian or a Pauli-sum Hamiltonian.

        For a diagonal Hamiltonian, this reduces to:
            E = Σ_i P(i) * H_ii
        where P(i) = |⟨i|ψ(θ)⟩|² is the probability of measuring basis state |i⟩.

        For a Pauli-sum Hamiltonian, this uses standard Pauli expectation evaluation
        via basis rotations and Z measurements.

        Args:
            params: Parameter tensor for the ansatz. Shape can be (num_parameters,)
                for unbatched or (batch_size, num_parameters) for batched inputs.

        Returns:
            Energy value(s) as a real tensor. Shape is () for unbatched or
            (batch_size,) for batched inputs.

        Raises:
            RuntimeError: If neither hamiltonian_diag nor hamiltonian_pauli is set
                (should not occur if constructor is used correctly).
        """
        # Build quantum state using the ansatz
        state = self.ansatz(params)  # shape: (2**n_qubits,) or (batch_size, 2**n_qubits)

        # Compute energy based on Hamiltonian type
        if self.hamiltonian_diag is not None:
            # Diagonal Hamiltonian path
            # Compute probabilities: |state|²
            probs = torch.abs(state) ** 2  # shape: same as state

            # Compute energy as expectation of diagonal Hamiltonian
            # probs: (..., 2**n_qubits)
            # hamiltonian_diag: (2**n_qubits,)
            # We need to broadcast and sum over the last dimension
            energy = (probs * self.hamiltonian_diag).sum(dim=-1)

            # Ensure result is real (should already be, but explicit for clarity)
            return energy.real
        elif self.hamiltonian_pauli is not None:
            # Pauli-sum Hamiltonian path
            energy = expectation_pauli_sum(state, self.hamiltonian_pauli)
            return energy
        else:
            raise RuntimeError(
                "Neither hamiltonian_diag nor hamiltonian_pauli is set. "
                "This should not occur if VQE is constructed correctly."
            )

