"""Base class for quantum noise models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class NoiseModel(ABC):
    """
    Base class for generic quantum noise models.

    Implementations should be textbook Kraus operator channels and
    must not contain any proprietary tricks.
    """

    @abstractmethod
    def apply_density_matrix(
        self,
        rho: torch.Tensor,
        n_qubits: int,
    ) -> torch.Tensor:
        """
        Apply this noise model to the given density matrix rho.

        Args:
            rho: Density matrix tensor of shape (..., dim, dim) with dim = 2**n_qubits.
            n_qubits: Number of qubits.

        Returns:
            A new density matrix of the same shape as rho.
        """
        pass

    def apply_statevector(
        self,
        state: torch.Tensor,
        n_qubits: int,
    ) -> torch.Tensor:
        """
        Convenience method: convert a pure state |psi> to rho = |psi><psi|,
        apply noise, and return the noisy density matrix.

        Args:
            state: Statevector tensor of shape (..., 2**n_qubits) with complex dtype.
            n_qubits: Number of qubits.

        Returns:
            A density matrix tensor of shape (..., dim, dim) representing the noisy state.
        """
        from ..backend.density_matrix import dm_from_statevector

        rho = dm_from_statevector(state)
        return self.apply_density_matrix(rho, n_qubits=n_qubits)


