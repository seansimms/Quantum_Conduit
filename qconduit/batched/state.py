"""Batched statevector containers and utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from qconduit.core.device import default_device


@dataclass(frozen=True)
class BatchedState:
    """
    Container for batched pure statevectors.

    This class represents a batch of quantum states, where each state is a
    complex statevector of dimension 2**n_qubits. The batch dimension is the
    first dimension, so the shape is (B, 2**n_qubits).

    Attributes
    ----------
    states : torch.Tensor
        Complex tensor of shape (B, dim) where dim = 2**n_qubits.
        Each row represents a single quantum state.
    n_qubits : int
        Number of qubits. Must satisfy dim = 2**n_qubits.

    Examples
    --------
    >>> import torch
    >>> from qconduit.batched import BatchedState
    >>> # Create a batch of 3 states for 2 qubits
    >>> states = torch.randn(3, 4, dtype=torch.complex128)
    >>> states = states / torch.linalg.norm(states, dim=1, keepdim=True)
    >>> batched = BatchedState(states, n_qubits=2)
    >>> print(batched.norms())  # Should be close to 1.0 for each state
    """

    states: torch.Tensor
    n_qubits: int

    def __post_init__(self) -> None:
        """Validate batched state structure."""
        # Check that states is 2D
        if self.states.ndim != 2:
            raise ValueError(
                f"states must be 2D tensor with shape (B, dim), got shape {self.states.shape}"
            )

        # Infer dimension and validate it's a power of 2
        dim = self.states.shape[1]
        n_inferred = int(math.log2(dim))
        if 2**n_inferred != dim:
            raise ValueError(
                f"states dimension {dim} is not a power of 2. "
                f"Expected shape (B, 2**n) for some n >= 1."
            )

        # Validate n_qubits matches dimension
        if self.n_qubits != n_inferred:
            raise ValueError(
                f"n_qubits={self.n_qubits} does not match inferred n_qubits={n_inferred} "
                f"from dimension {dim}"
            )

        # Ensure complex dtype
        if not torch.is_complex(self.states):
            # Convert to complex128 on default device
            device = self.states.device
            qdevice = default_device()
            target_device = qdevice.as_torch_device()
            new_states = self.states.to(dtype=torch.complex128, device=target_device)
            object.__setattr__(self, "states", new_states)

        # Check normalization: each row should have non-zero norm
        norms = torch.linalg.norm(self.states, dim=1)
        zero_norm_mask = norms < 1e-12
        if zero_norm_mask.any():
            zero_indices = torch.where(zero_norm_mask)[0].tolist()
            raise ValueError(
                f"Found {len(zero_indices)} zero-norm states at batch indices: {zero_indices}"
            )

    @classmethod
    def from_statevector(
        cls, state: torch.Tensor, batch_dim: int = 1
    ) -> BatchedState:
        """
        Create a BatchedState from a single statevector or already-batched states.

        Parameters
        ----------
        state : torch.Tensor
            Statevector tensor. If 1D with shape (dim,), it will be unsqueezed
            to shape (1, dim). If 2D with shape (B, dim), it is used as-is.
        batch_dim : int, optional
            Unused parameter for API compatibility. Default is 1.

        Returns
        -------
        BatchedState
            BatchedState instance containing the state(s).

        Examples
        --------
        >>> import torch
        >>> from qconduit.batched import BatchedState
        >>> # Single statevector
        >>> psi = torch.randn(4, dtype=torch.complex128)
        >>> psi = psi / torch.linalg.norm(psi)
        >>> batched = BatchedState.from_statevector(psi)
        >>> print(batched.states.shape)  # (1, 4)
        """
        if state.ndim == 1:
            # Single statevector: add batch dimension
            state = state.unsqueeze(0)
        elif state.ndim == 2:
            # Already batched
            pass
        else:
            raise ValueError(
                f"state must be 1D or 2D, got shape {state.shape}"
            )

        # Infer n_qubits from dimension
        dim = state.shape[1]
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"state dimension {dim} is not a power of 2"
            )

        return cls(states=state, n_qubits=n_qubits)

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> BatchedState:
        """
        Move and/or cast the batched states to a different device and/or dtype.

        Parameters
        ----------
        device : torch.device, optional
            Target device. If None, keeps current device.
        dtype : torch.dtype, optional
            Target dtype. If None, keeps current dtype. Must be complex dtype.

        Returns
        -------
        BatchedState
            New BatchedState with moved/cast states.

        Examples
        --------
        >>> import torch
        >>> from qconduit.batched import BatchedState
        >>> batched = BatchedState(...)
        >>> # Move to CPU and cast to complex128
        >>> batched_cpu = batched.to(device=torch.device("cpu"), dtype=torch.complex128)
        """
        new_states = self.states
        if device is not None:
            new_states = new_states.to(device=device)
        if dtype is not None:
            # Check if dtype is complex
            if dtype not in (torch.complex64, torch.complex128):
                raise ValueError(f"dtype must be complex (complex64 or complex128), got {dtype}")
            new_states = new_states.to(dtype=dtype)
        return BatchedState(states=new_states, n_qubits=self.n_qubits)

    def unstack(self) -> tuple[torch.Tensor, ...]:
        """
        Return a tuple of individual statevectors (one per batch element).

        This is mainly useful for debugging or when you need to process states
        individually.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of B statevectors, each with shape (dim,).

        Examples
        --------
        >>> import torch
        >>> from qconduit.batched import BatchedState
        >>> batched = BatchedState(...)
        >>> states_list = batched.unstack()
        >>> print(len(states_list))  # B
        >>> print(states_list[0].shape)  # (dim,)
        """
        return tuple(self.states[i] for i in range(self.states.shape[0]))

    def norms(self) -> torch.Tensor:
        """
        Compute the Euclidean norm of each state in the batch.

        Returns
        -------
        torch.Tensor
            Real tensor of shape (B,) containing the norm of each state.

        Examples
        --------
        >>> import torch
        >>> from qconduit.batched import BatchedState
        >>> batched = BatchedState(...)
        >>> norms = batched.norms()
        >>> print(norms.shape)  # (B,)
        """
        return torch.linalg.norm(self.states, dim=1)

    def renormalize(self) -> BatchedState:
        """
        Return a new BatchedState with each state normalized to unit norm.

        Returns
        -------
        BatchedState
            New BatchedState with normalized states.

        Examples
        --------
        >>> import torch
        >>> from qconduit.batched import BatchedState
        >>> # Create unnormalized batch
        >>> states = torch.randn(3, 4, dtype=torch.complex128)
        >>> batched = BatchedState.from_statevector(states)
        >>> # Normalize
        >>> normalized = batched.renormalize()
        >>> print(normalized.norms())  # Should all be close to 1.0
        """
        norms = self.norms().unsqueeze(1)  # (B, 1)
        # Avoid division by zero (shouldn't happen due to validation, but be safe)
        norms = torch.clamp(norms, min=1e-12)
        normalized_states = self.states / norms
        return BatchedState(states=normalized_states, n_qubits=self.n_qubits)


__all__ = ["BatchedState"]

