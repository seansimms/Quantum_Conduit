"""Hybrid quantum-classical blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..backend.statevector import measure_expectation_z
from ..core.device import Device, default_device
from ..core.module import QuantumModule
from ..layers.ansatzes import HardwareEfficientAnsatz

if TYPE_CHECKING:
    pass


class QuantumBlock(QuantumModule):
    """
    A hybrid quantum-classical block that maps classical features to quantum expectations.

    This block takes classical input features, maps them to rotation parameters via
    a learned linear layer, applies a hardware-efficient ansatz to build a quantum
    state, and returns classical features built from per-qubit Z expectation values.

    The output is purely classical (one scalar per qubit) and can be fed into
    downstream PyTorch layers (e.g., nn.Linear) for further processing.

    Args:
        n_qubits: Number of qubits. Must be >= 1.
        depth: Number of layers in the hardware-efficient ansatz. Must be >= 1.
        in_features: Number of input classical features. Must be >= 1.
        device: Device specification. Can be Device, str, torch.device, or None.

    Attributes:
        n_qubits: Number of qubits.
        in_features: Number of input features.
        ansatz: HardwareEfficientAnsatz instance.
        encoder: Linear layer mapping input features to ansatz parameters.
        qdevice: Quantum device instance.

    Example:
        >>> block = QuantumBlock(n_qubits=2, depth=1, in_features=3)
        >>> x = torch.randn(4, 3)  # batch_size=4
        >>> output = block(x)
        >>> print(output.shape)  # (4, 2)
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        in_features: int,
        device: Device | None = None,
    ) -> None:
        """Initialize a QuantumBlock."""
        super().__init__(n_qubits=n_qubits, device=device or default_device())

        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")

        self.in_features = in_features

        # Create the ansatz
        self.ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits, depth=depth, device=self.qdevice
        )

        # Create encoder: maps input features to ansatz parameters
        num_parameters = self.ansatz.num_parameters
        self.encoder = nn.Linear(in_features, num_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map classical features to quantum expectations.

        Args:
            x: Input tensor of shape (batch_size, in_features). Must be a float tensor.

        Returns:
            Output tensor of shape (batch_size, n_qubits) containing Z expectation
            values for each qubit.

        Raises:
            ValueError: If x.shape[-1] != self.in_features.
        """
        # Validate input shape
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected x.shape[-1] == {self.in_features}, got {x.shape[-1]}"
            )

        # Move input to correct device and dtype
        x = x.to(dtype=torch.float32, device=self.qdevice.as_torch_device())

        # Map input features to ansatz parameters
        params = self.encoder(x)  # shape: (batch_size, num_parameters)

        # Build quantum state using the ansatz
        state = self.ansatz(params)  # shape: (batch_size, 2**n_qubits)

        # Compute Z expectation for each qubit
        expectations = []
        for qubit_idx in range(self.n_qubits):
            exp_val = measure_expectation_z(
                state, qubit=qubit_idx, n_qubits=self.n_qubits
            )
            expectations.append(exp_val)

        # Stack expectations along last dimension
        # Each exp_val has shape (batch_size,), so stacking gives (batch_size, n_qubits)
        features = torch.stack(expectations, dim=-1)

        return features

