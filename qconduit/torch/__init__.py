"""PyTorch integration for qconduit quantum circuits.

This module provides PyTorch nn.Module integration for variational quantum
circuits, allowing quantum circuits to be used as layers in PyTorch models.

Example:
    >>> import torch
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.operators import PauliTerm, PauliSum
    >>> from qconduit.torch import QuantumModule
    >>>
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    >>> H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
    >>> module = QuantumModule(ansatz, H, gradient_method="parameter_shift")
    >>> optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    >>> loss = module()
    >>> loss.backward()
    >>> optimizer.step()

Gradient Methods:
    - "autograd": Uses PyTorch's automatic differentiation if the backend
      supports it. Falls back to parameter-shift if not available.
    - "parameter_shift": Uses the deterministic parameter-shift rule for
      computing gradients. Guaranteed to work for all ansätze using Rx, Ry, Rz
      rotation gates.
"""

from qconduit.torch.layers import QuantumModule
from qconduit.torch.param_shift import parameter_shift_gradients

__all__ = [
    "QuantumModule",
    "parameter_shift_gradients",
]

