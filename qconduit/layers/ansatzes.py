"""Parametric ansätze for variational quantum circuits."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import torch

from ..backend.statevector import apply_gate, apply_two_qubit_gate, zero_state
from ..core.device import Device, default_device
from ..core.module import QuantumModule
from ..gates.standard import CNOT, RX
from ..circuit import QuantumCircuit

if TYPE_CHECKING:
    pass


class ParametricAnsatz(QuantumModule):
    """
    Base class for parametric ansätze that map parameter tensors to quantum states.

    This class extends QuantumModule and provides a foundation for ansätze that
    take parameter tensors and return statevectors. Subclasses must implement
    the forward() method to define the specific ansatz structure.

    Args:
        n_qubits: Number of qubits. Must be >= 1.
        device: Device specification. Can be Device, str, torch.device, or None.

    Attributes:
        n_qubits: Number of qubits.
        qdevice: Quantum device instance.
    """

    def __init__(self, n_qubits: int, device: Device | None = None) -> None:
        """Initialize a ParametricAnsatz."""
        super().__init__(n_qubits=n_qubits, device=device or default_device())

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Map parameters to a quantum statevector.

        Args:
            params: Parameter tensor of shape (num_parameters,) for unbatched
                or (batch_size, num_parameters) for batched inputs. Must be
                a float tensor.

        Returns:
            Complex statevector tensor of shape (2**n_qubits,) for unbatched
            or (batch_size, 2**n_qubits) for batched inputs.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError(
            "ParametricAnsatz.forward() must be implemented by subclasses"
        )

    def build_circuit(self, params: Optional[torch.Tensor] = None) -> QuantumCircuit:
        """
        Build a QuantumCircuit representing this ansatz.

        The default implementation raises NotImplementedError.
        Subclasses may override to provide a concrete circuit IR.

        Parameters
        ----------
        params:
            Optional 1D tensor of numeric parameter values. If provided,
            gates that depend on parameters should use these numeric
            values in their GateOp entries. If None, subclasses may
            choose default values (e.g. zeros).

        Returns
        -------
        QuantumCircuit
            A circuit with n_qubits equal to this ansatz's n_qubits.

        Raises
        ------
        NotImplementedError:
            If this method is not implemented by the subclass.
        """
        raise NotImplementedError("build_circuit is not implemented for this ansatz.")


class HardwareEfficientAnsatz(ParametricAnsatz):
    """
    Hardware-efficient ansatz with RX rotations and CNOT entangling gates.

    This ansatz implements a simple, widely-used pattern consisting of alternating
    layers of single-qubit RX rotations and CNOT entangling gates. Each layer
    applies RX(θ) rotations to all qubits, followed by a ladder of CNOT gates
    connecting adjacent qubits.

    The ansatz is fully differentiable via PyTorch autograd, making it suitable
    for variational quantum algorithms.

    Convention: Qubit 0 is the least significant bit (LSB) in the computational
    basis index, matching the convention used by apply_gate and apply_two_qubit_gate.

    Args:
        n_qubits: Number of qubits. Must be >= 1.
        depth: Number of layers. Must be >= 1.
        device: Device specification. Can be Device, str, torch.device, or None.

    Attributes:
        n_qubits: Number of qubits.
        depth: Number of layers.
        num_parameters: Total number of parameters (n_qubits * depth).
        qdevice: Quantum device instance.

    Example:
        >>> ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
        >>> params = torch.tensor([0.1, 0.2, 0.3, 0.4])
        >>> state = ansatz(params)
        >>> print(state.shape)  # (4,)
    """

    def __init__(self, n_qubits: int, depth: int, device: Device | None = None) -> None:
        """Initialize a HardwareEfficientAnsatz."""
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        super().__init__(n_qubits=n_qubits, device=device)
        self.depth = depth
        self.num_parameters = n_qubits * depth

    def _split_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """
        Reshape parameters from (..., num_parameters) to (..., depth, n_qubits).

        Args:
            params: Parameter tensor with last dimension equal to num_parameters.

        Returns:
            Reshaped tensor with last two dimensions (depth, n_qubits).

        Raises:
            ValueError: If params.shape[-1] != self.num_parameters.
        """
        if params.shape[-1] != self.num_parameters:
            raise ValueError(
                f"Expected params.shape[-1] == {self.num_parameters}, "
                f"got {params.shape[-1]}"
            )

        # Reshape last dimension to (depth, n_qubits)
        # For unbatched: (num_parameters,) -> (depth, n_qubits)
        # For batched: (*batch, num_parameters) -> (*batch, depth, n_qubits)
        return params.reshape(*params.shape[:-1], self.depth, self.n_qubits)

    def _iterate_layers(self) -> Iterable[Tuple[str, Tuple[int, ...], Optional[int]]]:
        """
        Yield the sequence of gate placements for this ansatz.

        Each element is (name, qubits, param_index_or_none).

        name:
            Gate name, e.g. "RX" or "CNOT".
        qubits:
            Target qubits for the gate.
        param_index_or_none:
            Index into the flat parameter vector for parametric gates,
            or None for non-parametric gates like CNOT.

        This is purely structural and uses only generic hardware-efficient
        patterns (layers of single-qubit rotations + entanglers).
        """
        param_idx = 0
        for layer_idx in range(self.depth):
            # Single-qubit RX rotations on all qubits
            for qubit_idx in range(self.n_qubits):
                yield ("RX", (qubit_idx,), param_idx)
                param_idx += 1

            # CNOT entangling gates (ladder pattern)
            if self.n_qubits > 1:
                for qubit_idx in range(self.n_qubits - 1):
                    yield ("CNOT", (qubit_idx, qubit_idx + 1), None)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Build the quantum state from parameters.

        Args:
            params: Parameter tensor of shape (num_parameters,) for unbatched
                or (batch_size, num_parameters) for batched inputs. Must be
                a float tensor.

        Returns:
            Complex statevector tensor of shape (2**n_qubits,) for unbatched
            or (batch_size, 2**n_qubits) for batched inputs.
        """
        # Ensure params is on the correct device and dtype
        params = params.to(dtype=torch.float32, device=self.qdevice.as_torch_device())

        # Determine batch shape
        is_batched = params.ndim > 1
        if is_batched:
            batch_shape = params.shape[:-1]
            batch_size = batch_shape[0] if len(batch_shape) == 1 else math.prod(batch_shape)
        else:
            batch_shape = None
            batch_size = 1

        # Reshape parameters to (..., depth, n_qubits)
        theta = self._split_parameters(params)

        # Initialize state
        state = zero_state(
            n_qubits=self.n_qubits,
            batch_shape=batch_shape,
            device=self.qdevice,
            dtype=torch.complex64,
        )

        # Apply layers
        for layer_idx in range(self.depth):
            # Single-qubit RX rotations
            for qubit_idx in range(self.n_qubits):
                if is_batched:
                    # Batched: process each batch element separately
                    # This is acceptable for v0.1 small test sizes
                    # Collect modified slices to avoid in-place operations
                    modified_slices = []
                    for b in range(batch_size):
                        # Extract angle tensor for this batch element (preserve gradients)
                        if len(batch_shape) == 1:
                            angle = theta[b, layer_idx, qubit_idx]
                            state_slice = state[b : b + 1, ...]
                        else:
                            # Multi-dimensional batch: flatten to 1D for indexing
                            flat_idx = b
                            batch_dims = []
                            remaining = flat_idx
                            for dim_size in batch_shape:
                                batch_dims.append(remaining % dim_size)
                                remaining //= dim_size
                            batch_idx = tuple(batch_dims)
                            angle = theta[batch_idx + (layer_idx, qubit_idx)]
                            state_slice = state[batch_idx + (slice(None),)]
                            state_slice = state_slice.unsqueeze(0)

                        # RX now accepts tensor inputs and preserves gradients
                        gate_rx = RX(
                            angle,
                            dtype=state.dtype,
                            device=state.device,
                        )
                        # Apply gate (this creates a new tensor, preserving gradients)
                        state_slice = apply_gate(
                            state_slice,
                            gate_rx,
                            qubit=qubit_idx,
                            n_qubits=self.n_qubits,
                        )
                        modified_slices.append(state_slice)

                    # Reconstruct state by concatenating modified slices
                    if len(batch_shape) == 1:
                        state = torch.cat(modified_slices, dim=0)
                    else:
                        # For multi-dimensional batch, reshape and reconstruct
                        # This is more complex, but for v0.1 we'll handle 1D batches
                        state = torch.cat(modified_slices, dim=0)
                        # Reshape back to original batch shape
                        state = state.reshape(*batch_shape, 2**self.n_qubits)
                else:
                    # Unbatched: theta is (depth, n_qubits), pass tensor directly
                    angle = theta[layer_idx, qubit_idx]
                    gate_rx = RX(
                        angle,
                        dtype=state.dtype,
                        device=state.device,
                    )
                    state = apply_gate(
                        state, gate_rx, qubit=qubit_idx, n_qubits=self.n_qubits
                    )

            # CNOT entangling gates (ladder pattern)
            if self.n_qubits > 1:
                for qubit_idx in range(self.n_qubits - 1):
                    gate_cnot = CNOT(
                        dtype=state.dtype,
                        device=state.device,
                        control_first=True,
                    )
                    state = apply_two_qubit_gate(
                        state,
                        gate_cnot,
                        qubit1=qubit_idx,
                        qubit2=qubit_idx + 1,
                        n_qubits=self.n_qubits,
                    )

        return state

    def build_circuit(self, params: Optional[torch.Tensor] = None) -> QuantumCircuit:
        """
        Build a QuantumCircuit representing this ansatz.

        The circuit structure mirrors the forward() method: layers of RX rotations
        followed by CNOT entangling gates in a ladder pattern.

        Parameters
        ----------
        params:
            Optional 1D tensor of numeric parameter values. If provided,
            must have length equal to num_parameters. If None, parametric
            gates default to zero angles.

        Returns
        -------
        QuantumCircuit
            A circuit with n_qubits equal to this ansatz's n_qubits.

        Raises
        ------
        ValueError:
            If params is provided but has incorrect shape.
        """
        circuit = QuantumCircuit(self.n_qubits)

        param_values: Optional[torch.Tensor]
        if params is not None:
            if params.dim() != 1 or params.numel() != self.num_parameters:
                raise ValueError(
                    "params must be a 1D tensor with length equal to num_parameters."
                )
            param_values = params.detach().cpu()
        else:
            param_values = None

        for name, qubits, param_index in self._iterate_layers():
            if param_index is not None:
                if param_values is None:
                    # Default to zero angle when no params supplied
                    angle = 0.0
                else:
                    angle = float(param_values[param_index].item())
                circuit.add_gate(name=name, qubits=qubits, params=(angle,))
            else:
                circuit.add_gate(name=name, qubits=qubits, params=None)

        return circuit

