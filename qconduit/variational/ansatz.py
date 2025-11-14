"""Variational ansatz families for VQE and QAOA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from qconduit.circuit import QuantumCircuit
from qconduit.operators import PauliSum
from qconduit.time_evolution.circuits import build_trotter_step_circuit


class VariationalAnsatz(Protocol):
    """
    Protocol for variational ansatz families.

    A variational ansatz maps a parameter vector θ ∈ ℝ^d to a QuantumCircuit
    acting on `num_qubits` qubits.
    """

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits this ansatz acts on."""
        ...

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in this ansatz."""
        ...

    def build_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """
        Construct a QuantumCircuit for the given parameter vector.

        Parameters
        ----------
        params:
            1D tensor of shape (num_parameters,) on any device. Implementations
            must not mutate this tensor.

        Returns
        -------
        QuantumCircuit
            Circuit acting on `num_qubits` qubits.
        """
        ...


@dataclass(frozen=True)
class HardwareEfficientAnsatz:
    """
    Standard hardware-efficient variational ansatz on a linear chain of qubits.

    Each layer consists of:
        - Single-qubit rotations on each qubit: Rx and Rz with independent angles.
        - CNOT entanglers between nearest neighbors (0-1, 1-2, ..., n-2 -> n-1).

    The total number of parameters is:

        num_parameters = num_layers * num_qubits * 2,

    corresponding to [Rx, Rz] per qubit per layer.
    """

    num_qubits: int
    num_layers: int

    def __post_init__(self) -> None:
        """Validate ansatz parameters."""
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in this ansatz."""
        return self.num_qubits * self.num_layers * 2

    def build_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """
        Build a circuit for the given parameters.

        Parameters
        ----------
        params:
            1D tensor of shape (num_parameters,).

        Returns
        -------
        QuantumCircuit
            Circuit implementing the ansatz.

        Raises
        ------
        ValueError:
            If params has incorrect shape.
        """
        if params.ndim != 1:
            raise ValueError(f"params must be 1D, got shape {params.shape}")
        if params.shape[0] != self.num_parameters:
            raise ValueError(
                f"params length {params.shape[0]} does not match "
                f"num_parameters {self.num_parameters}"
            )

        circuit = QuantumCircuit(self.num_qubits)

        for layer in range(self.num_layers):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                base = 2 * (layer * self.num_qubits + qubit)
                theta_rx = float(params[base].item())
                theta_rz = float(params[base + 1].item())

                circuit.add_gate("RX", [qubit], params=[theta_rx])
                circuit.add_gate("RZ", [qubit], params=[theta_rz])

            # CNOT entanglers between nearest neighbors
            for qubit in range(self.num_qubits - 1):
                circuit.add_gate("CNOT", [qubit, qubit + 1])

        return circuit


@dataclass(frozen=True)
class LayeredEntanglerAnsatz:
    """
    General layered entangler ansatz with optional ring connectivity.

    Each layer applies:
        - Ry and Rz rotations on each qubit.
        - CX entanglers between neighbors (linear chain or ring).

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    num_layers:
        Number of repeated layers.
    ring_entanglement:
        If True, entangle last qubit with first as well (ring). If False, only
        use linear chain entanglement.
    """

    num_qubits: int
    num_layers: int
    ring_entanglement: bool = False

    def __post_init__(self) -> None:
        """Validate ansatz parameters."""
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in this ansatz."""
        return self.num_qubits * self.num_layers * 2

    def build_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """
        Build a circuit for the given parameters.

        Parameters
        ----------
        params:
            1D tensor of shape (num_parameters,).

        Returns
        -------
        QuantumCircuit
            Circuit implementing the ansatz.

        Raises
        ------
        ValueError:
            If params has incorrect shape.
        """
        if params.ndim != 1:
            raise ValueError(f"params must be 1D, got shape {params.shape}")
        if params.shape[0] != self.num_parameters:
            raise ValueError(
                f"params length {params.shape[0]} does not match "
                f"num_parameters {self.num_parameters}"
            )

        circuit = QuantumCircuit(self.num_qubits)

        for layer in range(self.num_layers):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                base = 2 * (layer * self.num_qubits + qubit)
                theta_ry = float(params[base].item())
                theta_rz = float(params[base + 1].item())

                circuit.add_gate("RY", [qubit], params=[theta_ry])
                circuit.add_gate("RZ", [qubit], params=[theta_rz])

            # Linear chain entanglers
            for qubit in range(self.num_qubits - 1):
                circuit.add_gate("CNOT", [qubit, qubit + 1])

            # Ring entanglement if requested
            if self.ring_entanglement and self.num_qubits > 2:
                circuit.add_gate("CNOT", [self.num_qubits - 1, 0])

        return circuit


@dataclass(frozen=True)
class QAOAAnsatz:
    """
    Standard QAOA ansatz with cost and mixer unitaries:

        U_QAOA(γ, β) = ∏_{p=1}^P e^{-i β_p H_M} e^{-i γ_p H_C},

    where H_C is the cost Hamiltonian and H_M = ∑_i X_i is the mixer.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    depth:
        Number of QAOA layers P.
    cost_hamiltonian:
        PauliSum representing the cost Hamiltonian H_C.
    """

    num_qubits: int
    depth: int
    cost_hamiltonian: PauliSum

    def __post_init__(self) -> None:
        """Validate ansatz parameters."""
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.cost_hamiltonian.n_qubits() != 0 and self.cost_hamiltonian.n_qubits() != self.num_qubits:
            raise ValueError(
                f"cost_hamiltonian.n_qubits() = {self.cost_hamiltonian.n_qubits()} "
                f"does not match num_qubits = {self.num_qubits}"
            )

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in this ansatz."""
        return 2 * self.depth

    def build_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """
        Build a QAOA circuit for the given parameters.

        Parameters
        ----------
        params:
            1D tensor of shape (2 * depth,) containing [γ_1, ..., γ_P, β_1, ..., β_P].

        Returns
        -------
        QuantumCircuit
            Circuit implementing the QAOA ansatz.

        Raises
        ------
        ValueError:
            If params has incorrect shape.
        """
        if params.ndim != 1:
            raise ValueError(f"params must be 1D, got shape {params.shape}")
        if params.shape[0] != 2 * self.depth:
            raise ValueError(
                f"params length {params.shape[0]} does not match "
                f"2 * depth = {2 * self.depth}"
            )

        circuit = QuantumCircuit(self.num_qubits)

        # Initialize in |+⟩^⊗n
        for qubit in range(self.num_qubits):
            circuit.add_gate("H", [qubit])

        # Split parameters
        gammas = params[: self.depth]
        betas = params[self.depth :]

        # Apply QAOA layers
        for p in range(self.depth):
            gamma_p = float(gammas[p].item())
            beta_p = float(betas[p].item())

            # Apply cost unitary exp(-i γ_p H_C)
            # Use first-order Trotter step with dt = gamma_p
            cost_step = build_trotter_step_circuit(
                hamiltonian=self.cost_hamiltonian,
                dt=gamma_p,
                n_qubits=self.num_qubits,
                order=1,
            )
            # Append all operations from the cost step
            for op in cost_step.ops:
                circuit.add_gate(
                    op.name,
                    list(op.qubits),
                    params=list(op.params) if op.params is not None else None,
                )

            # Apply mixer unitary exp(-i β_p ∑_i X_i)
            # This is equivalent to Rx(2*β_p) on each qubit
            for qubit in range(self.num_qubits):
                circuit.add_gate("RX", [qubit], params=[2.0 * beta_p])

        return circuit


__all__ = [
    "VariationalAnsatz",
    "HardwareEfficientAnsatz",
    "LayeredEntanglerAnsatz",
    "QAOAAnsatz",
]


