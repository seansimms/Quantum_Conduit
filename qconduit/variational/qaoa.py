"""Quantum Approximate Optimization Algorithm (QAOA) driver."""

from __future__ import annotations

from typing import Optional

import torch

from qconduit.core.device import default_device
from qconduit.operators import PauliSum
from qconduit.variational.ansatz import QAOAAnsatz
from qconduit.variational.vqe import VQEResult, run_vqe

# Alias for clarity
QAOAResult = VQEResult


def run_qaoa(
    cost_hamiltonian: PauliSum,
    num_qubits: int,
    depth: int,
    initial_params: Optional[torch.Tensor] = None,
    optimizer_name: str = "adam",
    max_iterations: int = 200,
    learning_rate: float = 0.05,
    tol_rel: float = 1e-6,
    device: Optional[torch.device] = None,
) -> QAOAResult:
    """
    Run a standard QAOA optimization for a given cost Hamiltonian and depth.

    This constructs a QAOAAnsatz with an X-mixer and optimizes its parameters
    (γ, β) to minimize the cost Hamiltonian expectation value.

    Parameters
    ----------
    cost_hamiltonian:
        PauliSum representing the cost Hamiltonian H_C.
    num_qubits:
        Number of qubits.
    depth:
        Number of QAOA layers P.
    initial_params:
        Optional initial parameters of shape (2 * depth,). If None, initialized
        from a small normal distribution with mean 0 and std 0.1.
    optimizer_name:
        "sgd" or "adam".
    max_iterations:
        Maximum number of optimization steps.
    learning_rate:
        Optimizer learning rate.
    tol_rel:
        Relative energy-change tolerance for early stopping.
    device:
        Optional computation device.

    Returns
    -------
    QAOAResult
        Alias of VQEResult summarizing the QAOA run.
    """
    # Validate inputs
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    # Construct ansatz
    ansatz = QAOAAnsatz(
        num_qubits=num_qubits,
        depth=depth,
        cost_hamiltonian=cost_hamiltonian,
    )

    # Initialize parameters if not provided
    if initial_params is None:
        if device is None:
            qdevice = default_device()
            device0 = qdevice.as_torch_device()
        else:
            device0 = device

        generator = torch.Generator(device=device0)
        generator.manual_seed(0)
        initial_params = torch.normal(
            mean=0.0,
            std=0.1,
            size=(ansatz.num_parameters,),
            generator=generator,
            device=device0,
            dtype=torch.float64,
        )
    else:
        # Validate provided parameters
        if initial_params.ndim != 1:
            raise ValueError(f"initial_params must be 1D, got shape {initial_params.shape}")
        if initial_params.shape[0] != ansatz.num_parameters:
            raise ValueError(
                f"initial_params length {initial_params.shape[0]} does not match "
                f"ansatz.num_parameters {ansatz.num_parameters}"
            )

    # Run VQE
    result = run_vqe(
        hamiltonian=cost_hamiltonian,
        ansatz=ansatz,
        initial_params=initial_params,
        optimizer_name=optimizer_name,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tol_rel=tol_rel,
        device=device,
    )

    return result


__all__ = [
    "QAOAResult",
    "run_qaoa",
]


