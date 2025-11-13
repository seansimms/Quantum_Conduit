"""VQE example: finding ground-state energy of a 2-qubit toy Hamiltonian.

This example demonstrates the Variational Quantum Eigensolver (VQE) algorithm
using a HardwareEfficientAnsatz to approximate the ground-state energy of a
simple diagonal Hamiltonian inspired by H₂ molecular structure.
"""

from __future__ import annotations

import torch

import qconduit as qc
from qconduit.layers import HardwareEfficientAnsatz


def main() -> None:
    """Run VQE optimization to find ground-state energy."""
    # Set seed for reproducibility
    torch.manual_seed(0)

    # Configuration
    n_qubits = 2
    depth = 2

    # Define a simple diagonal Hamiltonian for a 2-qubit system
    # Computational basis: |00>, |01>, |10>, |11> (indices 0..3)
    # This mimics a symmetric "bonding/antibonding"-like spectrum
    # Ground state is |00> with energy 0.0
    hamiltonian_diag = torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32)

    # Construct the ansatz
    ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, depth=depth)

    # Construct VQE
    vqe = qc.VQE(ansatz=ansatz, hamiltonian=hamiltonian_diag)

    # Initialize learnable parameters (small random initialization)
    params = torch.nn.Parameter(0.1 * torch.randn(ansatz.num_parameters, dtype=torch.float32))

    # Set up optimizer
    optimizer = torch.optim.Adam([params], lr=0.1)

    # Training loop
    num_steps = 50
    print("Starting VQE optimization...")
    print(f"Initial energy: {vqe.energy(params).item():.6f}")

    for step in range(num_steps):
        optimizer.zero_grad()
        energy = vqe.energy(params)  # scalar
        energy.backward()
        optimizer.step()

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:03d}: energy = {energy.item():.6f}")

    # Final results
    final_energy = vqe.energy(params).item()
    print(f"\nFinal estimated ground-state energy: {final_energy:.6f}")
    print("Reference ground-state energy (toy): 0.000000")


if __name__ == "__main__":
    main()

