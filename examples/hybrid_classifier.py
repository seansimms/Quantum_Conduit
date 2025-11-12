"""Hybrid quantum-classical classifier example.

This example demonstrates a hybrid quantum-classical neural network that uses
QuantumBlock to process classical features through a quantum circuit, then
feeds the quantum expectations into a classical classifier head.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

import qconduit as qc


def make_xor_dataset(n_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic XOR-style 2D dataset.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        Tuple of (features, labels) where:
        - features: Tensor of shape (n_samples, 2) with values in [-1, 1]
        - labels: Tensor of shape (n_samples,) with values 0 or 1
    """
    # Sample uniform points in [-1, 1]^2
    x = 2 * torch.rand(n_samples, 2) - 1

    # Define labels: points where x and y have the same sign -> label 1, else 0
    y = ((x[:, 0] * x[:, 1]) > 0).long()

    return x, y


class HybridClassifier(nn.Module):
    """Hybrid quantum-classical classifier."""

    def __init__(self) -> None:
        """Initialize the hybrid classifier."""
        super().__init__()
        n_qubits = 2
        depth = 1
        in_features = 2  # 2D input features

        # Quantum block: maps classical features to quantum expectations
        self.quantum = qc.QuantumBlock(n_qubits=n_qubits, depth=depth, in_features=in_features)

        # Classical head: maps quantum features to class logits
        self.head = nn.Sequential(nn.Linear(n_qubits, 2))  # 2 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 2).

        Returns:
            Logits tensor of shape (batch_size, 2).
        """
        q_features = self.quantum(x)  # shape (batch_size, n_qubits)
        logits = self.head(q_features)  # shape (batch_size, 2)
        return logits


def main() -> None:
    """Train and evaluate the hybrid classifier."""
    # Set seed for reproducibility
    torch.manual_seed(0)

    # Create dataset
    x, y = make_xor_dataset(200)

    # Split into train/test (first 160 for train, last 40 for test)
    train_x = x[:160]
    train_y = y[:160]
    test_x = x[160:]
    test_y = y[160:]

    # Model, loss, optimizer
    model = HybridClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Training loop
    num_epochs = 10
    batch_size = 32

    print("Training hybrid classifier...")
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = torch.randperm(train_x.size(0))
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, train_x.size(0), batch_size):
            batch_idx = perm[i : i + batch_size]
            xb = train_x[batch_idx]
            yb = train_y[batch_idx]

            # Forward pass
            logits = model(xb)
            loss = criterion(logits, yb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1:02d}: train loss = {avg_loss:.4f}")

    # Evaluate on test set
    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(dim=-1)
        acc = (pred == test_y).float().mean().item()

    print(f"\nTest accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()

