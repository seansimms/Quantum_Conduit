"""Training loop and utilities for VQE optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import torch
from torch.optim import Optimizer

from ..algorithms.vqe import VQE


@dataclass
class TrainingStepInfo:
    """
    Metrics recorded at each training step.

    This is a generic container for scalar metrics during optimization.
    No references to proprietary concepts are included.

    Args:
        step: Step number (1-indexed).
        epoch: Epoch number (1-indexed).
        energy: Energy value at this step.
        loss: Loss value at this step. For VQE, loss equals energy.
        grad_norm: Gradient norm, if available. None if gradients are not computed.
        param_norm: Parameter norm at this step, if available.
    """

    step: int
    epoch: int
    energy: float
    loss: float
    grad_norm: Optional[float] = None
    param_norm: Optional[float] = None


@dataclass
class TrainingHistory:
    """
    Accumulated training metrics over the course of optimization.

    This is a generic container for tracking training progress.
    """

    steps: List[TrainingStepInfo] = field(default_factory=list)

    def record(self, info: TrainingStepInfo) -> None:
        """
        Record a training step.

        Args:
            info: Training step information to record.
        """
        self.steps.append(info)

    def best_energy(self) -> Optional[float]:
        """
        Get the best (minimum) energy recorded during training.

        Returns:
            Minimum energy value, or None if no steps have been recorded.
        """
        if not self.steps:
            return None
        return min(step.energy for step in self.steps)

    def final_energy(self) -> Optional[float]:
        """
        Get the final energy recorded during training.

        Returns:
            Final energy value, or None if no steps have been recorded.
        """
        if not self.steps:
            return None
        return self.steps[-1].energy

    def num_steps(self) -> int:
        """
        Get the number of training steps recorded.

        Returns:
            Number of steps recorded.
        """
        return len(self.steps)


class TrainingCallback(Protocol):
    """
    Protocol for training callbacks.

    A callback is a callable that receives TrainingStepInfo at each step
    and can perform side effects (e.g., logging, checkpointing).
    """

    def __call__(self, info: TrainingStepInfo) -> None:
        """
        Called at each training step.

        Args:
            info: Training step information.
        """
        ...


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping during training.

    This uses standard early-stopping semantics: training stops if no
    improvement (by at least min_delta) is observed for patience steps.

    Args:
        patience: Number of steps to wait without improvement before stopping.
            If 0, early stopping is disabled.
        min_delta: Minimum improvement required to reset the patience counter.
            Defaults to 0.0.
    """

    patience: int = 0
    min_delta: float = 0.0


class VQETrainer:
    """
    Generic trainer for VQE optimization.

    This class provides a simple training loop for minimizing a scalar
    objective (energy) using gradient-based optimization. It tracks metrics,
    supports early stopping, maintains a best checkpoint, and invokes callbacks.

    This is purely generic infrastructure around minimizing E(θ); no
    proprietary tricks or domain-specific logic is included.

    Args:
        vqe: VQE instance to optimize.
        params: Parameter tensor to optimize. Must be a torch.nn.Parameter.
        optimizer: PyTorch optimizer instance.
        early_stopping: Optional early stopping configuration. If None, no
            early stopping is performed.

    Attributes:
        vqe: VQE instance.
        params: Parameter tensor.
        optimizer: Optimizer instance.
        early_stopping: Early stopping configuration.
        history: Training history containing all recorded steps.
    """

    def __init__(
        self,
        vqe: VQE,
        params: torch.nn.Parameter,
        optimizer: Optimizer,
        early_stopping: Optional[EarlyStoppingConfig] = None,
    ) -> None:
        """Initialize a VQETrainer."""
        self.vqe = vqe
        self.params = params
        self.optimizer = optimizer
        self.early_stopping = early_stopping

        self.history = TrainingHistory()
        self._best_energy: Optional[float] = None
        self._best_params: Optional[torch.Tensor] = None
        self._steps_taken = 0
        self._epochs_completed = 0

    def _compute_step_metrics(self) -> TrainingStepInfo:
        """
        Compute metrics for one forward/backward pass.

        Returns:
            TrainingStepInfo with metrics for the current step.

        Raises:
            ValueError: If VQE.energy does not return a scalar tensor.
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute energy
        energy_tensor = self.vqe.energy(self.params)

        # Ensure it is a scalar tensor
        if energy_tensor.ndim != 0:
            raise ValueError("VQE.energy must return a scalar tensor.")

        # For VQE, loss equals energy
        loss = energy_tensor

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm: Optional[float] = None
        if self.params.grad is not None:
            grad_norm = self.params.grad.norm().item()

        # Compute parameter norm
        param_norm = self.params.data.norm().item()

        # Convert to Python floats
        energy_value = energy_tensor.item()
        loss_value = loss.item()

        # Create step info
        # For single-epoch training, epoch is always 1
        info = TrainingStepInfo(
            step=self._steps_taken + 1,
            epoch=1,
            energy=float(energy_value),
            loss=float(loss_value),
            grad_norm=grad_norm,
            param_norm=param_norm,
        )

        return info

    def run(
        self,
        max_steps: int,
        max_epochs: int = 1,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> TrainingHistory:
        """
        Run the training loop.

        This performs a simple single-epoch training loop over max_steps steps.
        The epoch field in TrainingStepInfo is set to 1 for all steps.

        Args:
            max_steps: Maximum number of optimization steps to perform.
            max_epochs: Maximum number of epochs. Currently unused; training
                runs for max_steps steps in a single epoch.
            callbacks: Optional list of callbacks to invoke at each step.

        Returns:
            TrainingHistory containing all recorded steps.
        """
        callbacks = callbacks or []

        # Early stopping state
        best_energy_so_far: Optional[float] = None
        steps_since_improvement = 0

        # Training loop
        for step_idx in range(max_steps):
            # Compute metrics
            info = self._compute_step_metrics()

            # Optimizer step
            # LBFGS requires a closure function
            if isinstance(self.optimizer, torch.optim.LBFGS):
                def closure():
                    self.optimizer.zero_grad()
                    energy = self.vqe.energy(self.params)
                    energy.backward()
                    return energy

                self.optimizer.step(closure)
            else:
                self.optimizer.step()

            # Update counters
            self._steps_taken += 1
            self._epochs_completed = 1  # Single-epoch training

            # Update history
            self.history.record(info)

            # Update best checkpoint
            if self._best_energy is None or info.energy < self._best_energy - 1e-12:
                self._best_energy = info.energy
                self._best_params = self.params.detach().clone()

            # Early stopping logic
            if self.early_stopping is not None and self.early_stopping.patience > 0:
                if best_energy_so_far is None:
                    best_energy_so_far = info.energy
                    steps_since_improvement = 0
                else:
                    if info.energy < best_energy_so_far - self.early_stopping.min_delta:
                        # Improvement detected
                        best_energy_so_far = info.energy
                        steps_since_improvement = 0
                    else:
                        # No improvement
                        steps_since_improvement += 1

                # Check if we should stop
                if steps_since_improvement >= self.early_stopping.patience:
                    break

            # Invoke callbacks
            for cb in callbacks:
                cb(info)

        return self.history

    def best_energy(self) -> Optional[float]:
        """
        Get the best energy encountered during training.

        Returns:
            Best energy value, or None if no training has been performed.
        """
        return self._best_energy

    def best_params(self) -> Optional[torch.Tensor]:
        """
        Get a copy of the best parameters encountered during training.

        Returns:
            Detached copy of best parameters, or None if no training has been
            performed. The returned tensor is suitable for reuse.
        """
        if self._best_params is not None:
            return self._best_params.clone()
        return None

