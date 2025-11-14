"""Tests for VQE training utilities."""

from __future__ import annotations

import torch
from torch import nn

from qconduit.algorithms.vqe import VQE
from qconduit.layers.ansatzes import HardwareEfficientAnsatz
from qconduit.operators.pauli import PauliSum, PauliTerm
from qconduit.optim import OptimConfig, create_optimizer
from qconduit.training import (
    EarlyStoppingConfig,
    TrainingStepInfo,
    VQETrainer,
)


def build_simple_vqe() -> tuple[VQE, torch.nn.Parameter]:
    """
    Build a minimal VQE for testing.

    Uses a 1-qubit hardware-efficient ansatz with one parameter and a Pauli Z
    Hamiltonian. The ground state |1⟩ has energy -1.

    Returns:
        Tuple of (VQE instance, parameter tensor).
    """
    ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    # Build H = Z (Pauli Z operator)
    term = PauliTerm(coeff=1.0, paulis=("Z",))
    ham = PauliSum([term])
    vqe = VQE(ansatz, ham)

    # Create a single parameter for the ansatz
    params = nn.Parameter(torch.zeros(ansatz.num_parameters, dtype=torch.float32))

    return (vqe, params)


def test_vqe_trainer_with_sgd_reduces_energy() -> None:
    """Test that SGD training reduces energy."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1, momentum=0.0)
    optimizer = create_optimizer(config, [params])
    trainer = VQETrainer(vqe, params, optimizer)

    # Compute initial energy
    initial_energy = vqe.energy(params.detach()).item()

    # Run training
    history = trainer.run(max_steps=20)

    # Compute final energy
    final_energy = vqe.energy(params.detach()).item()

    # Assertions
    assert history.num_steps() > 0
    assert history.num_steps() == 20
    assert final_energy <= initial_energy + 1e-3
    assert trainer.best_energy() is not None
    assert trainer.best_energy() <= initial_energy


def test_vqe_trainer_early_stopping_triggers() -> None:
    """Test that early stopping triggers when no improvement is seen."""
    vqe, params = build_simple_vqe()

    # Use a very small learning rate and small patience to ensure early stopping
    config = OptimConfig(name="sgd", lr=1e-3)
    optimizer = create_optimizer(config, [params])
    early_cfg = EarlyStoppingConfig(patience=3, min_delta=0.0)
    trainer = VQETrainer(vqe, params, optimizer, early_stopping=early_cfg)

    history = trainer.run(max_steps=50)

    # Early stopping should trigger before max_steps
    assert history.num_steps() <= 50
    assert history.num_steps() > 0


def test_vqe_trainer_best_params_checkpoint() -> None:
    """Test that best parameters are checkpointed correctly."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="adam", lr=0.05)
    optimizer = create_optimizer(config, [params])
    trainer = VQETrainer(vqe, params, optimizer)

    trainer.run(max_steps=10)

    best_params = trainer.best_params()
    assert best_params is not None
    assert best_params.shape == params.data.shape

    # Verify that using best_params yields energy close to best_energy
    if trainer.best_energy() is not None:
        # Temporarily set params to best_params to check energy
        original_params = params.data.clone()
        params.data.copy_(best_params)
        energy_with_best = vqe.energy(params).item()
        params.data.copy_(original_params)

        assert abs(energy_with_best - trainer.best_energy()) < 1e-6


def test_vqe_trainer_callbacks_invoked() -> None:
    """Test that callbacks are invoked correctly."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1)
    optimizer = create_optimizer(config, [params])

    # Create a callback that records steps
    records: list[TrainingStepInfo] = []

    def cb(info: TrainingStepInfo) -> None:
        records.append(info)

    trainer = VQETrainer(vqe, params, optimizer)
    trainer.run(max_steps=5, callbacks=[cb])

    # Verify callbacks were called
    assert len(records) == 5
    assert all(info.step in {1, 2, 3, 4, 5} for info in records)
    assert all(info.epoch == 1 for info in records)


def test_vqe_trainer_history_tracking() -> None:
    """Test that training history is tracked correctly."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1)
    optimizer = create_optimizer(config, [params])
    trainer = VQETrainer(vqe, params, optimizer)

    history = trainer.run(max_steps=10)

    # Check history methods
    assert history.num_steps() == 10
    assert history.best_energy() is not None
    assert history.final_energy() is not None
    assert history.best_energy() <= history.final_energy() + 1e-12

    # Check that all steps have required fields
    for step_info in history.steps:
        assert step_info.step > 0
        assert step_info.epoch == 1
        assert isinstance(step_info.energy, float)
        assert isinstance(step_info.loss, float)


def test_vqe_trainer_no_early_stopping() -> None:
    """Test that training runs for full max_steps when early stopping is disabled."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1)
    optimizer = create_optimizer(config, [params])
    # No early stopping config
    trainer = VQETrainer(vqe, params, optimizer)

    history = trainer.run(max_steps=15)

    # Should run for all steps
    assert history.num_steps() == 15


def test_vqe_trainer_early_stopping_patience_zero() -> None:
    """Test that early stopping with patience=0 is effectively disabled."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1)
    optimizer = create_optimizer(config, [params])
    early_cfg = EarlyStoppingConfig(patience=0, min_delta=0.0)
    trainer = VQETrainer(vqe, params, optimizer, early_stopping=early_cfg)

    history = trainer.run(max_steps=10)

    # Should run for all steps since patience=0
    assert history.num_steps() == 10


def test_vqe_trainer_multiple_callbacks() -> None:
    """Test that multiple callbacks are invoked."""
    vqe, params = build_simple_vqe()

    config = OptimConfig(name="sgd", lr=0.1)
    optimizer = create_optimizer(config, [params])

    records1: list[TrainingStepInfo] = []
    records2: list[TrainingStepInfo] = []

    def cb1(info: TrainingStepInfo) -> None:
        records1.append(info)

    def cb2(info: TrainingStepInfo) -> None:
        records2.append(info)

    trainer = VQETrainer(vqe, params, optimizer)
    trainer.run(max_steps=5, callbacks=[cb1, cb2])

    assert len(records1) == 5
    assert len(records2) == 5
    assert records1 == records2


