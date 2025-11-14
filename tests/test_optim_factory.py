"""Tests for optimizer factory."""

from __future__ import annotations

import pytest
import torch

from qconduit.optim import OptimConfig, create_optimizer


def test_create_sgd_optimizer() -> None:
    """Test creation of SGD optimizer."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="sgd", lr=0.1, momentum=0.9, weight_decay=0.01)
    opt = create_optimizer(config, [p])
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["momentum"] == 0.9
    assert opt.defaults["weight_decay"] == 0.01


def test_create_adam_optimizer() -> None:
    """Test creation of Adam optimizer."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="adam", lr=0.01, betas=(0.8, 0.9))
    opt = create_optimizer(config, [p])
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 0.01
    assert opt.defaults["betas"] == (0.8, 0.9)


def test_create_adam_optimizer_default_betas() -> None:
    """Test creation of Adam optimizer with default betas."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="adam", lr=0.01)
    opt = create_optimizer(config, [p])
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 0.01
    assert opt.defaults["betas"] == (0.9, 0.999)


def test_create_lbfgs_optimizer() -> None:
    """Test creation of LBFGS optimizer."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(
        name="lbfgs", lr=0.5, max_iter=10, history_size=20, line_search_fn="strong_wolfe"
    )
    opt = create_optimizer(config, [p])
    assert isinstance(opt, torch.optim.LBFGS)
    assert opt.defaults["lr"] == 0.5
    assert opt.defaults["max_iter"] == 10
    assert opt.defaults["history_size"] == 20
    assert opt.defaults["line_search_fn"] == "strong_wolfe"


def test_create_optimizer_invalid_name_raises() -> None:
    """Test that invalid optimizer name raises ValueError."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="unknown", lr=0.1)
    with pytest.raises(ValueError, match="Unsupported optimizer name"):
        create_optimizer(config, [p])


def test_create_optimizer_invalid_lr_raises() -> None:
    """Test that invalid learning rate raises ValueError."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="sgd", lr=0.0)
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        create_optimizer(config, [p])


def test_create_optimizer_negative_lr_raises() -> None:
    """Test that negative learning rate raises ValueError."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="sgd", lr=-0.1)
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        create_optimizer(config, [p])


def test_create_optimizer_case_insensitive() -> None:
    """Test that optimizer name is case-insensitive."""
    p = torch.nn.Parameter(torch.tensor(0.0))
    config = OptimConfig(name="SGD", lr=0.1)
    opt = create_optimizer(config, [p])
    assert isinstance(opt, torch.optim.SGD)


