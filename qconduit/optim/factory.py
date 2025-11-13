"""Factory for creating PyTorch optimizers from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.optim as torch_optim
from torch.optim import Optimizer


@dataclass(frozen=True)
class OptimConfig:
    """
    Configuration for creating a PyTorch optimizer.

    This is a generic wrapper around standard torch.optim optimizers. Unsupported
    fields are simply ignored by optimizers that do not use them.

    Args:
        name: Optimizer name. Supported values: "sgd", "adam", "lbfgs".
        lr: Learning rate. Must be positive.
        weight_decay: Weight decay (L2 regularization) coefficient. Defaults to 0.0.
        momentum: Momentum factor for SGD. Defaults to 0.0.
        betas: Beta parameters for Adam-like optimizers. Defaults to None, which
            uses (0.9, 0.999) for Adam.
        max_iter: Maximum number of iterations per optimization step for LBFGS.
            Defaults to 20.
        history_size: History size for LBFGS. Defaults to 100.
        line_search_fn: Line search function for LBFGS. Can be "strong_wolfe" or
            None. Defaults to None.
    """

    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.0
    betas: tuple[float, float] | None = None
    max_iter: int = 20
    history_size: int = 100
    line_search_fn: str | None = None


def create_optimizer(
    config: OptimConfig, params: Iterable[torch.nn.Parameter]
) -> Optimizer:
    """
    Create a PyTorch optimizer from a configuration.

    This function constructs a standard torch.optim optimizer based on the
    provided configuration. Supported optimizer types are SGD, Adam, and LBFGS.

    Args:
        config: Optimizer configuration.
        params: Iterable of parameters to optimize.

    Returns:
        A torch.optim.Optimizer instance configured according to the config.

    Raises:
        ValueError: If the optimizer name is not supported or if the learning
            rate is not positive.
    """
    if config.lr <= 0.0:
        raise ValueError("Learning rate must be positive.")

    name_lower = config.name.lower()

    if name_lower == "sgd":
        return torch_optim.SGD(
            params=params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif name_lower == "adam":
        betas = config.betas if config.betas is not None else (0.9, 0.999)
        return torch_optim.Adam(
            params=params,
            lr=config.lr,
            betas=betas,
            weight_decay=config.weight_decay,
        )
    elif name_lower == "lbfgs":
        # LBFGS does not support weight_decay
        return torch_optim.LBFGS(
            params=params,
            lr=config.lr,
            max_iter=config.max_iter,
            history_size=config.history_size,
            line_search_fn=config.line_search_fn,
        )
    else:
        supported = ["sgd", "adam", "lbfgs"]
        raise ValueError(
            f"Unsupported optimizer name '{config.name}'. "
            f"Supported names: {supported}"
        )

