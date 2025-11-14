"""Adiabatic schedule functions for discrete adiabatic evolution."""

from __future__ import annotations

from typing import Callable

import torch

from qconduit.core.device import default_device

# Type alias for schedule functions
ScheduleFn = Callable[[torch.Tensor], torch.Tensor]
"""
A ScheduleFn takes a 1D tensor t of times in [0, 1] and returns a 1D tensor
s(t) of the same shape with values in [0, 1].
"""


def linear_schedule(num_steps: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Construct a linear adiabatic schedule s_k = k / (num_steps - 1),
    for k = 0, ..., num_steps-1.

    Parameters
    ----------
    num_steps:
        Number of discrete steps in the schedule. Must be >= 2.
    device:
        Optional device on which to place the returned tensor.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_steps,) with values in [0, 1], monotonically
        increasing from 0.0 to 1.0 inclusive.

    Raises
    ------
    ValueError
        If num_steps < 2.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    if device is None:
        device = default_device().as_torch_device()

    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    schedule = indices / float(num_steps - 1)

    return schedule


def polynomial_schedule(
    num_steps: int,
    power: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Construct a monotone polynomial schedule s(t) = t^power on t in [0,1].

    The returned array is s_k = (k / (num_steps - 1))**power.

    Parameters
    ----------
    num_steps:
        Number of discrete steps in the schedule. Must be >= 2.
    power:
        Exponent p >= 1.0 for t^p. Larger p spends more time near s = 0.
    device:
        Optional device.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_steps,) with values in [0, 1], with s_0 = 0.0
        and s_{num_steps-1} = 1.0.

    Raises
    ------
    ValueError
        If num_steps < 2 or power < 1.0.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    if power < 1.0:
        raise ValueError("power must be >= 1.0.")

    if device is None:
        device = default_device().as_torch_device()

    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = indices / float(num_steps - 1)
    schedule = t.pow(power)

    return schedule


def sample_schedule(
    schedule_fn: ScheduleFn,
    num_steps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Sample an arbitrary schedule function ScheduleFn on a uniform grid
    t_k = k / (num_steps - 1), k=0,...,num_steps-1.

    Parameters
    ----------
    schedule_fn:
        Callable mapping a 1D tensor of t in [0,1] to a 1D tensor of s(t) in [0,1].
    num_steps:
        Number of discrete steps (>= 2).
    device:
        Optional device.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_steps,) with the schedule values.

    Raises
    ------
    ValueError
        If num_steps < 2, or if schedule_fn returns invalid output (wrong shape,
        values outside [0,1], or incorrect endpoints).
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    if device is None:
        device = default_device().as_torch_device()

    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = indices / float(num_steps - 1)

    s = schedule_fn(t)

    # Validate output shape
    if s.shape != (num_steps,):
        raise ValueError(
            f"schedule_fn must return a tensor of shape ({num_steps},), "
            f"but got shape {s.shape}."
        )

    # Validate values are in [0, 1]
    s_min = torch.min(s).item()
    s_max = torch.max(s).item()
    if s_min < -1e-6 or s_max > 1.0 + 1e-6:
        raise ValueError(
            f"schedule_fn must return values in [0, 1], "
            f"but got range [{s_min}, {s_max}]."
        )

    # Validate endpoints
    s_0 = s[0].item()
    s_end = s[-1].item()
    if abs(s_0) > 1e-6:
        raise ValueError(
            f"schedule_fn must return s(0) ≈ 0, but got s[0] = {s_0}."
        )
    if abs(s_end - 1.0) > 1e-6:
        raise ValueError(
            f"schedule_fn must return s(1) ≈ 1, but got s[-1] = {s_end}."
        )

    return s


__all__ = [
    "ScheduleFn",
    "linear_schedule",
    "polynomial_schedule",
    "sample_schedule",
]


