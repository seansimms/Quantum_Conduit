"""Parameter-shift rule for computing gradients of quantum expectation values."""

from __future__ import annotations

import math
from typing import Callable

import torch


def parameter_shift_gradients(
    evaluate_fn: Callable[[torch.Tensor], float],
    params: torch.Tensor,
    shift: float = math.pi / 2.0,
) -> torch.Tensor:
    """
    Compute gradients using the parameter-shift rule.

    This implements the textbook parameter-shift rule for gates of the form
    exp(-i θ G / 2) where G is a generator with eigenvalues ±r. For rotation
    gates Rx, Ry, Rz, the standard shift is π/2 and the gradient is computed as:

        ∂E/∂θᵢ = 0.5 * (E(θ + s·eᵢ) - E(θ - s·eᵢ))

    where s = shift (default π/2) and eᵢ is the unit vector in direction i.

    **Important assumptions:**
    - This method assumes the ansatz uses only parameterized gates of the form
      exp(-i θ G / 2) where G has eigenvalues ±r (e.g., Rx, Ry, Rz rotations).
    - Correctness is guaranteed only for Rx, Ry, Rz gates. Other parameterized
      gates may produce incorrect gradients.
    - The evaluation function must be deterministic (no randomness).

    Parameters
    ----------
    evaluate_fn:
        Function that takes a 1D parameter tensor (float dtype on CPU) and
        returns a Python float expectation value (or a scalar 0-dim torch.Tensor).
        Must be deterministic.
    params:
        1D parameter tensor of shape (num_parameters,) with float dtype.
        Should be on CPU for compatibility with evaluate_expectation_value.
    shift:
        Shift amount for the parameter-shift rule. Default is π/2, which is
        correct for standard rotation gates Rx, Ry, Rz.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_parameters,) containing the gradient vector
        with dtype torch.float64.

    Raises
    ------
    ValueError
        If params is not 1D or has unexpected shape.

    Notes
    -----
    This method requires O(d) evaluations of evaluate_fn where d is the number
    of parameters. For large ansätze, this is expected behavior (textbook
    parameter-shift complexity).
    """
    if params.ndim != 1:
        raise ValueError(f"params must be 1D, got shape {params.shape}")

    num_params = params.shape[0]
    gradients = []

    # Ensure params is on CPU and float64
    params_cpu = params.detach().cpu().to(dtype=torch.float64)

    # Compute gradient for each parameter
    for i in range(num_params):
        # Create shifted parameter vectors
        params_plus = params_cpu.clone()
        params_plus[i] += shift

        params_minus = params_cpu.clone()
        params_minus[i] -= shift

        # Evaluate expectation values at shifted parameters
        v_plus = evaluate_fn(params_plus)
        v_minus = evaluate_fn(params_minus)

        # Convert to float if needed
        if isinstance(v_plus, torch.Tensor):
            v_plus = float(v_plus.item())
        if isinstance(v_minus, torch.Tensor):
            v_minus = float(v_minus.item())

        # Compute gradient element: 0.5 * (v_plus - v_minus)
        grad_i = 0.5 * (v_plus - v_minus)
        gradients.append(grad_i)

    # Return as 1D tensor
    return torch.tensor(gradients, dtype=torch.float64)


def batched_parameter_shift_gradients(
    evaluate_fn_batch: Callable[[torch.Tensor], float],
    params_batch: torch.Tensor,
    shift: float = math.pi / 2.0,
) -> torch.Tensor:
    """
    Compute parameter-shift gradients for a batch of parameter vectors.

    This is a convenience wrapper that computes gradients for each parameter
    vector in a batch by calling parameter_shift_gradients in a loop.

    Parameters
    ----------
    evaluate_fn_batch:
        Function that takes a 1D parameter tensor and returns a scalar float.
        Same interface as evaluate_fn in parameter_shift_gradients.
    params_batch:
        2D tensor of shape (batch_size, num_parameters) containing parameter
        vectors.
    shift:
        Shift amount for the parameter-shift rule. Default is π/2.

    Returns
    -------
    torch.Tensor
        2D tensor of shape (batch_size, num_parameters) containing gradient
        vectors for each parameter vector in the batch.

    Raises
    ------
    ValueError
        If params_batch is not 2D.
    """
    if params_batch.ndim != 2:
        raise ValueError(f"params_batch must be 2D, got shape {params_batch.shape}")

    batch_size = params_batch.shape[0]
    gradients_list = []

    # Compute gradients for each parameter vector in the batch
    for b in range(batch_size):
        params_single = params_batch[b]
        grad = parameter_shift_gradients(evaluate_fn_batch, params_single, shift=shift)
        gradients_list.append(grad)

    # Stack into batch tensor
    return torch.stack(gradients_list, dim=0)


__all__ = [
    "parameter_shift_gradients",
    "batched_parameter_shift_gradients",
]

