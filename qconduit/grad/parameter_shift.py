"""
Generic parameter-shift gradient engine for variational quantum algorithms.

This module implements the standard, textbook parameter-shift rule for computing
analytic gradients of scalar objective functions. The implementation follows
Nielsen & Chuang / Schuld & Killoran material: no proprietary tricks, no heuristic
optimizers, just clean plumbing.

For gates whose generator has eigenvalues ±1/2 (e.g. RX, RY, RZ), the standard
parameter-shift rule is:

    ∂f/∂θ = (1/2) [f(θ + π/2) − f(θ − π/2)]

This is exactly what shift=π/2 and prefactor=0.5 encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from ..algorithms.vqe import VQE


# Type alias for scalar objective functions
ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]
"""
An ObjectiveFn accepts a 1D parameter vector params (torch.Tensor with ndim == 1)
and returns a scalar torch.Tensor (0D). This will typically be something like a
VQE energy or an expectation value.
"""


@dataclass(frozen=True)
class ParameterShiftRule:
    """
    Parameter-shift rule configuration.

    For gates whose generator has eigenvalues ±1/2 (e.g. RX, RY, RZ), the standard
    parameter-shift rule is:

        ∂f/∂θ = (1/2) [f(θ + π/2) − f(θ − π/2)]

    This is exactly what shift=π/2 and prefactor=0.5 encode.

    Args:
        shift: The shift s in the parameter-shift formula. Default: π/2.
        prefactor: The overall factor applied to (f(θ + s) - f(θ - s)).
            Default: 0.5 (for standard rotations e^{-i θ σ / 2} with eigenvalues ±1).
    """

    shift: float
    prefactor: float

    def __post_init__(self) -> None:
        """Validate ParameterShiftRule invariants."""
        if not isinstance(self.shift, (int, float)):
            raise TypeError(f"shift must be a number, got {type(self.shift)}")
        if not isinstance(self.prefactor, (int, float)):
            raise TypeError(f"prefactor must be a number, got {type(self.prefactor)}")
        if not math.isfinite(self.shift):
            raise ValueError(f"shift must be finite, got {self.shift}")
        if not math.isfinite(self.prefactor):
            raise ValueError(f"prefactor must be finite, got {self.prefactor}")


def default_parameter_shift_rule() -> ParameterShiftRule:
    """
    Return the standard parameter-shift rule.

    Returns:
        ParameterShiftRule with shift=π/2 and prefactor=0.5.
    """
    return ParameterShiftRule(shift=float(math.pi / 2), prefactor=0.5)


def parameter_shift_single(
    objective: ObjectiveFn,
    params: torch.Tensor,
    index: int,
    rule: Optional[ParameterShiftRule] = None,
    detach: bool = True,
) -> float:
    """
    Compute the gradient of a scalar objective function with respect to a single parameter
    using the parameter-shift rule.

    This function implements the standard parameter-shift rule:

        ∂f/∂θ_i = prefactor * [f(θ + s*e_i) - f(θ - s*e_i)]

    where e_i is the unit vector in the i-th direction, s is the shift, and prefactor
    is the rule's prefactor.

    Args:
        objective: Callable taking a 1D tensor of parameters and returning a scalar tensor.
        params: 1D tensor of length n_params.
        index: Integer parameter index 0 <= index < n_params whose derivative we want.
        rule: Parameter-shift rule. If None, uses the default rule (shift=π/2, prefactor=0.5).
        detach: If True (default), evaluations are done under torch.no_grad().
            If False, you still return a Python float, but do not wrap the evaluations
            in no_grad. (This is mainly for consistency; in practice parameter-shift is
            used as a non-autograd gradient, so tests can focus on detach=True.)

    Returns:
        Gradient value as a Python float.

    Raises:
        ValueError: If params is not 1D, index is out of range, or objective returns
            a non-scalar tensor.
    """
    # Validation
    if params.ndim != 1:
        raise ValueError(f"params must be a 1D tensor, got shape {params.shape} with ndim={params.ndim}")

    n_params = params.numel()
    if not (0 <= index < n_params):
        raise ValueError(f"index {index} out of range for params with {n_params} elements")

    # Set default rule if needed
    if rule is None:
        rule = default_parameter_shift_rule()

    # Build shifted parameter vectors
    theta_plus = params.clone()
    theta_minus = params.clone()
    theta_plus[index] = theta_plus[index] + rule.shift
    theta_minus[index] = theta_minus[index] - rule.shift

    # Evaluate objective
    if detach:
        with torch.no_grad():
            value_plus = objective(theta_plus)
            value_minus = objective(theta_minus)
    else:
        value_plus = objective(theta_plus)
        value_minus = objective(theta_minus)

    # Validate scalar outputs
    if value_plus.ndim != 0:
        raise ValueError(
            f"objective must return a scalar tensor (0D), got shape {value_plus.shape} with ndim={value_plus.ndim}"
        )
    if value_minus.ndim != 0:
        raise ValueError(
            f"objective must return a scalar tensor (0D), got shape {value_minus.shape} with ndim={value_minus.ndim}"
        )

    # Compute gradient
    f_plus = float(value_plus.item())
    f_minus = float(value_minus.item())
    grad = rule.prefactor * (f_plus - f_minus)

    return grad


def parameter_shift_gradient(
    objective: ObjectiveFn,
    params: torch.Tensor,
    indices: Optional[Sequence[int]] = None,
    rule: Optional[ParameterShiftRule] = None,
    detach: bool = True,
) -> torch.Tensor:
    """
    Compute the gradient of a scalar objective function with respect to parameters
    using the parameter-shift rule.

    This function computes gradients for a subset or all parameters. The returned
    gradient tensor has length equal to len(indices), with entries aligned in the
    same order as indices.

    Args:
        objective: Callable taking a 1D tensor of parameters and returning a scalar tensor.
        params: 1D tensor of length n_params.
        indices: If None, compute gradient for all parameters: indices = range(n_params).
            If provided, must be a non-empty sequence of integers; each index must be
            in [0, n_params). Duplicates are allowed and will result in the same
            gradient being computed multiple times.
        rule: Parameter shift rule, or default if None.
        detach: Passed through to parameter_shift_single.

    Returns:
        1D tensor of dtype float64 (CPU) containing gradients. Length is len(indices),
        with gradient[i] corresponding to the derivative w.r.t. params[indices[i]].

    Raises:
        ValueError: If params is not 1D or empty, indices contains invalid values,
            or indices is empty when provided.
    """
    # Validation
    if params.ndim != 1:
        raise ValueError(f"params must be a 1D tensor, got shape {params.shape} with ndim={params.ndim}")
    if params.numel() == 0:
        raise ValueError("params must be non-empty")

    n_params = params.numel()

    # Process indices
    if indices is None:
        indices_list = list(range(n_params))
    else:
        indices_list = list(indices)
        if len(indices_list) == 0:
            raise ValueError("indices must be non-empty when provided")

        # Validate all indices are integers and in range
        for idx in indices_list:
            if not isinstance(idx, int):
                raise TypeError(f"All indices must be integers, got {type(idx)}")
            if not (0 <= idx < n_params):
                raise ValueError(f"Index {idx} out of range for params with {n_params} elements")

    # Set default rule if needed
    if rule is None:
        rule = default_parameter_shift_rule()

    # Allocate gradient vector
    grad = torch.empty(len(indices_list), dtype=torch.float64)

    # Compute gradients
    for k, idx in enumerate(indices_list):
        grad[k] = parameter_shift_single(objective, params, index=idx, rule=rule, detach=detach)

    return grad


def autograd_gradient(
    objective: ObjectiveFn,
    params: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Compute the gradient of a scalar objective function using PyTorch's autograd.

    This is a thin convenience wrapper that uses PyTorch's autograd for users who
    are fully in the differentiable regime. It assumes objective is differentiable
    and uses standard autograd.

    Note: This is conceptually separate from parameter-shift (which is typically
    used for hardware-compatible gradients or sampling-based objectives).

    Args:
        objective: Callable taking a 1D tensor of parameters and returning a scalar tensor.
        params: 1D parameter tensor.
        create_graph: If True, allows computing higher-order derivatives.

    Returns:
        1D tensor containing gradients, same shape as params.

    Raises:
        ValueError: If params is not 1D.
        RuntimeError: If autograd did not produce gradients for params.
    """
    # Validation
    if params.ndim != 1:
        raise ValueError(f"params must be a 1D tensor, got shape {params.shape} with ndim={params.ndim}")

    # Clone parameters and enable gradients
    params_local = params.clone().detach().requires_grad_(True)

    # Evaluate objective
    value = objective(params_local)

    # Validate scalar output
    if value.ndim != 0:
        raise ValueError(
            f"objective must return a scalar tensor (0D), got shape {value.shape} with ndim={value.ndim}"
        )

    # Compute gradient
    value.backward(create_graph=create_graph)

    # Retrieve gradient
    grad = params_local.grad
    if grad is None:
        raise RuntimeError("Autograd did not produce gradients for params.")

    return grad.detach()


def _vqe_energy_objective(vqe: "VQE") -> ObjectiveFn:
    """
    Create an objective function from a VQE instance.

    This helper function creates a closure that wraps vqe.energy() for use with
    the generic parameter-shift gradient functions.

    Args:
        vqe: VQE instance.

    Returns:
        Objective function that accepts a 1D tensor of parameters and returns
        a scalar tensor (the VQE energy).

    Note:
        This assumes vqe.energy accepts a 1D tensor of parameters and returns
        a scalar tensor.
    """
    def objective(params: torch.Tensor) -> torch.Tensor:
        return vqe.energy(params)

    return objective


def vqe_parameter_shift_gradient(
    vqe: "VQE",
    params: torch.Tensor,
    indices: Optional[Sequence[int]] = None,
    rule: Optional[ParameterShiftRule] = None,
    detach: bool = True,
) -> torch.Tensor:
    """
    Compute analytic gradients of VQE energy using parameter-shift.

    This function computes the gradient of the VQE energy with respect to ansatz
    parameters using the parameter-shift rule. It assumes the ansatz is built from
    rotation gates with standard generators (e.g. RX/RY/RZ-style).

    Args:
        vqe: VQE instance.
        params: 1D parameter tensor. Length must match ansatz.num_parameters.
        indices: If None, compute gradient for all parameters. If provided, must be
            a non-empty sequence of integers; each index must be in [0, n_params).
        rule: Parameter shift rule, or default if None.
        detach: Passed through to parameter_shift_gradient.

    Returns:
        1D tensor containing gradients. Length is len(indices) if indices is provided,
        otherwise ansatz.num_parameters.

    Raises:
        ValueError: If params length does not match ansatz.num_parameters, or if
            indices contains invalid values.
    """
    # Validate parameter length matches ansatz
    if hasattr(vqe.ansatz, "num_parameters"):
        expected_n_params = vqe.ansatz.num_parameters
        if params.numel() != expected_n_params:
            raise ValueError(
                f"Length of params ({params.numel()}) does not match "
                f"ansatz.num_parameters ({expected_n_params})."
            )

    # Construct objective
    objective = _vqe_energy_objective(vqe)

    # Compute gradient
    return parameter_shift_gradient(objective, params, indices=indices, rule=rule, detach=detach)

