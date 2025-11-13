"""
Parameter-shift gradient engine for expectation values.

This module implements standard parameter-shift gradients for expectation values,
as described in basic quantum computing texts. The implementation uses the
textbook parameter-shift rule for single-parameter gates:

    dE/dθ = 0.5 * [E(θ + π/2) - E(θ - π/2)]

This is generic, non-proprietary logic that any quantum developer could implement
from basic references. It does not contain any HarmonicQ-specific tricks or
advanced optimizations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

import torch
from torch.autograd import Function

from qconduit.core.device import Device
from qconduit.layers.ansatzes import ParametricAnsatz
from qconduit.operators.pauli import PauliSum
from qconduit.operators.expectation import expectation_pauli_sum, expectation_pauli_sum_dm

if TYPE_CHECKING:
    from qconduit.noise import NoiseModel

# Type alias for Hamiltonian representations
HamiltonianLike = Union[torch.Tensor, PauliSum]


def evaluate_energy(
    ansatz: ParametricAnsatz,
    hamiltonian: HamiltonianLike,
    params: torch.Tensor,
    device: Device | None = None,
    noise_model: "NoiseModel | None" = None,
) -> torch.Tensor:
    """
    Compute the energy E(theta) = <psi(theta)| H |psi(theta)>.

    This uses the existing ansatz and Hamiltonian plumbing only:
    - If `hamiltonian` is a 1D real tensor, it is treated as a diagonal in the computational basis.
    - If `hamiltonian` is a PauliSum, it is evaluated with `expectation_pauli_sum` or
      `expectation_pauli_sum_dm` depending on whether noise is present.

    If `noise_model` is provided, the energy is computed using the noisy density matrix
    Tr(ρ_noisy H) instead of the pure state expectation.

    `params` is expected to be a 1D tensor of length equal to ansatz.num_parameters.

    Args:
        ansatz: Parametric ansatz that maps parameters to a statevector.
        hamiltonian: Either a diagonal 1D tensor or a PauliSum.
        params: 1D tensor of parameters.
        device: Optional Device; if None, ansatz.device is used.
        noise_model: Optional NoiseModel to apply to the state before computing energy.

    Returns:
        A scalar tensor (0-D) with real dtype containing the energy.

    Raises:
        ValueError: If params is not 1D, or if dimensions don't match.
    """
    # Validate params is 1D
    if params.dim() != 1:
        raise ValueError(
            f"params must be 1D tensor, got shape {params.shape} with {params.dim()} dimensions"
        )

    # Determine device
    if device is None:
        device = ansatz.device

    # Move params to device and ensure float dtype
    params = params.to(dtype=torch.float32, device=device.as_torch_device())

    # Build state using ansatz
    state = ansatz(params)

    # Handle noise if present
    if noise_model is None:
        # Noiseless path: use pure state expectations
        if isinstance(hamiltonian, PauliSum):
            # PauliSum path
            if hamiltonian.n_qubits() != ansatz.n_qubits:
                raise ValueError(
                    f"hamiltonian.n_qubits() = {hamiltonian.n_qubits()} does not "
                    f"match ansatz.n_qubits = {ansatz.n_qubits}"
                )
            energy = expectation_pauli_sum(state, hamiltonian)
        else:
            # Diagonal tensor path
            if hamiltonian.dim() != 1:
                raise ValueError(
                    f"Diagonal hamiltonian must be 1D tensor, got shape {hamiltonian.shape}"
                )
            if hamiltonian.shape[0] != state.shape[-1]:
                raise ValueError(
                    f"hamiltonian length {hamiltonian.shape[0]} does not match "
                    f"state dimension {state.shape[-1]}"
                )

            # Compute probabilities: |state|²
            probs = torch.abs(state) ** 2  # real

            # Move hamiltonian to correct device and dtype
            hamiltonian = hamiltonian.to(dtype=probs.dtype, device=probs.device)

            # Compute energy: sum over probabilities * diagonal elements
            energy = (probs * hamiltonian).sum(dim=-1)
    else:
        # Noisy path: use density matrix expectations
        from qconduit.backend.density_matrix import dm_from_statevector, measure_probs_dm

        n_qubits = ansatz.n_qubits
        rho = noise_model.apply_statevector(state, n_qubits=n_qubits)

        if isinstance(hamiltonian, PauliSum):
            # PauliSum path with density matrix
            if hamiltonian.n_qubits() != n_qubits:
                raise ValueError(
                    f"hamiltonian.n_qubits() = {hamiltonian.n_qubits()} does not "
                    f"match ansatz.n_qubits = {n_qubits}"
                )
            energy = expectation_pauli_sum_dm(rho, hamiltonian)
        else:
            # Diagonal tensor path with density matrix
            if hamiltonian.dim() != 1:
                raise ValueError(
                    f"Diagonal hamiltonian must be 1D tensor, got shape {hamiltonian.shape}"
                )
            if hamiltonian.shape[0] != 2**n_qubits:
                raise ValueError(
                    f"hamiltonian length {hamiltonian.shape[0]} does not match "
                    f"2**n_qubits = {2**n_qubits}"
                )

            # Get diagonal of density matrix
            diag = rho.diagonal(dim1=-2, dim2=-1).real

            # Move hamiltonian to correct device and dtype
            hamiltonian = hamiltonian.to(dtype=diag.dtype, device=diag.device)

            # Compute energy: sum over diagonal * hamiltonian elements
            energy = (diag * hamiltonian).sum(dim=-1)

    # Ensure result is scalar (0-D tensor) and real
    if energy.dim() > 0:
        energy = energy.squeeze()
    if energy.dim() > 0:
        # Still not scalar, take first element (shouldn't happen for 1D params)
        energy = energy[0]

    return energy.real


class ParamShiftEnergy(Function):
    """
    Custom autograd.Function implementing parameter-shift gradients
    for E(theta) = <psi(theta)| H |psi(theta)>.

    This uses the textbook rule for single-parameter gates:

        dE/dθ = 0.5 * (E(θ + s) - E(θ - s))

    with s = π/2 by default.

    It treats the ansatz and Hamiltonian as opaque objects; gradients
    are computed only with respect to `params`.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        params: torch.Tensor,
        ansatz: ParametricAnsatz,
        hamiltonian: HamiltonianLike,
        device: Device | None,
        shift: float,
        noise_model: "NoiseModel | None",
    ) -> torch.Tensor:
        """
        Forward pass: compute energy.

        Args:
            ctx: Context object for storing information needed in backward.
            params: 1D parameter tensor.
            ansatz: Parametric ansatz.
            hamiltonian: Hamiltonian (diagonal tensor or PauliSum).
            device: Optional Device.
            shift: Parameter-shift amount (typically π/2).
            noise_model: Optional NoiseModel.

        Returns:
            Scalar energy tensor.
        """
        # Compute energy
        energy = evaluate_energy(
            ansatz, hamiltonian, params, device=device, noise_model=noise_model
        )

        # Save context for backward
        ctx.ansatz = ansatz
        ctx.hamiltonian = hamiltonian
        ctx.device = device
        ctx.shift = shift
        ctx.noise_model = noise_model
        ctx.num_params = params.numel()

        # Save params for backward (detached, as we'll modify copies)
        ctx.save_for_backward(params.detach().clone())

        return energy

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """
        Backward pass: compute gradients using parameter-shift rule.

        Args:
            ctx: Context object from forward.
            grad_output: Upstream gradient (scalar).

        Returns:
            Tuple of gradients: (grad_params, None, None, None, None)
        """
        # Retrieve saved values
        ansatz = ctx.ansatz
        hamiltonian = ctx.hamiltonian
        device = ctx.device
        shift = ctx.shift
        noise_model = ctx.noise_model
        num_params = ctx.num_params

        # Retrieve saved params
        (params,) = ctx.saved_tensors

        # Get device for gradient tensor
        torch_device = device.as_torch_device() if device is not None else params.device

        # Allocate gradient tensor
        grad_params = torch.empty(
            num_params, dtype=torch.float32, device=torch_device
        )

        # Compute gradients using parameter-shift rule
        # Work under no_grad to avoid nesting autograd graphs
        with torch.no_grad():
            for i in range(num_params):
                # Create shifted parameter vectors
                params_plus = params.clone()
                params_minus = params.clone()
                params_plus[i] += shift
                params_minus[i] -= shift

                # Compute energies at shifted points
                e_plus = evaluate_energy(
                    ansatz, hamiltonian, params_plus, device=device, noise_model=noise_model
                )
                e_minus = evaluate_energy(
                    ansatz, hamiltonian, params_minus, device=device, noise_model=noise_model
                )

                # Apply parameter-shift rule
                grad_i = 0.5 * (e_plus - e_minus)
                grad_params[i] = grad_i.to(grad_params.dtype)

        # Apply chain rule: multiply by upstream gradient
        if grad_output.dim() != 0:
            grad_output = grad_output.squeeze()
        grad_params = grad_params * grad_output.to(grad_params.dtype)

        # Return gradients: (params, ansatz, hamiltonian, device, shift, noise_model)
        return grad_params, None, None, None, None, None


def param_shift_energy(
    ansatz: ParametricAnsatz,
    hamiltonian: HamiltonianLike,
    params: torch.Tensor,
    device: Device | None = None,
    shift: float = math.pi / 2.0,
    noise_model: "NoiseModel | None" = None,
) -> torch.Tensor:
    """
    Compute the energy E(theta) with a custom autograd path using
    the standard parameter-shift rule.

    This is a thin wrapper around ParamShiftEnergy.apply, and is
    fully differentiable w.r.t. `params`.

    Parameters
    ----------
    ansatz:
        Parametric ansatz that maps parameters to a statevector.
    hamiltonian:
        Either a diagonal 1D tensor or a PauliSum.
    params:
        1D tensor of parameters. Must have requires_grad=True if
        gradients are desired.
    device:
        Optional Device; if None, ansatz.device is used.
    shift:
        Parameter-shift amount (defaults to π/2).
    noise_model:
        Optional NoiseModel to apply before computing energy.

    Returns
    -------
    Scalar energy tensor that is differentiable w.r.t. params.

    Example
    -------
    >>> from qconduit.layers.ansatzes import HardwareEfficientAnsatz
    >>> from qconduit.operators.pauli import PauliTerm, PauliSum
    >>> ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
    >>> h = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
    >>> params = torch.tensor([0.3], requires_grad=True)
    >>> energy = param_shift_energy(ansatz, h, params)
    >>> energy.backward()
    >>> print(params.grad)  # Gradient computed via parameter-shift
    """
    return ParamShiftEnergy.apply(params, ansatz, hamiltonian, device, shift, noise_model)

