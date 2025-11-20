"""PyTorch nn.Module integration for quantum variational circuits."""

from __future__ import annotations

import math
import weakref
from typing import Literal, Optional

import torch
import torch.nn as nn

from qconduit.operators import PauliSum
from qconduit.torch.param_shift import parameter_shift_gradients
from qconduit.torch.utils import validate_params_shape
from qconduit.variational.ansatz import VariationalAnsatz
from qconduit.variational.vqe import _vqe_energy_autograd, evaluate_expectation_value

if torch.__version__ < "1.0":
    raise RuntimeError("qconduit.torch requires PyTorch >= 1.0")


class ParameterShiftFunction(torch.autograd.Function):
    """
    Custom autograd Function for parameter-shift gradient computation.

    This Function wraps the forward evaluation and implements backward using
    the parameter-shift rule. It integrates cleanly with torch.optim optimizers.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        module_ref: weakref.ref,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: evaluate expectation value.

        Parameters
        ----------
        ctx:
            Context object for storing information needed in backward.
        module_ref:
            Weak reference to the QuantumModule instance (to avoid circular
            references and pickling issues).
        params:
            Parameter tensor with shape (num_parameters,).

        Returns
        -------
        torch.Tensor
            Scalar tensor (0-dim) with the expectation value.
        """
        module = module_ref()
        if module is None:
            raise RuntimeError("QuantumModule was garbage collected")

        # Get ansatz and hamiltonian from module
        ansatz = module._ansatz
        hamiltonian = module._hamiltonian

        # Evaluate expectation value using detached CPU params
        params_cpu = params.detach().cpu().to(dtype=torch.float64)

        # Create evaluation function closure
        def evaluate_fn(p: torch.Tensor) -> float:
            return evaluate_expectation_value(
                ansatz=ansatz,
                params=p,
                hamiltonian=hamiltonian,
                device=None,  # Use default device
            )

        # Evaluate expectation
        expectation = evaluate_fn(params_cpu)

        # Store for backward: save params and module metadata
        ctx.save_for_backward(params_cpu)
        ctx.module_ref = module_ref
        ctx.evaluate_fn = evaluate_fn

        # Return scalar tensor with requires_grad=True
        # The tensor must be on the same device as params for autograd
        result = torch.tensor(
            expectation, dtype=torch.float64, device=params.device, requires_grad=True
        )
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Backward pass: compute gradients using parameter-shift rule.

        Parameters
        ----------
        ctx:
            Context object containing saved tensors and metadata.
        grad_output:
            Gradient of the loss with respect to the forward output (scalar).

        Returns
        -------
        tuple
            Gradients with respect to module_ref (None) and params (tensor).
        """
        # Retrieve saved information
        params_cpu, = ctx.saved_tensors
        evaluate_fn = ctx.evaluate_fn

        # Compute parameter-shift gradients
        grad_vector = parameter_shift_gradients(
            evaluate_fn=evaluate_fn,
            params=params_cpu,
            shift=math.pi / 2.0,
        )

        # Scale by grad_output (chain rule)
        # grad_output is a scalar, grad_vector is 1D
        grad_params = grad_output * grad_vector

        # Return gradients: None for module_ref, grad_params for params
        return None, grad_params


class QuantumModule(nn.Module):
    """
    PyTorch nn.Module wrapper for variational quantum circuits.

    This module wraps a VariationalAnsatz and a PauliSum Hamiltonian, providing
    a PyTorch-compatible interface for computing expectation values and gradients.

    The module supports two gradient computation modes:
    - "autograd": Uses PyTorch's automatic differentiation if the backend
      supports it. Falls back to parameter-shift if not available.
    - "parameter_shift": Uses the deterministic parameter-shift rule for
      computing gradients. This is guaranteed to work for all ansätze using
      Rx, Ry, Rz rotation gates.

    Parameters
    ----------
    ansatz:
        Variational ansatz implementing the VariationalAnsatz protocol.
    hamiltonian:
        PauliSum representing the Hamiltonian to compute expectation values for.
    init_params:
        Optional initial parameter values. If None, parameters are initialized
        from a small normal distribution (std=0.01) with deterministic seed 0.
    gradient_method:
        Gradient computation method: "autograd" or "parameter_shift".
        Default is "parameter_shift".
    device:
        Optional PyTorch device. Parameters are stored on this device, but
        evaluation is performed on CPU (as required by evaluate_expectation_value).

    Example
    -------
    >>> import torch
    >>> from qconduit.variational import HardwareEfficientAnsatz
    >>> from qconduit.operators import PauliTerm, PauliSum
    >>> ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=1)
    >>> H = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
    >>> module = QuantumModule(ansatz, H, gradient_method="parameter_shift")
    >>> optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    >>> loss = module()
    >>> loss.backward()
    >>> optimizer.step()
    """

    def __init__(
        self,
        ansatz: VariationalAnsatz,
        hamiltonian: PauliSum,
        init_params: Optional[torch.Tensor] = None,
        gradient_method: Literal["autograd", "parameter_shift"] = "parameter_shift",
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize QuantumModule."""
        super().__init__()

        # Validate inputs
        if gradient_method not in ("autograd", "parameter_shift"):
            raise ValueError(
                f"gradient_method must be 'autograd' or 'parameter_shift', "
                f"got {gradient_method!r}"
            )

        # Store ansatz and hamiltonian
        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._gradient_method = gradient_method

        # Determine device (default to CPU)
        if device is None:
            self._device = torch.device("cpu")
        else:
            self._device = device

        # Validate ansatz and hamiltonian compatibility
        if hamiltonian.n_qubits() != 0 and hamiltonian.n_qubits() != ansatz.num_qubits:
            raise ValueError(
                f"hamiltonian.n_qubits() = {hamiltonian.n_qubits()} does not match "
                f"ansatz.num_qubits = {ansatz.num_qubits}"
            )

        # Initialize parameters
        num_params = ansatz.num_parameters
        if init_params is not None:
            validate_params_shape(init_params, num_params)
            params_data = init_params.detach().clone().to(
                dtype=torch.float64, device=self._device
            )
        else:
            # Initialize from small normal distribution with deterministic seed
            generator = torch.Generator(device=self._device)
            generator.manual_seed(0)
            params_data = torch.normal(
                mean=0.0,
                std=0.01,
                size=(num_params,),
                generator=generator,
                dtype=torch.float64,
                device=self._device,
            )

        # Create nn.Parameter
        self.params = nn.Parameter(params_data)

    def forward(self) -> torch.Tensor:
        """
        Compute expectation value for current parameters.

        Returns
        -------
        torch.Tensor
            0-dimensional scalar tensor with the expectation value ⟨H⟩.
        """
        if self._gradient_method == "autograd":
            return self._forward_autograd()
        else:
            return self._forward_parameter_shift()

    def _forward_autograd(self) -> torch.Tensor:
        """
        Forward pass using autograd mode.

        Returns
        -------
        torch.Tensor
            Scalar tensor with requires_grad=True if autograd is supported.

        Raises
        ------
        RuntimeError
            If autograd is not supported by the backend.
        """
        # Ensure params have requires_grad=True
        params = self.params.requires_grad_(True)

        # Try to use autograd-compatible evaluation
        # Use CPU device for evaluation (as required by evaluate_expectation_value)
        device_cpu = torch.device("cpu")
        try:
            energy = _vqe_energy_autograd(
                params=params.to(device=device_cpu),
                ansatz=self._ansatz,
                hamiltonian=self._hamiltonian,
                device=device_cpu,
            )

            # Check if result has requires_grad (autograd is working)
            if not isinstance(energy, torch.Tensor) or not energy.requires_grad:
                raise RuntimeError(
                    "Autograd mode requested but evaluate_expectation_value does not "
                    "support autograd. Use gradient_method='parameter_shift' instead."
                )

            # Move result back to module's device
            return energy.to(device=self.params.device)

        except Exception as e:
            # If autograd fails, raise clear error
            raise RuntimeError(
                f"Autograd mode failed: {e}. "
                "Use gradient_method='parameter_shift' instead."
            ) from e

    def _forward_parameter_shift(self) -> torch.Tensor:
        """
        Forward pass using parameter-shift mode.

        Returns
        -------
        torch.Tensor
            Scalar tensor with gradients computed via parameter-shift rule.
        """
        # Create weak reference to self to avoid circular references
        self_ref = weakref.ref(self)

        # Use ParameterShiftFunction for autograd integration
        return ParameterShiftFunction.apply(self_ref, self.params)

    def get_parameters(self) -> torch.Tensor:
        """
        Get current parameter values as a detached CPU tensor.

        Returns
        -------
        torch.Tensor
            1D tensor of shape (num_parameters,) with dtype torch.float64.
        """
        return self.params.detach().cpu().clone()

    def set_parameters(self, new_params: torch.Tensor) -> None:
        """
        Set parameter values.

        Parameters
        ----------
        new_params:
            1D tensor with shape (num_parameters,). Will be converted to
            float64 and moved to the module's device.

        Raises
        ------
        ValueError
            If new_params has incorrect shape.
        """
        validate_params_shape(new_params, self._ansatz.num_parameters)
        with torch.no_grad():
            self.params.data = new_params.detach().clone().to(
                dtype=torch.float64, device=self._device
            )

    def to(self, device: Optional[torch.device] = None, **kwargs) -> "QuantumModule":
        """
        Move module to a device.

        This overrides nn.Module.to() to ensure parameters are moved correctly.
        Note: evaluation is still performed on CPU as required by
        evaluate_expectation_value.

        Parameters
        ----------
        device:
            Target device.
        **kwargs:
            Additional arguments passed to nn.Module.to().

        Returns
        -------
        QuantumModule
            Self (for method chaining).
        """
        if device is not None:
            self._device = device
        result = super().to(device=device, **kwargs)
        return result


__all__ = ["QuantumModule"]

