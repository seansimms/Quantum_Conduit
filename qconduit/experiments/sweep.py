"""Parameter sweep utilities for variational circuits and VQE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from ..algorithms.vqe import VQE


@dataclass(frozen=True)
class SweepResult1D:
    """
    Result container for 1D parameter sweeps.

    This dataclass holds the parameter values and corresponding objective values
    from a 1D sweep. All tensors are stored on CPU and detached by default,
    making them ready for plotting or analysis.

    Attributes:
        points: 1D tensor of shape (n_points,) containing parameter values.
        values: 1D tensor of shape (n_points,) containing objective values.
        metadata: Dictionary with free-form metadata (e.g., labels, units).

    Example:
        >>> result = SweepResult1D(
        ...     points=torch.linspace(0.0, 1.0, 10),
        ...     values=torch.zeros(10),
        ...     metadata={"param_name": "theta", "x_label": "Parameter"}
        ... )
        >>> assert result.points.shape == (10,)
        >>> assert result.values.shape == (10,)
    """

    points: torch.Tensor
    values: torch.Tensor
    metadata: Dict[str, str]

    def __post_init__(self) -> None:
        """Validate SweepResult1D invariants."""
        # Validate points
        if self.points.ndim != 1:
            raise ValueError(
                f"points must be 1D tensor, got shape {self.points.shape}"
            )

        # Validate values
        if self.values.ndim != 1:
            raise ValueError(
                f"values must be 1D tensor, got shape {self.values.shape}"
            )

        # Validate matching lengths
        if self.points.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"points and values must have the same length, "
                f"got {self.points.shape[0]} and {self.values.shape[0]}"
            )

        # Ensure metadata is a shallow copy (frozen dataclass protection)
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(self.metadata)}")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SweepResult2D:
    """
    Result container for 2D parameter sweeps.

    This dataclass holds the parameter grid and corresponding objective values
    from a 2D sweep. All tensors are stored on CPU and detached by default,
    making them ready for plotting or analysis.

    The values tensor follows a row/column convention: values[i, j] corresponds
    to x_points[i] and y_points[j].

    Attributes:
        x_points: 1D tensor of shape (n_x,) containing x-axis parameter values.
        y_points: 1D tensor of shape (n_y,) containing y-axis parameter values.
        values: 2D tensor of shape (n_x, n_y) containing objective values.
            values[i, j] corresponds to x_points[i] and y_points[j].
        metadata: Dictionary with free-form metadata (e.g., labels, units).

    Example:
        >>> result = SweepResult2D(
        ...     x_points=torch.linspace(0.0, 1.0, 5),
        ...     y_points=torch.linspace(0.0, 1.0, 3),
        ...     values=torch.zeros(5, 3),
        ...     metadata={"x_label": "theta", "y_label": "phi"}
        ... )
        >>> assert result.values.shape == (5, 3)
    """

    x_points: torch.Tensor
    y_points: torch.Tensor
    values: torch.Tensor
    metadata: Dict[str, str]

    def __post_init__(self) -> None:
        """Validate SweepResult2D invariants."""
        # Validate x_points
        if self.x_points.ndim != 1:
            raise ValueError(
                f"x_points must be 1D tensor, got shape {self.x_points.shape}"
            )

        # Validate y_points
        if self.y_points.ndim != 1:
            raise ValueError(
                f"y_points must be 1D tensor, got shape {self.y_points.shape}"
            )

        # Validate values
        if self.values.ndim != 2:
            raise ValueError(
                f"values must be 2D tensor, got shape {self.values.shape}"
            )

        # Validate matching shapes
        expected_shape = (self.x_points.shape[0], self.y_points.shape[0])
        if self.values.shape != expected_shape:
            raise ValueError(
                f"values must have shape (n_x, n_y) = {expected_shape}, "
                f"got {self.values.shape}"
            )

        # Ensure metadata is a shallow copy (frozen dataclass protection)
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(self.metadata)}")
        object.__setattr__(self, "metadata", dict(self.metadata))


def run_1d_sweep(
    objective: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    base_params: Optional[torch.Tensor] = None,
    index: int = 0,
    detach: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> SweepResult1D:
    """
    Run a 1D parameter sweep over an arbitrary scalar objective function.

    This is a generic utility that evaluates an objective function at each point
    in a 1D parameter grid. The objective must return a scalar tensor (0-dimensional).

    Args:
        objective: Function that takes a parameter tensor and returns a scalar
            tensor (0-dimensional). The function signature is:
            objective(params: torch.Tensor) -> torch.Tensor (scalar).
        points: 1D tensor of shape (n_points,) with parameter values to sweep.
            Can be on any device/dtype.
        base_params: Optional 1D tensor of shape (n_params,). If provided, the
            sweep varies base_params[index] while keeping other parameters fixed.
            If None, the objective is assumed to be a function of a single scalar
            parameter.
        index: Index into base_params to vary. Defaults to 0. Must be 0 if
            base_params is None.
        detach: If True (default), evaluate under torch.no_grad() and return
            detached CPU tensors. If False, allow autograd but still return
            detached clones in the result.
        metadata: Optional dictionary with metadata (e.g., "param_name",
            "x_label", "y_label", "units").

    Returns:
        SweepResult1D containing points, values, and metadata.

    Raises:
        ValueError: If points is not 1D or empty, if base_params is provided
            but index is invalid, or if objective returns a non-scalar tensor.

    Example:
        >>> def objective(params):
        ...     return torch.cos(params[0])
        >>> points = torch.linspace(0.0, 3.14159, 10)
        >>> result = run_1d_sweep(objective, points)
        >>> assert result.points.shape == (10,)
        >>> assert result.values.shape == (10,)
    """
    # Validate points
    if points.ndim != 1:
        raise ValueError(f"points must be 1D tensor, got shape {points.shape}")
    if points.numel() == 0:
        raise ValueError("points must be non-empty")

    # Handle base_params
    if base_params is None:
        if index != 0:
            raise ValueError(
                f"If base_params is None, index must be 0, got {index}"
            )
    else:
        if base_params.ndim != 1:
            raise ValueError(
                f"base_params must be 1D tensor, got shape {base_params.shape}"
            )
        if not (0 <= index < base_params.numel()):
            raise ValueError(
                f"index must be in [0, {base_params.numel()}), got {index}"
            )

    # Determine dtype for values (use float64 or points.dtype if floating)
    if torch.is_floating_point(points):
        values_dtype = points.dtype
    else:
        values_dtype = torch.float64

    # Allocate values tensor on CPU
    n_points = points.shape[0]
    values = torch.zeros(n_points, dtype=values_dtype, device=torch.device("cpu"))

    # Detach points to avoid side effects
    points = points.detach().clone()

    # Loop over points
    for i in range(n_points):
        point = points[i].item()

        # Construct params
        if base_params is None:
            # Single scalar parameter
            params = torch.tensor([point], dtype=points.dtype, device=points.device)
        else:
            # Copy base_params and set index
            params = base_params.clone()
            params[index] = point

        # Evaluate objective
        if detach:
            with torch.no_grad():
                val = objective(params)
        else:
            val = objective(params)

        # Validate scalar return
        if val.ndim != 0:
            raise ValueError(
                f"objective must return a scalar tensor (0-dimensional), "
                f"got shape {val.shape}"
            )

        # Store value (cast to float to avoid complex)
        values[i] = float(val.item())

    # Create result with detached CPU tensors
    result = SweepResult1D(
        points=points.detach().cpu(),
        values=values.detach().cpu(),
        metadata=metadata or {},
    )
    return result


def run_2d_sweep(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x_points: torch.Tensor,
    y_points: torch.Tensor,
    base_params: Optional[torch.Tensor] = None,
    x_index: int = 0,
    y_index: int = 1,
    detach: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> SweepResult2D:
    """
    Run a 2D parameter sweep over an arbitrary scalar objective function.

    This is a generic utility that evaluates an objective function at each point
    in a 2D parameter grid. The objective must return a scalar tensor (0-dimensional).

    Args:
        objective: Function that takes a parameter tensor and returns a scalar
            tensor (0-dimensional). The function signature is:
            objective(params: torch.Tensor) -> torch.Tensor (scalar).
        x_points: 1D tensor of shape (n_x,) with x-axis parameter values.
            Can be on any device/dtype.
        y_points: 1D tensor of shape (n_y,) with y-axis parameter values.
            Can be on any device/dtype.
        base_params: Optional 1D tensor of shape (n_params,). If provided, the
            sweep varies base_params[x_index] and base_params[y_index] while
            keeping other parameters fixed. If None, the objective is assumed
            to be a function of two scalar parameters.
        x_index: Index into base_params for x-axis. Defaults to 0.
        y_index: Index into base_params for y-axis. Defaults to 1.
        detach: If True (default), evaluate under torch.no_grad() and return
            detached CPU tensors. If False, allow autograd but still return
            detached clones in the result.
        metadata: Optional dictionary with metadata (e.g., "x_label", "y_label",
            "z_label", "units").

    Returns:
        SweepResult2D containing x_points, y_points, values, and metadata.
        values[i, j] corresponds to x_points[i] and y_points[j].

    Raises:
        ValueError: If x_points or y_points are not 1D or empty, if base_params
            is provided but indices are invalid, if x_index == y_index, or if
            objective returns a non-scalar tensor.

    Example:
        >>> def objective(params):
        ...     return params[0] + 2.0 * params[1]
        >>> x_points = torch.linspace(0.0, 1.0, 5)
        >>> y_points = torch.linspace(0.0, 1.0, 3)
        >>> result = run_2d_sweep(objective, x_points, y_points)
        >>> assert result.values.shape == (5, 3)
    """
    # Validate x_points
    if x_points.ndim != 1:
        raise ValueError(f"x_points must be 1D tensor, got shape {x_points.shape}")
    if x_points.numel() == 0:
        raise ValueError("x_points must be non-empty")

    # Validate y_points
    if y_points.ndim != 1:
        raise ValueError(f"y_points must be 1D tensor, got shape {y_points.shape}")
    if y_points.numel() == 0:
        raise ValueError("y_points must be non-empty")

    # Handle base_params
    if base_params is None:
        # Two-parameter function
        if x_index != 0 or y_index != 1:
            raise ValueError(
                "If base_params is None, x_index must be 0 and y_index must be 1, "
                f"got x_index={x_index}, y_index={y_index}"
            )
    else:
        if base_params.ndim != 1:
            raise ValueError(
                f"base_params must be 1D tensor, got shape {base_params.shape}"
            )
        n_params = base_params.numel()
        if not (0 <= x_index < n_params):
            raise ValueError(
                f"x_index must be in [0, {n_params}), got {x_index}"
            )
        if not (0 <= y_index < n_params):
            raise ValueError(
                f"y_index must be in [0, {n_params}), got {y_index}"
            )
        if x_index == y_index:
            raise ValueError(
                f"x_index and y_index must be different, got both {x_index}"
            )

    # Determine dtype for values (use float64 or x_points.dtype if floating)
    if torch.is_floating_point(x_points):
        values_dtype = x_points.dtype
    else:
        values_dtype = torch.float64

    # Allocate values tensor on CPU
    n_x = x_points.shape[0]
    n_y = y_points.shape[0]
    values = torch.zeros(
        (n_x, n_y), dtype=values_dtype, device=torch.device("cpu")
    )

    # Detach points to avoid side effects
    x_points = x_points.detach().clone()
    y_points = y_points.detach().clone()

    # Loop over grid
    for i in range(n_x):
        x_val = x_points[i].item()
        for j in range(n_y):
            y_val = y_points[j].item()

            # Construct params
            if base_params is None:
                # Two scalar parameters
                params = torch.tensor(
                    [x_val, y_val],
                    dtype=x_points.dtype,
                    device=x_points.device,
                )
            else:
                # Copy base_params and set indices
                params = base_params.clone()
                params[x_index] = x_val
                params[y_index] = y_val

            # Evaluate objective
            if detach:
                with torch.no_grad():
                    val = objective(params)
            else:
                val = objective(params)

            # Validate scalar return
            if val.ndim != 0:
                raise ValueError(
                    f"objective must return a scalar tensor (0-dimensional), "
                    f"got shape {val.shape}"
                )

            # Store value (cast to float to avoid complex)
            values[i, j] = float(val.item())

    # Create result with detached CPU tensors
    result = SweepResult2D(
        x_points=x_points.detach().cpu(),
        y_points=y_points.detach().cpu(),
        values=values.detach().cpu(),
        metadata=metadata or {},
    )
    return result


def sweep_vqe_1d(
    vqe: VQE,
    points: torch.Tensor,
    base_params: torch.Tensor,
    index: int = 0,
    detach: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> SweepResult1D:
    """
    Run a 1D parameter sweep over VQE energy.

    This is a convenience wrapper around run_1d_sweep that uses VQE.energy as
    the objective function. It validates that base_params has the correct length
    for the ansatz and adds VQE-specific metadata.

    Args:
        vqe: VQE instance with an ansatz and Hamiltonian.
        points: 1D tensor of shape (n_points,) with parameter values to sweep.
        base_params: 1D tensor of shape (num_parameters,) where num_parameters
            must match vqe.ansatz.num_parameters.
        index: Index into base_params to vary. Defaults to 0.
        detach: If True (default), evaluate under torch.no_grad() and return
            detached CPU tensors.
        metadata: Optional dictionary with additional metadata. VQE-specific
            metadata will be added automatically.

    Returns:
        SweepResult1D containing points, energy values, and metadata.

    Raises:
        ValueError: If base_params.numel() does not match ansatz.num_parameters,
            or if other validation fails (see run_1d_sweep).

    Example:
        >>> from qconduit.layers.ansatzes import HardwareEfficientAnsatz
        >>> from qconduit.operators import PauliTerm, PauliSum
        >>> ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=1)
        >>> hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        >>> vqe = VQE(ansatz, hamiltonian)
        >>> points = torch.linspace(0.0, 6.28318, 41)
        >>> base_params = torch.zeros(1)
        >>> result = sweep_vqe_1d(vqe, points, base_params, index=0)
        >>> assert result.points.shape == (41,)
        >>> assert result.values.shape == (41,)
    """
    # Validate base_params length
    if hasattr(vqe, "ansatz") and hasattr(vqe.ansatz, "num_parameters"):
        expected_num_params = vqe.ansatz.num_parameters
        if base_params.numel() != expected_num_params:
            raise ValueError(
                f"base_params must have length {expected_num_params} "
                f"(ansatz.num_parameters), got {base_params.numel()}"
            )

    # Define objective
    def objective(params: torch.Tensor) -> torch.Tensor:
        return vqe.energy(params)

    # Compose metadata
    metadata_dict = dict(metadata or {})
    metadata_dict["sweep_type"] = "vqe_1d"
    metadata_dict["param_index"] = str(index)

    # Add Hamiltonian info if available
    if hasattr(vqe, "hamiltonian_pauli") and vqe.hamiltonian_pauli is not None:
        metadata_dict["hamiltonian_type"] = "pauli_sum"
        metadata_dict["hamiltonian_terms"] = str(len(vqe.hamiltonian_pauli.terms))
    elif hasattr(vqe, "hamiltonian_diag") and vqe.hamiltonian_diag is not None:
        metadata_dict["hamiltonian_type"] = "diagonal"

    # Run sweep
    return run_1d_sweep(
        objective=objective,
        points=points,
        base_params=base_params,
        index=index,
        detach=detach,
        metadata=metadata_dict,
    )


def sweep_vqe_2d(
    vqe: VQE,
    x_points: torch.Tensor,
    y_points: torch.Tensor,
    base_params: torch.Tensor,
    x_index: int = 0,
    y_index: int = 1,
    detach: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> SweepResult2D:
    """
    Run a 2D parameter sweep over VQE energy.

    This is a convenience wrapper around run_2d_sweep that uses VQE.energy as
    the objective function. It validates that base_params has the correct length
    for the ansatz and adds VQE-specific metadata.

    Args:
        vqe: VQE instance with an ansatz and Hamiltonian.
        x_points: 1D tensor of shape (n_x,) with x-axis parameter values.
        y_points: 1D tensor of shape (n_y,) with y-axis parameter values.
        base_params: 1D tensor of shape (num_parameters,) where num_parameters
            must match vqe.ansatz.num_parameters.
        x_index: Index into base_params for x-axis. Defaults to 0.
        y_index: Index into base_params for y-axis. Defaults to 1.
        detach: If True (default), evaluate under torch.no_grad() and return
            detached CPU tensors.
        metadata: Optional dictionary with additional metadata. VQE-specific
            metadata will be added automatically.

    Returns:
        SweepResult2D containing x_points, y_points, energy values, and metadata.
        values[i, j] corresponds to x_points[i] and y_points[j].

    Raises:
        ValueError: If base_params.numel() does not match ansatz.num_parameters,
            if x_index == y_index, or if other validation fails (see run_2d_sweep).

    Example:
        >>> from qconduit.layers.ansatzes import HardwareEfficientAnsatz
        >>> from qconduit.operators import PauliTerm, PauliSum
        >>> ansatz = HardwareEfficientAnsatz(n_qubits=1, depth=2)
        >>> hamiltonian = PauliSum.from_terms([PauliTerm(1.0, ("Z",))])
        >>> vqe = VQE(ansatz, hamiltonian)
        >>> x_points = torch.linspace(0.0, 3.14159, 5)
        >>> y_points = torch.linspace(0.0, 2.0, 3)
        >>> base_params = torch.zeros(2)
        >>> result = sweep_vqe_2d(vqe, x_points, y_points, base_params, x_index=0, y_index=1)
        >>> assert result.values.shape == (5, 3)
    """
    # Validate base_params length
    if hasattr(vqe, "ansatz") and hasattr(vqe.ansatz, "num_parameters"):
        expected_num_params = vqe.ansatz.num_parameters
        if base_params.numel() != expected_num_params:
            raise ValueError(
                f"base_params must have length {expected_num_params} "
                f"(ansatz.num_parameters), got {base_params.numel()}"
            )

    # Validate indices
    if x_index == y_index:
        raise ValueError(
            f"x_index and y_index must be different, got both {x_index}"
        )

    # Define objective
    def objective(params: torch.Tensor) -> torch.Tensor:
        return vqe.energy(params)

    # Compose metadata
    metadata_dict = dict(metadata or {})
    metadata_dict["sweep_type"] = "vqe_2d"
    metadata_dict["param_index_x"] = str(x_index)
    metadata_dict["param_index_y"] = str(y_index)

    # Add Hamiltonian info if available
    if hasattr(vqe, "hamiltonian_pauli") and vqe.hamiltonian_pauli is not None:
        metadata_dict["hamiltonian_type"] = "pauli_sum"
        metadata_dict["hamiltonian_terms"] = str(len(vqe.hamiltonian_pauli.terms))
    elif hasattr(vqe, "hamiltonian_diag") and vqe.hamiltonian_diag is not None:
        metadata_dict["hamiltonian_type"] = "diagonal"

    # Run sweep
    return run_2d_sweep(
        objective=objective,
        x_points=x_points,
        y_points=y_points,
        base_params=base_params,
        x_index=x_index,
        y_index=y_index,
        detach=detach,
        metadata=metadata_dict,
    )

