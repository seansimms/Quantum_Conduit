"""Utility functions for PyTorch integration with qconduit."""

from __future__ import annotations

from typing import Optional

import torch

from qconduit.core.device import default_device


def infer_device(device: Optional[torch.device]) -> torch.device:
    """
    Infer the PyTorch device to use.

    If a device is provided, return it. Otherwise, return the default device
    from qconduit's device abstraction.

    Parameters
    ----------
    device:
        Optional PyTorch device. If None, uses default_device().

    Returns
    -------
    torch.device
        The device to use for computation.
    """
    if device is not None:
        return device
    qdevice = default_device()
    return qdevice.as_torch_device()


def as_float_tensor(
    t: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a tensor to float64 dtype on the specified device.

    Parameters
    ----------
    t:
        Input tensor.
    device:
        Optional device. If None, uses infer_device().

    Returns
    -------
    torch.Tensor
        Tensor with dtype torch.float64 on the specified device.
    """
    target_device = infer_device(device)
    return t.to(device=target_device, dtype=torch.float64)


def as_complex_tensor(
    t: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a tensor to complex128 dtype on the specified device.

    Parameters
    ----------
    t:
        Input tensor.
    device:
        Optional device. If None, uses infer_device().

    Returns
    -------
    torch.Tensor
        Tensor with dtype torch.complex128 on the specified device.
    """
    target_device = infer_device(device)
    return t.to(device=target_device, dtype=torch.complex128)


def validate_params_shape(params: torch.Tensor, expected_len: int) -> None:
    """
    Validate that a parameter tensor has the expected shape.

    Parameters
    ----------
    params:
        Parameter tensor to validate.
    expected_len:
        Expected length of the 1D parameter vector.

    Raises
    ------
    ValueError
        If params is not 1D or has incorrect length.
    """
    if params.ndim != 1:
        raise ValueError(f"params must be 1D, got shape {params.shape}")
    if params.shape[0] != expected_len:
        raise ValueError(
            f"params length {params.shape[0]} does not match expected length {expected_len}"
        )


__all__ = [
    "infer_device",
    "as_float_tensor",
    "as_complex_tensor",
    "validate_params_shape",
]

