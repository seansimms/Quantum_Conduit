"""Device abstraction for quantum operations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Device:
    """
    Represents a logical quantum device with an underlying PyTorch device and dtype settings.

    This class encapsulates device information for quantum operations. It is immutable
    in the sense that its attributes should not be modified after construction.
    """

    def __init__(
        self,
        name: str,
        torch_device: torch.device,
        dtype: torch.dtype = torch.float32,
        complex_dtype: torch.dtype = torch.complex64,
    ) -> None:
        """
        Initialize a Device.

        Args:
            name: Logical device name (e.g., "sv_cpu", "sv_cuda").
            torch_device: Underlying PyTorch device.
            dtype: Default floating-point dtype for real components.
            complex_dtype: Default complex dtype for quantum states.
        """
        self.name = name
        self.torch_device = torch_device
        self.dtype = dtype
        self.complex_dtype = complex_dtype

    def __repr__(self) -> str:
        """Return a string representation of the device."""
        return (
            f"Device(name={self.name!r}, torch_device={self.torch_device}, "
            f"complex_dtype={self.complex_dtype})"
        )

    def as_torch_device(self) -> torch.device:
        """
        Return the underlying PyTorch device.

        Returns:
            The PyTorch device object.
        """
        return self.torch_device


def device(name: str) -> Device:
    """
    Create a Device instance from a device name.

    Supported device names:
        - "sv_cpu": CPU-based statevector device
        - "sv_cuda": CUDA-based statevector device (only if CUDA is available)

    Args:
        name: Device name string.

    Returns:
        A Device instance.

    Raises:
        RuntimeError: If "sv_cuda" is requested but CUDA is not available.
        ValueError: If the device name is not supported.
    """
    if name == "sv_cpu":
        return Device(
            name="sv_cpu",
            torch_device=torch.device("cpu"),
            dtype=torch.float32,
            complex_dtype=torch.complex64,
        )
    elif name == "sv_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False"
            )
        return Device(
            name="sv_cuda",
            torch_device=torch.device("cuda"),
            dtype=torch.float32,
            complex_dtype=torch.complex64,
        )
    else:
        supported = ["sv_cpu", "sv_cuda"]
        raise ValueError(
            f"Unsupported device name: {name!r}. Supported devices: {supported}"
        )


def default_device() -> Device:
    """
    Return the default device (CPU-based statevector device).

    Returns:
        A Device instance for "sv_cpu".
    """
    return device("sv_cpu")

