"""Base quantum module class."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from .device import Device, device as device_factory, default_device

if TYPE_CHECKING:
    pass


class QuantumModule(nn.Module):
    """
    Base class for all quantum layers, mirroring torch.nn.Module semantics.

    This class provides a foundation for quantum operations that integrate with
    PyTorch's module system. It manages the number of qubits and device placement.

    Subclasses should implement the forward() method to define quantum operations.
    """

    def __init__(
        self,
        n_qubits: int,
        device: Device | str | torch.device | None = None,
    ) -> None:
        """
        Initialize a QuantumModule.

        Args:
            n_qubits: Number of logical qubits this module operates on. Must be >= 1.
            device: Device specification. Can be a Device instance, device name string
                ("sv_cpu", "sv_cuda"), torch.device, or None (defaults to "sv_cpu").

        Raises:
            ValueError: If n_qubits < 1.
        """
        super().__init__()
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

        self.n_qubits = n_qubits

        # Resolve device
        if device is None:
            qdevice = default_device()
        elif isinstance(device, Device):
            qdevice = device
        elif isinstance(device, str):
            qdevice = device_factory(device)
        elif isinstance(device, torch.device):
            if device.type == "cpu":
                qdevice = device_factory("sv_cpu")
            elif device.type == "cuda":
                qdevice = device_factory("sv_cuda")
            else:
                raise ValueError(
                    f"Unsupported torch.device type: {device.type}. "
                    "Only 'cpu' and 'cuda' are supported."
                )
        else:
            raise TypeError(
                f"device must be Device, str, torch.device, or None, got {type(device)}"
            )

        self.qdevice = qdevice

    @property
    def device(self) -> Device:
        """
        Return the quantum device associated with this module.

        Returns:
            The Device instance.
        """
        return self.qdevice

    def to(
        self,
        device: Device | str | torch.device | None = None,
        *args,
        **kwargs,
    ) -> QuantumModule:
        """
        Move the module to a specified device.

        This method updates both the quantum device (qdevice) and moves all
        parameters and buffers to the underlying PyTorch device.

        Args:
            device: Device specification. Can be a Device instance, device name string,
                torch.device, or None (in which case only args/kwargs are passed to
                the parent to() method).
            *args: Additional positional arguments passed to nn.Module.to().
            **kwargs: Additional keyword arguments passed to nn.Module.to().

        Returns:
            self (for method chaining).
        """
        if device is None:
            # Just pass through to parent
            super().to(*args, **kwargs)
            return self

        # Resolve device to Device instance
        if isinstance(device, Device):
            qdevice = device
        elif isinstance(device, str):
            qdevice = device_factory(device)
        elif isinstance(device, torch.device):
            if device.type == "cpu":
                qdevice = device_factory("sv_cpu")
            elif device.type == "cuda":
                qdevice = device_factory("sv_cuda")
            else:
                raise ValueError(
                    f"Unsupported torch.device type: {device.type}. "
                    "Only 'cpu' and 'cuda' are supported."
                )
        else:
            raise TypeError(
                f"device must be Device, str, torch.device, or None, got {type(device)}"
            )

        # Update internal device
        self.qdevice = qdevice

        # Move parameters and buffers to the underlying PyTorch device
        super().to(qdevice.as_torch_device(), *args, **kwargs)

        return self

