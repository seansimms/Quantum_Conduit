"""Core abstractions for quantum modules and devices."""

from .device import Device, device, default_device
from .module import QuantumModule

__all__ = ["Device", "device", "default_device", "QuantumModule"]
