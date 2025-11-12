"""Tests for core device and module abstractions."""

import pytest
import torch
import qconduit as qc
from qconduit.core.device import Device, device, default_device
from qconduit.core.module import QuantumModule


class TestDevice:
    """Tests for Device class."""

    def test_device_creation(self):
        """Test Device can be created with all parameters."""
        dev = Device(
            name="test",
            torch_device=torch.device("cpu"),
            dtype=torch.float32,
            complex_dtype=torch.complex64,
        )
        assert dev.name == "test"
        assert dev.torch_device == torch.device("cpu")
        assert dev.dtype == torch.float32
        assert dev.complex_dtype == torch.complex64

    def test_device_repr(self):
        """Test Device __repr__."""
        dev = Device(
            name="sv_cpu",
            torch_device=torch.device("cpu"),
            complex_dtype=torch.complex64,
        )
        repr_str = repr(dev)
        assert "sv_cpu" in repr_str
        assert "cpu" in repr_str

    def test_as_torch_device(self):
        """Test as_torch_device returns correct device."""
        dev = Device(
            name="sv_cpu",
            torch_device=torch.device("cpu"),
            complex_dtype=torch.complex64,
        )
        assert dev.as_torch_device() == torch.device("cpu")


class TestDeviceFactory:
    """Tests for device factory function."""

    def test_device_sv_cpu(self):
        """Test device('sv_cpu') returns correct Device."""
        dev = device("sv_cpu")
        assert dev.name == "sv_cpu"
        assert dev.torch_device == torch.device("cpu")
        assert dev.complex_dtype == torch.complex64

    def test_device_sv_cuda_available(self):
        """Test device('sv_cuda') when CUDA is available."""
        if torch.cuda.is_available():
            dev = device("sv_cuda")
            assert dev.name == "sv_cuda"
            assert dev.torch_device.type == "cuda"
            assert dev.complex_dtype == torch.complex64
        else:
            pytest.skip("CUDA not available")

    def test_device_sv_cuda_unavailable(self):
        """Test device('sv_cuda') raises when CUDA is unavailable."""
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA device requested"):
                device("sv_cuda")
        else:
            pytest.skip("CUDA is available, cannot test failure case")

    def test_device_unsupported_name(self):
        """Test device() raises for unsupported device name."""
        with pytest.raises(ValueError, match="Unsupported device name"):
            device("invalid_device")

    def test_default_device(self):
        """Test default_device returns sv_cpu."""
        dev = default_device()
        assert dev.name == "sv_cpu"
        assert dev.torch_device == torch.device("cpu")


class TestQuantumModule:
    """Tests for QuantumModule base class."""

    def test_quantum_module_creation_default_device(self):
        """Test QuantumModule creation with default device."""
        module = QuantumModule(n_qubits=2)
        assert module.n_qubits == 2
        assert module.device.name == "sv_cpu"

    def test_quantum_module_creation_with_device_string(self):
        """Test QuantumModule creation with device string."""
        module = QuantumModule(n_qubits=2, device="sv_cpu")
        assert module.n_qubits == 2
        assert module.device.name == "sv_cpu"

    def test_quantum_module_creation_with_device_object(self):
        """Test QuantumModule creation with Device object."""
        dev = device("sv_cpu")
        module = QuantumModule(n_qubits=2, device=dev)
        assert module.n_qubits == 2
        assert module.device == dev

    def test_quantum_module_creation_with_torch_device(self):
        """Test QuantumModule creation with torch.device."""
        module = QuantumModule(n_qubits=2, device=torch.device("cpu"))
        assert module.n_qubits == 2
        assert module.device.name == "sv_cpu"

    def test_quantum_module_invalid_n_qubits(self):
        """Test QuantumModule raises for invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            QuantumModule(n_qubits=0)

        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            QuantumModule(n_qubits=-1)

    def test_quantum_module_to_device_string(self):
        """Test QuantumModule.to() with device string."""
        module = QuantumModule(n_qubits=2, device="sv_cpu")
        module.to("sv_cpu")
        assert module.device.name == "sv_cpu"

    def test_quantum_module_to_device_object(self):
        """Test QuantumModule.to() with Device object."""
        module = QuantumModule(n_qubits=2)
        new_dev = device("sv_cpu")
        module.to(new_dev)
        assert module.device == new_dev

    def test_quantum_module_to_torch_device(self):
        """Test QuantumModule.to() with torch.device."""
        module = QuantumModule(n_qubits=2)
        module.to(torch.device("cpu"))
        assert module.device.name == "sv_cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantum_module_to_cuda(self):
        """Test QuantumModule.to() with CUDA device."""
        module = QuantumModule(n_qubits=2, device="sv_cpu")
        module.to("sv_cuda")
        assert module.device.name == "sv_cuda"
        assert module.device.torch_device.type == "cuda"

    def test_quantum_module_to_none(self):
        """Test QuantumModule.to() with None (passes through to parent)."""
        module = QuantumModule(n_qubits=2)
        # Should not raise
        result = module.to(None)
        assert result is module

    def test_quantum_module_has_parameters(self):
        """Test QuantumModule can have parameters moved to device."""
        # Create a simple subclass with a parameter
        class TestModule(QuantumModule):
            def __init__(self, n_qubits):
                super().__init__(n_qubits)
                self.param = torch.nn.Parameter(torch.randn(2, 2))

        module = TestModule(n_qubits=2)
        assert module.param.device.type == "cpu"

        if torch.cuda.is_available():
            module.to("sv_cuda")
            assert module.param.device.type == "cuda"

    def test_quantum_module_invalid_torch_device_type(self):
        """Test QuantumModule raises for unsupported torch.device type."""
        with pytest.raises(ValueError, match="Unsupported torch.device type"):
            QuantumModule(n_qubits=2, device=torch.device("mps"))

    def test_quantum_module_invalid_device_type(self):
        """Test QuantumModule raises for invalid device type."""
        with pytest.raises(TypeError):
            QuantumModule(n_qubits=2, device=123)  # type: ignore

    def test_quantum_module_to_invalid_torch_device_type(self):
        """Test QuantumModule.to() raises for unsupported torch.device type."""
        module = QuantumModule(n_qubits=2)
        with pytest.raises(ValueError, match="Unsupported torch.device type"):
            module.to(torch.device("mps"))


class TestQuantumModuleSubclass:
    """Tests for QuantumModule subclassing."""

    def test_quantum_module_can_be_subclassed(self):
        """Test QuantumModule can be subclassed."""

        class SimpleQuantumLayer(QuantumModule):
            def forward(self, x):
                return x

        module = SimpleQuantumLayer(n_qubits=2)
        assert module.n_qubits == 2
        assert isinstance(module, QuantumModule)
        assert isinstance(module, torch.nn.Module)

