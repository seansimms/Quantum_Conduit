"""Tests for qconduit.channels.circuit_integration."""

import pytest
import torch

from qconduit.backend.statevector import zero_state
from qconduit.channels.builtins import (
    BitFlipChannel,
    DepolarizingChannel,
    PhaseDampingChannel,
)
from qconduit.channels.circuit_integration import (
    NoisyCircuit,
    annotate_circuit_with_channels,
    apply_channel_schedule_to_state,
    apply_circuit_with_noise,
)
from qconduit.channels.utils import density_from_statevector
from qconduit.circuit import QuantumCircuit


class TestAnnotateCircuitWithChannels:
    """Test annotate_circuit_with_channels."""

    def test_annotate_simple_circuit(self):
        """Test annotating a simple circuit."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        channel = DepolarizingChannel(p=0.1)

        # Annotate: apply channel after first gate (H)
        noisy_circuit = annotate_circuit_with_channels(
            circuit, [(0, channel)]
        )

        assert noisy_circuit.circuit is circuit
        assert len(noisy_circuit.channel_locations) == 1
        assert 0 in noisy_circuit.channel_locations

    def test_annotate_invalid_gate_index_raises(self):
        """Test that invalid gate index raises error."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        channel = DepolarizingChannel(p=0.1)

        with pytest.raises(ValueError, match="out of range"):
            annotate_circuit_with_channels(circuit, [(10, channel)])


class TestApplyCircuitWithNoise:
    """Test apply_circuit_with_noise."""

    def test_no_noise_returns_statevector(self):
        """Test that circuit without noise returns statevector."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        result = apply_circuit_with_noise(circuit, [])

        # Should be statevector (1D)
        assert result.dim() == 1
        assert result.shape[0] == 2

    def test_with_noise_returns_density_matrix(self):
        """Test that circuit with noise returns density matrix."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        channel = DepolarizingChannel(p=0.1)

        result = apply_circuit_with_noise(circuit, [(0, channel)])

        # Should be density matrix (2D)
        assert result.dim() == 2
        assert result.shape == (2, 2)

        # Check trace = 1
        trace = torch.trace(result).real
        assert abs(trace - 1.0) < 1e-8

        # Check Hermitian
        diff_herm = torch.abs(result - result.conj().T)
        assert torch.max(diff_herm).item() < 1e-8

    def test_noise_reduces_purity(self):
        """Test that noise reduces purity (Tr(ρ²) < 1)."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        # Without noise: pure state
        result_pure = apply_circuit_with_noise(circuit, [])
        rho_pure = density_from_statevector(result_pure)
        purity_pure = torch.trace(rho_pure @ rho_pure).real
        assert abs(purity_pure - 1.0) < 1e-8

        # With noise: mixed state
        channel = DepolarizingChannel(p=0.5)
        result_mixed = apply_circuit_with_noise(circuit, [(0, channel)])
        purity_mixed = torch.trace(result_mixed @ result_mixed).real
        assert purity_mixed < 1.0 - 1e-8

    def test_small_circuit_with_dephasing(self):
        """Test small circuit with dephasing noise."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        # Apply dephasing after H gate
        channel = PhaseDampingChannel(p=0.5)

        result = apply_circuit_with_noise(circuit, [(0, channel)])

        # Should be density matrix
        assert result.dim() == 2
        assert result.shape == (4, 4)

        # Check trace = 1
        trace = torch.trace(result).real
        assert abs(trace - 1.0) < 1e-8

    def test_custom_initial_state(self):
        """Test with custom initial state."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("X", [0])

        channel = BitFlipChannel(p=0.1)

        # Start from |1⟩ instead of |0⟩
        device = torch.device("cpu")
        dtype = torch.complex128
        psi1 = torch.zeros(2, dtype=dtype, device=device)
        psi1[1] = 1.0

        result = apply_circuit_with_noise(
            circuit, [(0, channel)], psi0=psi1, dtype=dtype
        )

        # Should be density matrix (since noise is present)
        assert result.dim() == 2
        assert result.shape == (2, 2)


class TestApplyChannelScheduleToState:
    """Test apply_channel_schedule_to_state."""

    def test_single_channel_schedule(self):
        """Test applying a single channel."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |0⟩ density
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        channel = DepolarizingChannel(p=0.1)
        schedule = [(channel, [0])]

        rho_out = apply_channel_schedule_to_state(rho0, schedule, n_qubits=1)

        # Check trace preserved
        trace = torch.trace(rho_out).real
        assert abs(trace - 1.0) < 1e-10

    def test_multiple_channels_schedule(self):
        """Test applying multiple channels sequentially."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |0⟩ density
        psi0 = zero_state(n_qubits=2, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        channel1 = DepolarizingChannel(p=0.1)
        channel2 = BitFlipChannel(p=0.2)

        # Apply channel1 to qubit 0, then channel2 to qubit 1
        schedule = [(channel1, [0]), (channel2, [1])]

        rho_out = apply_channel_schedule_to_state(rho0, schedule, n_qubits=2)

        # Check trace preserved
        trace = torch.trace(rho_out).real
        assert abs(trace - 1.0) < 1e-10

        # Check Hermitian
        diff_herm = torch.abs(rho_out - rho_out.conj().T)
        assert torch.max(diff_herm).item() < 1e-10


class TestNoisyCircuit:
    """Test NoisyCircuit dataclass."""

    def test_construction(self):
        """Test NoisyCircuit construction."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])

        channel = DepolarizingChannel(p=0.1)
        channel_locations = {0: (channel, [0])}

        noisy_circuit = NoisyCircuit(
            circuit=circuit, channel_locations=channel_locations
        )

        assert noisy_circuit.circuit is circuit
        assert len(noisy_circuit.channel_locations) == 1

    def test_invalid_gate_index_raises(self):
        """Test that invalid gate index raises error."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        channel = DepolarizingChannel(p=0.1)
        channel_locations = {10: (channel, [0])}  # Invalid index

        with pytest.raises(ValueError, match="out of range"):
            NoisyCircuit(circuit=circuit, channel_locations=channel_locations)

    def test_invalid_target_qubits_raises(self):
        """Test that invalid target qubits raise error."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        channel = DepolarizingChannel(p=0.1)
        channel_locations = {0: (channel, [10])}  # Invalid qubit

        with pytest.raises(ValueError, match="out of range"):
            NoisyCircuit(circuit=circuit, channel_locations=channel_locations)


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_bell_state_with_noise(self):
        """Test creating Bell state with noise."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        # Apply dephasing after H gate
        channel = PhaseDampingChannel(p=0.3)

        result = apply_circuit_with_noise(circuit, [(0, channel)])

        # Should be density matrix
        assert result.dim() == 2
        assert result.shape == (4, 4)

        # Check trace = 1
        trace = torch.trace(result).real
        assert abs(trace - 1.0) < 1e-8

        # Purity should be less than 1 (due to noise)
        purity = torch.trace(result @ result).real
        assert purity < 1.0 - 1e-8

    def test_compare_pure_vs_noisy(self):
        """Test comparing pure and noisy circuit simulation."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        # Pure simulation
        result_pure = apply_circuit_with_noise(circuit, [])
        rho_pure = density_from_statevector(result_pure)

        # Noisy simulation
        channel = DepolarizingChannel(p=0.01)  # Small noise
        result_noisy = apply_circuit_with_noise(circuit, [(0, channel)])

        # Noisy result should be close but not identical to pure
        diff = torch.abs(result_noisy - rho_pure)
        max_diff = torch.max(diff).item()

        # Should be different (noise present)
        assert max_diff > 1e-6

        # But close (small noise)
        assert max_diff < 0.1



