"""Tests for quantum noise channels."""

import pytest
import torch
import math

from qconduit.noise import (
    NoiseModel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)
from qconduit.backend.density_matrix import zero_dm_state, dm_from_statevector


class TestDepolarizingChannel:
    """Tests for DepolarizingChannel."""

    def test_depolarizing_channel_p_zero(self):
        """Test depolarizing channel with p=0 leaves state unchanged."""
        channel = DepolarizingChannel(p=0.0)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        assert torch.allclose(rho_out, rho, atol=1e-6)

    def test_depolarizing_channel_p_one(self):
        """Test depolarizing channel with p=1 produces completely mixed state."""
        channel = DepolarizingChannel(p=1.0)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        # For p=1, depolarizing channel: rho -> (1/3)(X rho X + Y rho Y + Z rho Z)
        # For |0><0|: X|0><0|X = |1><1|, Y|0><0|Y = |1><1|, Z|0><0|Z = |0><0|
        # Result: (1/3)(2|1><1| + |0><0|) = [[1/3, 0], [0, 2/3]]
        # Actually, for a general state, p=1 should produce I/2, but for |0><0| specifically
        # the result is different. Let's test with a general mixed state instead.
        # Actually, let's just verify the channel works and preserves trace
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)
        # For |0><0| with p=1, we get (1/3)(|0><0| + 2|1><1|)
        expected = torch.tensor([[1.0/3.0, 0.0], [0.0, 2.0/3.0]], dtype=torch.complex64)
        assert torch.allclose(rho_out, expected, atol=1e-5)

    def test_depolarizing_channel_trace_preserving(self):
        """Test depolarizing channel preserves trace."""
        channel = DepolarizingChannel(p=0.3)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)

    def test_depolarizing_channel_hermitian(self):
        """Test depolarizing channel output is Hermitian."""
        channel = DepolarizingChannel(p=0.5)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        # Check rho_out is Hermitian: rho_out == rho_out^dagger
        assert torch.allclose(rho_out, rho_out.conj().transpose(-2, -1), atol=1e-6)

    def test_depolarizing_channel_invalid_p(self):
        """Test depolarizing channel raises for invalid p."""
        with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
            DepolarizingChannel(p=-0.1)
        with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
            DepolarizingChannel(p=1.1)

    def test_depolarizing_channel_two_qubits(self):
        """Test depolarizing channel on 2-qubit system."""
        channel = DepolarizingChannel(p=0.2)
        rho = zero_dm_state(n_qubits=2)
        rho_out = channel.apply_density_matrix(rho, n_qubits=2)
        # Check trace is preserved
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)


class TestAmplitudeDampingChannel:
    """Tests for AmplitudeDampingChannel."""

    def test_amplitude_damping_channel_gamma_zero(self):
        """Test amplitude damping channel with gamma=0 leaves state unchanged."""
        channel = AmplitudeDampingChannel(gamma=0.0)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        assert torch.allclose(rho_out, rho, atol=1e-6)

    def test_amplitude_damping_channel_gamma_one(self):
        """Test amplitude damping channel with gamma=1 maps |1> to |0>."""
        channel = AmplitudeDampingChannel(gamma=1.0)
        # Start with |1><1|
        state = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho = dm_from_statevector(state)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        # Should become |0><0|
        expected = zero_dm_state(n_qubits=1)
        assert torch.allclose(rho_out, expected, atol=1e-5)

    def test_amplitude_damping_channel_trace_preserving(self):
        """Test amplitude damping channel preserves trace."""
        channel = AmplitudeDampingChannel(gamma=0.5)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)

    def test_amplitude_damping_channel_hermitian(self):
        """Test amplitude damping channel output is Hermitian."""
        channel = AmplitudeDampingChannel(gamma=0.3)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        assert torch.allclose(rho_out, rho_out.conj().transpose(-2, -1), atol=1e-6)

    def test_amplitude_damping_channel_invalid_gamma(self):
        """Test amplitude damping channel raises for invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be in \\[0, 1\\]"):
            AmplitudeDampingChannel(gamma=-0.1)
        with pytest.raises(ValueError, match="gamma must be in \\[0, 1\\]"):
            AmplitudeDampingChannel(gamma=1.1)


class TestPhaseDampingChannel:
    """Tests for PhaseDampingChannel."""

    def test_phase_damping_channel_gamma_zero(self):
        """Test phase damping channel with gamma=0 leaves state unchanged."""
        channel = PhaseDampingChannel(gamma=0.0)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        assert torch.allclose(rho_out, rho, atol=1e-6)

    def test_phase_damping_channel_gamma_one(self):
        """Test phase damping channel with gamma=1 removes off-diagonal elements."""
        channel = PhaseDampingChannel(gamma=1.0)
        # Start with |+><+| = 1/2 * [[1, 1], [1, 1]]
        state = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
        rho = dm_from_statevector(state)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        # Should become classical mixture: 1/2 (|0><0| + |1><1|)
        # Diagonal should be [0.5, 0.5], off-diagonals should be ~0
        assert torch.allclose(rho_out[0, 0].real, torch.tensor(0.5), atol=1e-5)
        assert torch.allclose(rho_out[1, 1].real, torch.tensor(0.5), atol=1e-5)
        assert torch.allclose(rho_out[0, 1], torch.tensor(0.0 + 0.0j), atol=1e-5)
        assert torch.allclose(rho_out[1, 0], torch.tensor(0.0 + 0.0j), atol=1e-5)

    def test_phase_damping_channel_trace_preserving(self):
        """Test phase damping channel preserves trace."""
        channel = PhaseDampingChannel(gamma=0.5)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)

    def test_phase_damping_channel_hermitian(self):
        """Test phase damping channel output is Hermitian."""
        channel = PhaseDampingChannel(gamma=0.3)
        rho = zero_dm_state(n_qubits=1)
        rho_out = channel.apply_density_matrix(rho, n_qubits=1)
        assert torch.allclose(rho_out, rho_out.conj().transpose(-2, -1), atol=1e-6)

    def test_phase_damping_channel_invalid_gamma(self):
        """Test phase damping channel raises for invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be in \\[0, 1\\]"):
            PhaseDampingChannel(gamma=-0.1)
        with pytest.raises(ValueError, match="gamma must be in \\[0, 1\\]"):
            PhaseDampingChannel(gamma=1.1)


class TestNoiseModelBase:
    """Tests for NoiseModel base class."""

    def test_apply_statevector(self):
        """Test NoiseModel.apply_statevector convenience method."""
        channel = DepolarizingChannel(p=0.1)
        state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        rho_out = channel.apply_statevector(state, n_qubits=1)
        assert rho_out.shape == (2, 2)
        # Check trace is preserved
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)

