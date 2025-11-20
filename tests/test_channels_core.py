"""Tests for qconduit.channels.core."""

import pytest
import torch

from qconduit.backend.statevector import zero_state
from qconduit.channels.builtins import BitFlipChannel, DepolarizingChannel
from qconduit.channels.core import KrausChannel
from qconduit.channels.utils import density_from_statevector


class TestKrausChannelConstruction:
    """Test KrausChannel construction and validation."""

    def test_valid_single_qubit_channel(self):
        """Test construction of a valid single-qubit channel."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Identity channel: K_0 = I
        I = torch.eye(2, dtype=dtype, device=device)
        channel = KrausChannel(kraus_ops=(I,), n_qubits=1)
        assert channel.n_qubits == 1
        assert len(channel.kraus_ops) == 1
        assert channel.is_cptp()

    def test_depolarizing_channel_is_cptp(self):
        """Test that DepolarizingChannel is CPTP."""
        channel = DepolarizingChannel(p=0.1)
        assert channel.is_cptp()
        assert channel.n_qubits == 1

    def test_invalid_shape_raises(self):
        """Test that invalid Kraus operator shapes raise errors."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Non-square matrix
        K_bad = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        with pytest.raises(ValueError, match="must be square"):
            KrausChannel(kraus_ops=(K_bad,), n_qubits=1)

        # Wrong dimension
        K_wrong_dim = torch.eye(4, dtype=dtype, device=device)
        with pytest.raises(ValueError, match="must have shape"):
            KrausChannel(kraus_ops=(K_wrong_dim,), n_qubits=1)

    def test_non_cptp_raises(self):
        """Test that non-CPTP channels raise errors."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Non-CPTP: K = 2*I (too large)
        K = 2.0 * torch.eye(2, dtype=dtype, device=device)
        with pytest.raises(ValueError, match="CPTP"):
            KrausChannel(kraus_ops=(K,), n_qubits=1)


class TestApplyToDensity:
    """Test apply_to_density method."""

    def test_identity_channel_preserves_state(self):
        """Test that identity channel preserves density matrix."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Create |0⟩ density matrix
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        # Identity channel
        I = torch.eye(2, dtype=dtype, device=device)
        channel = KrausChannel(kraus_ops=(I,), n_qubits=1)

        rho_out = channel.apply_to_density(rho0)
        assert torch.allclose(rho_out, rho0, atol=1e-10)

    def test_depolarizing_complete_depolarization(self):
        """Test complete depolarization (p=1) on |0⟩⟨0|."""
        channel = DepolarizingChannel(p=1.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |0⟩ density
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)

        # For p=1, applied to |0⟩⟨0|, we get (1/3)|0⟩⟨0| + (2/3)|1⟩⟨1|
        # This is because X|0⟩=|1⟩, Y|0⟩=i|1⟩, Z|0⟩=|0⟩
        expected = torch.zeros((2, 2), dtype=dtype, device=device)
        expected[0, 0] = 1.0 / 3.0
        expected[1, 1] = 2.0 / 3.0
        assert torch.allclose(rho_out, expected, atol=1e-8)

        # Check trace = 1
        trace = torch.trace(rho_out).real
        assert abs(trace - 1.0) < 1e-10

        # Check Hermitian
        diff_herm = torch.abs(rho_out - rho_out.conj().T)
        assert torch.max(diff_herm).item() < 1e-10

    def test_bitflip_channel_on_zero_state(self):
        """Test bit-flip channel on |0⟩."""
        channel = BitFlipChannel(p=1.0)  # Complete bit flip
        device = torch.device("cpu")
        dtype = torch.complex128

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)

        # Should be |1⟩⟨1|
        psi1 = torch.zeros(2, dtype=dtype, device=device)
        psi1[1] = 1.0
        rho1 = density_from_statevector(psi1)

        assert torch.allclose(rho_out, rho1, atol=1e-8)


class TestComposition:
    """Test channel composition."""

    def test_compose_identity_channels(self):
        """Test composing two identity channels."""
        device = torch.device("cpu")
        dtype = torch.complex128

        I = torch.eye(2, dtype=dtype, device=device)
        chan1 = KrausChannel(kraus_ops=(I,), n_qubits=1)
        chan2 = KrausChannel(kraus_ops=(I,), n_qubits=1)

        composed = chan1.compose(chan2)
        assert composed.n_qubits == 1
        assert len(composed.kraus_ops) == 1
        assert torch.allclose(composed.kraus_ops[0], I, atol=1e-10)

    def test_compose_different_channels(self):
        """Test composing different channels."""
        chan1 = DepolarizingChannel(p=0.1)
        chan2 = BitFlipChannel(p=0.2)

        composed = chan1.compose(chan2)
        assert composed.n_qubits == 1
        # Composed channel should have len(chan1.kraus_ops) * len(chan2.kraus_ops) operators
        assert len(composed.kraus_ops) == len(chan1.kraus_ops) * len(chan2.kraus_ops)
        assert composed.is_cptp()

    def test_compose_mismatched_n_qubits_raises(self):
        """Test that composing channels with different n_qubits raises error."""
        chan1 = DepolarizingChannel(p=0.1)  # 1 qubit

        # Create a 2-qubit channel (identity)
        device = torch.device("cpu")
        dtype = torch.complex128
        I2 = torch.eye(4, dtype=dtype, device=device)
        chan2 = KrausChannel(kraus_ops=(I2,), n_qubits=2)

        with pytest.raises(ValueError, match="different n_qubits"):
            chan1.compose(chan2)


class TestSampleStateAfterChannel:
    """Test stochastic Kraus sampling."""

    def test_bitflip_deterministic_sampling(self):
        """Test deterministic sampling with bit-flip channel."""
        channel = BitFlipChannel(p=1.0)  # Complete bit flip
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |0⟩
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)

        # Sample with fixed seed
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        psi_out, idx = channel.sample_state_after_channel(psi0, generator=generator)

        # With p=1.0, should always get |1⟩
        psi1 = torch.zeros(2, dtype=dtype, device=device)
        psi1[1] = 1.0

        assert torch.allclose(psi_out, psi1, atol=1e-8)
        assert idx == 1  # Second Kraus operator (X)

    def test_identity_channel_preserves_state(self):
        """Test that identity channel preserves state in sampling."""
        device = torch.device("cpu")
        dtype = torch.complex128

        I = torch.eye(2, dtype=dtype, device=device)
        channel = KrausChannel(kraus_ops=(I,), n_qubits=1)

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)

        generator = torch.Generator(device=device)
        generator.manual_seed(0)

        psi_out, idx = channel.sample_state_after_channel(psi0, generator=generator)

        assert torch.allclose(psi_out, psi0, atol=1e-8)
        assert idx == 0


class TestTensorExtend:
    """Test tensor extension to larger systems."""

    def test_single_qubit_extend_to_two_qubits(self):
        """Test extending single-qubit channel to 2-qubit system."""
        channel = DepolarizingChannel(p=0.1)
        device = torch.device("cpu")

        # Extend to act on qubit 1 in a 2-qubit system
        extended = channel.tensor_extend(total_n_qubits=2, target_qubits=[1])

        assert extended.n_qubits == 2
        assert extended.is_cptp()

        # Apply to |00⟩ density
        psi00 = zero_state(n_qubits=2, device=device, dtype=torch.complex128)
        rho00 = density_from_statevector(psi00)

        rho_out = extended.apply_to_density(rho00)

        # Check trace preserved
        trace = torch.trace(rho_out).real
        assert abs(trace - 1.0) < 1e-10

        # Check Hermitian
        diff_herm = torch.abs(rho_out - rho_out.conj().T)
        assert torch.max(diff_herm).item() < 1e-10

    def test_extend_invalid_qubits_raises(self):
        """Test that invalid target qubits raise errors."""
        channel = DepolarizingChannel(p=0.1)

        # Target qubit out of range
        with pytest.raises(ValueError, match="must be in"):
            channel.tensor_extend(total_n_qubits=2, target_qubits=[2])

        # Wrong number of target qubits
        with pytest.raises(ValueError, match="must equal"):
            channel.tensor_extend(total_n_qubits=2, target_qubits=[0, 1])


class TestSuperoperator:
    """Test superoperator representation."""

    def test_superoperator_shape(self):
        """Test that superoperator has correct shape."""
        channel = DepolarizingChannel(p=0.1)
        S = channel.as_superoperator()

        # For 1 qubit: (2*2, 2*2) = (4, 4)
        assert S.shape == (4, 4)

    def test_superoperator_applies_correctly(self):
        """Test that superoperator correctly applies channel."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Identity channel
        I = torch.eye(2, dtype=dtype, device=device)
        channel = KrausChannel(kraus_ops=(I,), n_qubits=1)

        # Test on |0⟩ density
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        # Apply via channel
        rho_channel = channel.apply_to_density(rho0)

        # Apply via superoperator: vec(rho) = S @ vec(rho0)
        S = channel.as_superoperator()
        rho0_vec = rho0.flatten()
        rho_super_vec = S @ rho0_vec
        rho_super = rho_super_vec.reshape(2, 2)

        assert torch.allclose(rho_channel, rho_super, atol=1e-8)

