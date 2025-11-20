"""Tests for qconduit.channels.builtins."""

import pytest
import torch

from qconduit.backend.statevector import zero_state
from qconduit.channels.builtins import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    GeneralKraus,
    PhaseDampingChannel,
    PhaseFlipChannel,
)
from qconduit.channels.utils import density_from_statevector


class TestDepolarizingChannel:
    """Test DepolarizingChannel."""

    def test_construction_valid_p(self):
        """Test construction with valid p values."""
        for p in [0.0, 0.1, 0.5, 1.0]:
            channel = DepolarizingChannel(p=p)
            assert channel.n_qubits == 1
            assert len(channel.kraus_ops) == 4
            assert channel.is_cptp()

    def test_construction_invalid_p(self):
        """Test that invalid p values raise errors."""
        with pytest.raises(ValueError, match="must be in"):
            DepolarizingChannel(p=-0.1)
        with pytest.raises(ValueError, match="must be in"):
            DepolarizingChannel(p=1.1)

    def test_p_zero_is_identity(self):
        """Test that p=0 gives identity channel."""
        channel = DepolarizingChannel(p=0.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)
        assert torch.allclose(rho_out, rho0, atol=1e-8)

    def test_p_one_complete_depolarization(self):
        """Test that p=1 on |0⟩⟨0| gives (1/3)|0⟩⟨0| + (2/3)|1⟩⟨1|."""
        channel = DepolarizingChannel(p=1.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)

        # For p=1, applied to |0⟩⟨0|, we get (1/3)|0⟩⟨0| + (2/3)|1⟩⟨1|
        expected = torch.zeros((2, 2), dtype=dtype, device=device)
        expected[0, 0] = 1.0 / 3.0
        expected[1, 1] = 2.0 / 3.0
        assert torch.allclose(rho_out, expected, atol=1e-8)


class TestBitFlipChannel:
    """Test BitFlipChannel."""

    def test_construction_valid_p(self):
        """Test construction with valid p values."""
        for p in [0.0, 0.1, 0.5, 1.0]:
            channel = BitFlipChannel(p=p)
            assert channel.n_qubits == 1
            assert len(channel.kraus_ops) == 2
            assert channel.is_cptp()

    def test_construction_invalid_p(self):
        """Test that invalid p values raise errors."""
        with pytest.raises(ValueError, match="must be in"):
            BitFlipChannel(p=-0.1)
        with pytest.raises(ValueError, match="must be in"):
            BitFlipChannel(p=1.1)

    def test_p_one_flips_zero_to_one(self):
        """Test that p=1 flips |0⟩ to |1⟩."""
        channel = BitFlipChannel(p=1.0)
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


class TestPhaseFlipChannel:
    """Test PhaseFlipChannel."""

    def test_construction_valid_p(self):
        """Test construction with valid p values."""
        for p in [0.0, 0.1, 0.5, 1.0]:
            channel = PhaseFlipChannel(p=p)
            assert channel.n_qubits == 1
            assert len(channel.kraus_ops) == 2
            assert channel.is_cptp()

    def test_phase_flip_on_superposition(self):
        """Test phase flip on |+⟩ state."""
        channel = PhaseFlipChannel(p=1.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        # |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
        psi_plus = torch.ones(2, dtype=dtype, device=device) / torch.sqrt(
            torch.tensor(2.0, dtype=dtype)
        )
        rho_plus = density_from_statevector(psi_plus)

        rho_out = channel.apply_to_density(rho_plus)

        # Phase flip should give |−⟩ = (|0⟩ - |1⟩) / sqrt(2)
        psi_minus = torch.tensor([1.0, -1.0], dtype=dtype, device=device) / torch.sqrt(
            torch.tensor(2.0, dtype=dtype)
        )
        rho_minus = density_from_statevector(psi_minus)

        assert torch.allclose(rho_out, rho_minus, atol=1e-8)


class TestPhaseDampingChannel:
    """Test PhaseDampingChannel."""

    def test_construction_valid_p(self):
        """Test construction with valid p values."""
        for p in [0.0, 0.1, 0.5, 1.0]:
            channel = PhaseDampingChannel(p=p)
            assert channel.n_qubits == 1
            assert len(channel.kraus_ops) == 2
            assert channel.is_cptp()

    def test_phase_damping_preserves_populations(self):
        """Test that phase damping preserves diagonal elements."""
        channel = PhaseDampingChannel(p=0.5)
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |1⟩
        psi1 = torch.zeros(2, dtype=dtype, device=device)
        psi1[1] = 1.0
        rho1 = density_from_statevector(psi1)

        rho_out = channel.apply_to_density(rho1)

        # Population of |1⟩ should be preserved
        assert abs(rho_out[1, 1].real - 1.0) < 1e-8

    def test_phase_damping_reduces_coherence(self):
        """Test that phase damping reduces off-diagonal coherence."""
        channel = PhaseDampingChannel(p=1.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
        psi_plus = torch.ones(2, dtype=dtype, device=device) / torch.sqrt(
            torch.tensor(2.0, dtype=dtype)
        )
        rho_plus = density_from_statevector(psi_plus)

        rho_out = channel.apply_to_density(rho_plus)

        # Off-diagonal should be zero (complete dephasing)
        assert abs(rho_out[0, 1]) < 1e-8
        assert abs(rho_out[1, 0]) < 1e-8


class TestAmplitudeDampingChannel:
    """Test AmplitudeDampingChannel."""

    def test_construction_valid_gamma(self):
        """Test construction with valid gamma values."""
        for gamma in [0.0, 0.1, 0.5, 1.0]:
            channel = AmplitudeDampingChannel(gamma=gamma)
            assert channel.n_qubits == 1
            assert len(channel.kraus_ops) == 2
            assert channel.is_cptp()

    def test_construction_invalid_gamma(self):
        """Test that invalid gamma values raise errors."""
        with pytest.raises(ValueError, match="must be in"):
            AmplitudeDampingChannel(gamma=-0.1)
        with pytest.raises(ValueError, match="must be in"):
            AmplitudeDampingChannel(gamma=1.1)

    def test_gamma_zero_is_identity(self):
        """Test that gamma=0 gives identity channel."""
        channel = AmplitudeDampingChannel(gamma=0.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)
        assert torch.allclose(rho_out, rho0, atol=1e-8)

    def test_gamma_one_decays_one_to_zero(self):
        """Test that gamma=1 decays |1⟩ to |0⟩."""
        channel = AmplitudeDampingChannel(gamma=1.0)
        device = torch.device("cpu")
        dtype = torch.complex128

        # Start with |1⟩
        psi1 = torch.zeros(2, dtype=dtype, device=device)
        psi1[1] = 1.0
        rho1 = density_from_statevector(psi1)

        rho_out = channel.apply_to_density(rho1)

        # Should be |0⟩⟨0|
        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        assert torch.allclose(rho_out, rho0, atol=1e-8)

    def test_amplitude_damping_preserves_zero(self):
        """Test that amplitude damping preserves |0⟩."""
        channel = AmplitudeDampingChannel(gamma=0.5)
        device = torch.device("cpu")
        dtype = torch.complex128

        psi0 = zero_state(n_qubits=1, device=device, dtype=dtype)
        rho0 = density_from_statevector(psi0)

        rho_out = channel.apply_to_density(rho0)

        # Should still be |0⟩⟨0|
        assert torch.allclose(rho_out, rho0, atol=1e-8)


class TestGeneralKraus:
    """Test GeneralKraus factory function."""

    def test_construction_valid_operators(self):
        """Test construction with valid Kraus operators."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # Identity channel
        I = torch.eye(2, dtype=dtype, device=device)
        channel = GeneralKraus([I])
        assert channel.n_qubits == 1
        assert len(channel.kraus_ops) == 1
        assert channel.is_cptp()

    def test_construction_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="must contain"):
            GeneralKraus([])

    def test_construction_mismatched_shapes_raises(self):
        """Test that mismatched shapes raise error."""
        device = torch.device("cpu")
        dtype = torch.complex128

        I = torch.eye(2, dtype=dtype, device=device)
        I4 = torch.eye(4, dtype=dtype, device=device)

        with pytest.raises(ValueError, match="shape"):
            GeneralKraus([I, I4])

    def test_construction_non_power_of_two_raises(self):
        """Test that non-power-of-two dimension raises error."""
        device = torch.device("cpu")
        dtype = torch.complex128

        # 3x3 matrix (not a power of 2)
        K = torch.eye(3, dtype=dtype, device=device)

        with pytest.raises(ValueError, match="power of 2"):
            GeneralKraus([K])


class TestTensorExtend:
    """Test tensor extension of built-in channels."""

    def test_depolarizing_extend_to_two_qubits(self):
        """Test extending depolarizing channel to 2-qubit system."""
        channel = DepolarizingChannel(p=0.1)
        device = torch.device("cpu")

        # Extend to act on qubit 0 in a 2-qubit system
        extended = channel.tensor_extend(total_n_qubits=2, target_qubits=[0])

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

    def test_bitflip_extend_preserves_other_qubit(self):
        """Test that extending bit-flip preserves other qubit."""
        channel = BitFlipChannel(p=1.0)  # Complete bit flip
        device = torch.device("cpu")
        dtype = torch.complex128

        # Extend to act on qubit 0 in a 2-qubit system
        extended = channel.tensor_extend(total_n_qubits=2, target_qubits=[0])

        # Start with |00⟩
        psi00 = zero_state(n_qubits=2, device=device, dtype=dtype)
        rho00 = density_from_statevector(psi00)

        rho_out = extended.apply_to_density(rho00)

        # Should be |01⟩ (qubit 0 flipped, qubit 0 is LSB)
        psi01 = torch.zeros(4, dtype=dtype, device=device)
        psi01[1] = 1.0  # |01⟩ = index 1 (qubit 0 is LSB)
        rho01 = density_from_statevector(psi01)

        assert torch.allclose(rho_out, rho01, atol=1e-8)

