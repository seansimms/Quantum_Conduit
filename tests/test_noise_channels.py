"""Tests for textbook noise channels."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.backend.density_matrix import apply_kraus_single_qubit, dm_from_statevector
from qconduit.backend.statevector import zero_state
from qconduit.noise import (
    SingleQubitChannel,
    amplitude_damping_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
)


class TestSingleQubitChannelValidation:
    """Test SingleQubitChannel validation."""

    def test_identity_channel_valid(self):
        """Test that identity channel is valid."""
        ch = identity_channel()
        assert len(ch.kraus_operators) == 1
        assert ch.kraus_operators[0].shape == (2, 2)
        assert ch.name == "identity"

    def test_invalid_empty_kraus_operators(self):
        """Test that empty Kraus operators raise ValueError."""
        I = torch.eye(2, dtype=torch.complex64)
        with pytest.raises(ValueError, match="non-empty"):
            SingleQubitChannel(name="test", kraus_operators=())

    def test_invalid_wrong_shape(self):
        """Test that wrong shape Kraus operators raise ValueError."""
        K = torch.eye(3, dtype=torch.complex64)
        with pytest.raises(ValueError, match="shape \\(2, 2\\)"):
            SingleQubitChannel(name="test", kraus_operators=(K,))

    def test_invalid_not_complex(self):
        """Test that non-complex Kraus operators raise ValueError."""
        K = torch.eye(2, dtype=torch.float32)
        with pytest.raises(ValueError, match="complex dtype"):
            SingleQubitChannel(name="test", kraus_operators=(K,))

    def test_invalid_not_trace_preserving(self):
        """Test that non-trace-preserving Kraus operators raise ValueError."""
        # Create invalid Kraus operators that don't satisfy sum K^dagger K = I
        K0 = torch.eye(2, dtype=torch.complex64) * 0.5  # Too small
        with pytest.raises(ValueError, match="trace-preserving"):
            SingleQubitChannel(name="test", kraus_operators=(K0,))


class TestTracePreservation:
    """Test that channels preserve trace and Hermiticity."""

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
    def test_depolarizing_trace_preservation(self, p):
        """Test depolarizing channel preserves trace."""
        ch = depolarizing_channel(p)
        # Test on random pure states
        for _ in range(5):
            # Generate random normalized statevector
            psi = torch.randn(2, dtype=torch.complex64)
            psi = psi / torch.norm(psi)
            rho = dm_from_statevector(psi)

            # Apply channel
            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )

            # Check trace
            trace = torch.trace(rho_new).real
            assert abs(trace - 1.0) < 1e-6, f"Trace not preserved: {trace}"

            # Check Hermiticity
            diff = rho_new - rho_new.conj().T
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6, f"Not Hermitian: max diff = {max_diff}"

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
    def test_phase_damping_trace_preservation(self, p):
        """Test phase damping channel preserves trace."""
        ch = phase_damping_channel(p)
        for _ in range(5):
            psi = torch.randn(2, dtype=torch.complex64)
            psi = psi / torch.norm(psi)
            rho = dm_from_statevector(psi)

            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )

            trace = torch.trace(rho_new).real
            assert abs(trace - 1.0) < 1e-6

            diff = rho_new - rho_new.conj().T
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6

    @pytest.mark.parametrize("gamma", [0.0, 0.1, 0.5, 1.0])
    def test_amplitude_damping_trace_preservation(self, gamma):
        """Test amplitude damping channel preserves trace."""
        ch = amplitude_damping_channel(gamma)
        for _ in range(5):
            psi = torch.randn(2, dtype=torch.complex64)
            psi = psi / torch.norm(psi)
            rho = dm_from_statevector(psi)

            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )

            trace = torch.trace(rho_new).real
            assert abs(trace - 1.0) < 1e-6

            diff = rho_new - rho_new.conj().T
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6


class TestLimitingCases:
    """Test limiting cases of channels."""

    def test_depolarizing_p0_is_identity(self):
        """Test depolarizing channel with p=0 is identity."""
        ch = depolarizing_channel(0.0)
        # Test on |0> and |1>
        for state_name, state_vec in [("|0>", torch.tensor([1.0, 0.0], dtype=torch.complex64)),
                                       ("|1>", torch.tensor([0.0, 1.0], dtype=torch.complex64))]:
            rho = dm_from_statevector(state_vec)
            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )
            diff = rho_new - rho
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6, f"p=0 should be identity for {state_name}"

    def test_depolarizing_p1_on_different_states(self):
        """Test depolarizing channel with p=1 on different states."""
        ch = depolarizing_channel(1.0)

        # For |0>, the channel produces (1/3)|0><0| + (2/3)|1><1|
        state_0 = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        rho_0 = dm_from_statevector(state_0)
        rho_new_0 = apply_kraus_single_qubit(
            rho_0, ch.kraus_operators, qubit=0, n_qubits=1
        )
        # Should be [[1/3, 0], [0, 2/3]]
        expected_0 = torch.zeros((2, 2), dtype=torch.complex64)
        expected_0[0, 0] = 1.0 / 3.0
        expected_0[1, 1] = 2.0 / 3.0
        diff_0 = rho_new_0 - expected_0
        max_diff_0 = torch.max(torch.abs(diff_0)).item()
        assert max_diff_0 < 1e-6, f"p=1 on |0> should produce [[1/3,0],[0,2/3]], got max diff = {max_diff_0}"

        # For |1>, the channel produces (2/3)|0><0| + (1/3)|1><1|
        state_1 = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho_1 = dm_from_statevector(state_1)
        rho_new_1 = apply_kraus_single_qubit(
            rho_1, ch.kraus_operators, qubit=0, n_qubits=1
        )
        # Should be [[2/3, 0], [0, 1/3]]
        expected_1 = torch.zeros((2, 2), dtype=torch.complex64)
        expected_1[0, 0] = 2.0 / 3.0
        expected_1[1, 1] = 1.0 / 3.0
        diff_1 = rho_new_1 - expected_1
        max_diff_1 = torch.max(torch.abs(diff_1)).item()
        assert max_diff_1 < 1e-6, f"p=1 on |1> should produce [[2/3,0],[0,1/3]], got max diff = {max_diff_1}"

        # The average over |0> and |1> should be I/2
        rho_avg = (rho_new_0 + rho_new_1) / 2.0
        I_half = torch.eye(2, dtype=torch.complex64) / 2.0
        diff_avg = rho_avg - I_half
        max_diff_avg = torch.max(torch.abs(diff_avg)).item()
        assert max_diff_avg < 1e-6, f"Average should be I/2, got max diff = {max_diff_avg}"

    def test_phase_damping_p0_is_identity(self):
        """Test phase damping channel with p=0 is identity."""
        ch = phase_damping_channel(0.0)
        state_vec = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        rho = dm_from_statevector(state_vec)
        rho_new = apply_kraus_single_qubit(
            rho, ch.kraus_operators, qubit=0, n_qubits=1
        )
        diff = rho_new - rho
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6

    def test_phase_damping_p1_on_plus_state(self):
        """Test phase damping with p=1 on |+> produces diagonal state."""
        ch = phase_damping_channel(1.0)
        # |+> = (|0> + |1>)/sqrt(2)
        state_vec = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
        rho = dm_from_statevector(state_vec)
        rho_new = apply_kraus_single_qubit(
            rho, ch.kraus_operators, qubit=0, n_qubits=1
        )

        # Should be I/2: diagonal [0.5, 0.5], off-diagonals 0
        expected = torch.eye(2, dtype=torch.complex64) / 2.0
        diff = rho_new - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"p=1 should dephase |+> to I/2, got max diff = {max_diff}"

    def test_phase_damping_p1_on_computational_basis(self):
        """Test phase damping with p=1 leaves |0> and |1> unchanged."""
        ch = phase_damping_channel(1.0)
        for state_vec in [
            torch.tensor([1.0, 0.0], dtype=torch.complex64),  # |0>
            torch.tensor([0.0, 1.0], dtype=torch.complex64),  # |1>
        ]:
            rho = dm_from_statevector(state_vec)
            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )
            diff = rho_new - rho
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6, "p=1 should leave |0> and |1> unchanged"

    def test_amplitude_damping_gamma0_is_identity(self):
        """Test amplitude damping channel with gamma=0 is identity."""
        ch = amplitude_damping_channel(0.0)
        state_vec = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        rho = dm_from_statevector(state_vec)
        rho_new = apply_kraus_single_qubit(
            rho, ch.kraus_operators, qubit=0, n_qubits=1
        )
        diff = rho_new - rho
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6

    def test_amplitude_damping_gamma1_on_excited_state(self):
        """Test amplitude damping with gamma=1 maps |1> to |0>."""
        ch = amplitude_damping_channel(1.0)
        # |1> = [0, 1]
        state_vec = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho = dm_from_statevector(state_vec)
        rho_new = apply_kraus_single_qubit(
            rho, ch.kraus_operators, qubit=0, n_qubits=1
        )

        # Should be |0><0| = [[1, 0], [0, 0]]
        expected = torch.zeros((2, 2), dtype=torch.complex64)
        expected[0, 0] = 1.0
        diff = rho_new - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"gamma=1 should map |1> to |0><0|, got max diff = {max_diff}"

    def test_amplitude_damping_gamma1_on_ground_state(self):
        """Test amplitude damping with gamma=1 leaves |0> unchanged."""
        ch = amplitude_damping_channel(1.0)
        state_vec = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        rho = dm_from_statevector(state_vec)
        rho_new = apply_kraus_single_qubit(
            rho, ch.kraus_operators, qubit=0, n_qubits=1
        )
        diff = rho_new - rho
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, "gamma=1 should leave |0> unchanged"

    def test_identity_channel_does_nothing(self):
        """Test identity channel leaves states unchanged."""
        ch = identity_channel()
        # Test on random states
        for _ in range(5):
            psi = torch.randn(2, dtype=torch.complex64)
            psi = psi / torch.norm(psi)
            rho = dm_from_statevector(psi)

            rho_new = apply_kraus_single_qubit(
                rho, ch.kraus_operators, qubit=0, n_qubits=1
            )

            diff = rho_new - rho
            max_diff = torch.max(torch.abs(diff)).item()
            assert max_diff < 1e-6, "Identity channel should not change state"


class TestParameterValidation:
    """Test parameter validation for channel constructors."""

    def test_depolarizing_invalid_p(self):
        """Test depolarizing channel rejects invalid p."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            depolarizing_channel(-0.1)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            depolarizing_channel(1.1)

    def test_phase_damping_invalid_p(self):
        """Test phase damping channel rejects invalid p."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            phase_damping_channel(-0.1)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            phase_damping_channel(1.1)

    def test_amplitude_damping_invalid_gamma(self):
        """Test amplitude damping channel rejects invalid gamma."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            amplitude_damping_channel(-0.1)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            amplitude_damping_channel(1.1)


class TestOldNoiseModelClasses:
    """Test the old NoiseModel classes for backward compatibility."""

    def test_depolarizing_channel_class(self):
        """Test DepolarizingChannel class."""
        from qconduit.backend.density_matrix import zero_dm_state
        from qconduit.noise import DepolarizingChannel

        ch = DepolarizingChannel(p=0.1)
        rho = zero_dm_state(n_qubits=1)
        rho_new = ch.apply_density_matrix(rho, n_qubits=1)

        # Should preserve trace
        trace = torch.trace(rho_new).real
        assert abs(trace - 1.0) < 1e-6

    def test_amplitude_damping_channel_class(self):
        """Test AmplitudeDampingChannel class."""
        from qconduit.backend.density_matrix import zero_dm_state
        from qconduit.noise import AmplitudeDampingChannel

        ch = AmplitudeDampingChannel(gamma=0.5)
        rho = zero_dm_state(n_qubits=1)
        rho_new = ch.apply_density_matrix(rho, n_qubits=1)

        trace = torch.trace(rho_new).real
        assert abs(trace - 1.0) < 1e-6

    def test_phase_damping_channel_class(self):
        """Test PhaseDampingChannel class."""
        from qconduit.backend.density_matrix import zero_dm_state
        from qconduit.noise import PhaseDampingChannel

        ch = PhaseDampingChannel(gamma=0.5)
        rho = zero_dm_state(n_qubits=1)
        rho_new = ch.apply_density_matrix(rho, n_qubits=1)

        trace = torch.trace(rho_new).real
        assert abs(trace - 1.0) < 1e-6

    def test_noise_model_apply_statevector(self):
        """Test NoiseModel.apply_statevector method."""
        from qconduit.noise import DepolarizingChannel

        ch = DepolarizingChannel(p=0.1)
        state = zero_state(n_qubits=1)
        rho_new = ch.apply_statevector(state, n_qubits=1)

        # Should be a density matrix
        assert rho_new.shape == (2, 2)
        trace = torch.trace(rho_new).real
        assert abs(trace - 1.0) < 1e-6
