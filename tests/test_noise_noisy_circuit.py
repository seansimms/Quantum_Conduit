"""Tests for noisy circuit simulation."""

from __future__ import annotations

import pytest
import torch

from qconduit.backend.density_matrix import dm_from_statevector, measure_probs_dm
from qconduit.circuit import QuantumCircuit
from qconduit.noise import (
    NoiseConfig,
    amplitude_damping_channel,
    depolarizing_channel,
    identity_channel,
    phase_damping_channel,
    sample_noisy_circuit_dm,
    simulate_noisy_circuit_dm,
)


def make_single_qubit_hadamard_circuit() -> QuantumCircuit:
    """Create a 1-qubit circuit with H gate."""
    circ = QuantumCircuit(n_qubits=1)
    circ.add_gate("H", qubits=[0])
    return circ


def make_single_qubit_identity_circuit() -> QuantumCircuit:
    """Create a 1-qubit circuit with no gates."""
    circ = QuantumCircuit(n_qubits=1)
    return circ


def make_bell_circuit() -> QuantumCircuit:
    """Create a 2-qubit Bell state circuit: H(0) -> CNOT(0,1)."""
    circ = QuantumCircuit(n_qubits=2)
    circ.add_gate("H", qubits=[0])
    circ.add_gate("CNOT", qubits=[0, 1])
    return circ


class TestIdentityChannelYieldsNoiselessResult:
    """Test that identity channel yields same result as noiseless simulation."""

    def test_identity_channel_on_hadamard(self):
        """Test identity channel on H gate circuit."""
        circ = make_single_qubit_hadamard_circuit()

        # Simulate noiselessly
        state_noiseless = circ.simulate_state()
        rho_noiseless = dm_from_statevector(state_noiseless)

        # Simulate with identity channel noise
        noise = NoiseConfig(per_qubit_channels={0: identity_channel()})
        rho_noisy = simulate_noisy_circuit_dm(circ, noise)

        # Should be identical
        diff = rho_noisy - rho_noiseless
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"Identity channel should match noiseless, got diff = {max_diff}"


class TestAmplitudeDampingOnExcitedState:
    """Test amplitude damping on excited state."""

    def test_amplitude_damping_gamma1_on_excited_state(self):
        """Test amplitude damping with gamma=1 on |1>."""
        # Use a circuit with an identity gate so noise is applied
        circ = QuantumCircuit(n_qubits=1)
        circ.add_gate("I", qubits=[0])
        initial_state = torch.tensor([0.0, 1.0], dtype=torch.complex64)  # |1>

        noise = NoiseConfig(per_qubit_channels={0: amplitude_damping_channel(gamma=1.0)})
        rho_final = simulate_noisy_circuit_dm(
            circ, noise, initial_state=initial_state, use_statevector_backend=True
        )

        # Should be |0><0|
        expected = torch.zeros((2, 2), dtype=torch.complex64)
        expected[0, 0] = 1.0

        diff = rho_final - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"Should be |0><0|, got max diff = {max_diff}"

        # Check diagonal
        diag = rho_final.diagonal().real
        assert abs(diag[0] - 1.0) < 1e-6, f"Diagonal[0] should be 1.0, got {diag[0]}"
        assert abs(diag[1] - 0.0) < 1e-6, f"Diagonal[1] should be 0.0, got {diag[1]}"


class TestPhaseDampingOnPlusState:
    """Test phase damping on |+> state."""

    def test_phase_damping_p1_on_plus(self):
        """Test phase damping with p=1 on |+>."""
        circ = make_single_qubit_hadamard_circuit()  # H|0> = |+>

        noise = NoiseConfig(per_qubit_channels={0: phase_damping_channel(p=1.0)})
        rho_final = simulate_noisy_circuit_dm(circ, noise)

        # Should be I/2: diagonal [0.5, 0.5], off-diagonals 0
        expected = torch.eye(2, dtype=torch.complex64) / 2.0

        diff = rho_final - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"Should be I/2, got max diff = {max_diff}"

        # Check diagonal
        diag = rho_final.diagonal().real
        assert abs(diag[0] - 0.5) < 1e-6, f"Diagonal[0] should be 0.5, got {diag[0]}"
        assert abs(diag[1] - 0.5) < 1e-6, f"Diagonal[1] should be 0.5, got {diag[1]}"

        # Check off-diagonals are zero
        off_diag = rho_final[0, 1]
        assert abs(off_diag) < 1e-6, f"Off-diagonal should be ~0, got {off_diag}"


class TestDepolarizingOnZeroState:
    """Test depolarizing channel on |0> state."""

    def test_depolarizing_p1_on_zero(self):
        """Test depolarizing with p=1 on |0>."""
        # Use a circuit with an identity gate so noise is applied
        circ = QuantumCircuit(n_qubits=1)
        circ.add_gate("I", qubits=[0])

        noise = NoiseConfig(per_qubit_channels={0: depolarizing_channel(p=1.0)})
        rho_final = simulate_noisy_circuit_dm(circ, noise)

        # For |0> with p=1, should be (1/3)|0><0| + (2/3)|1><1|
        expected = torch.zeros((2, 2), dtype=torch.complex64)
        expected[0, 0] = 1.0 / 3.0
        expected[1, 1] = 2.0 / 3.0

        diff = rho_final - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"Should be [[1/3,0],[0,2/3]], got max diff = {max_diff}"

        # Check diagonal
        diag = rho_final.diagonal().real
        assert abs(diag[0] - 1.0 / 3.0) < 1e-6
        assert abs(diag[1] - 2.0 / 3.0) < 1e-6


class TestTwoQubitCircuitWithLocalNoise:
    """Test 2-qubit circuit with local noise on one qubit."""

    def test_bell_circuit_with_depolarizing_on_one_qubit(self):
        """Test Bell circuit with depolarizing noise on qubit 0."""
        circ = make_bell_circuit()

        noise = NoiseConfig(per_qubit_channels={0: depolarizing_channel(p=1.0)})
        rho_final = simulate_noisy_circuit_dm(circ, noise)

        # Check basic properties
        # Trace should be 1
        trace = torch.trace(rho_final).real
        assert abs(trace - 1.0) < 1e-6, f"Trace should be 1, got {trace}"

        # Should be Hermitian
        diff = rho_final - rho_final.conj().T
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6, f"Should be Hermitian, got max diff = {max_diff}"

        # Reduced state of qubit 1 should be maximally mixed
        # Partial trace over qubit 0
        rho_1 = torch.zeros((2, 2), dtype=torch.complex64)
        for i in range(2):
            for j in range(2):
                # Trace over qubit 0: sum over |i0><j0| part
                rho_1[i, j] = rho_final[i, j] + rho_final[i + 2, j + 2]

        # Should be I/2
        expected_1 = torch.eye(2, dtype=torch.complex64) / 2.0
        diff_1 = rho_1 - expected_1
        max_diff_1 = torch.max(torch.abs(diff_1)).item()
        assert max_diff_1 < 1e-6, f"Qubit 1 reduced state should be I/2, got max diff = {max_diff_1}"


class TestSamplingConsistency:
    """Test sampling from noisy circuits."""

    def test_sampling_from_phase_damped_plus_state(self):
        """Test sampling from |+> with full phase damping."""
        circ = make_single_qubit_hadamard_circuit()
        noise = NoiseConfig(per_qubit_channels={0: phase_damping_channel(p=1.0)})

        # After full dephasing, rho_final has diagonal [0.5, 0.5]
        rho_final = simulate_noisy_circuit_dm(circ, noise)
        probs = measure_probs_dm(rho_final)
        assert abs(probs[0] - 0.5) < 1e-6
        assert abs(probs[1] - 0.5) < 1e-6

        # Sample with fixed generator
        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_noisy_circuit_dm(
            circ, noise, n_shots=5000, generator=generator
        )

        # Check shape
        assert samples.shape == (5000, 1), f"Expected shape (5000, 1), got {samples.shape}"

        # Compute empirical frequencies
        counts_0 = (samples[:, 0] == 0).sum().item()
        counts_1 = (samples[:, 0] == 1).sum().item()
        freq_0 = counts_0 / 5000.0
        freq_1 = counts_1 / 5000.0

        # Should be approximately 0.5 each (within ±0.05)
        assert abs(freq_0 - 0.5) < 0.05, f"Frequency of 0 should be ~0.5, got {freq_0}"
        assert abs(freq_1 - 0.5) < 0.05, f"Frequency of 1 should be ~0.5, got {freq_1}"

    def test_sampling_reproducibility(self):
        """Test that sampling with same generator seed is reproducible."""
        circ = make_single_qubit_hadamard_circuit()
        noise = NoiseConfig(per_qubit_channels={0: phase_damping_channel(p=0.5)})

        generator1 = torch.Generator()
        generator1.manual_seed(123)
        samples1 = sample_noisy_circuit_dm(
            circ, noise, n_shots=100, generator=generator1
        )

        generator2 = torch.Generator()
        generator2.manual_seed(123)
        samples2 = sample_noisy_circuit_dm(
            circ, noise, n_shots=100, generator=generator2
        )

        # Should be identical
        assert torch.allclose(samples1, samples2), "Samples should be identical with same seed"


class TestNoiseConfigValidation:
    """Test NoiseConfig validation."""

    def test_negative_qubit_index(self):
        """Test that negative qubit index raises ValueError."""
        ch = depolarizing_channel(0.1)
        with pytest.raises(ValueError, match="non-negative"):
            NoiseConfig(per_qubit_channels={-1: ch})

    def test_empty_noise_config(self):
        """Test that empty noise config is allowed (no noise)."""
        noise = NoiseConfig(per_qubit_channels={})
        circ = make_single_qubit_hadamard_circuit()
        # Should work fine - no noise applied
        rho = simulate_noisy_circuit_dm(circ, noise)
        # Should match noiseless result
        state_noiseless = circ.simulate_state()
        rho_noiseless = dm_from_statevector(state_noiseless)
        diff = rho - rho_noiseless
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6


class TestInitialStateHandling:
    """Test initial state handling in simulate_noisy_circuit_dm."""

    def test_initial_statevector(self):
        """Test providing initial state as statevector."""
        # Use a circuit with an identity gate so noise is applied
        circ = QuantumCircuit(n_qubits=1)
        circ.add_gate("I", qubits=[0])
        initial_state = torch.tensor([0.0, 1.0], dtype=torch.complex64)  # |1>

        noise = NoiseConfig(per_qubit_channels={0: amplitude_damping_channel(gamma=1.0)})
        rho = simulate_noisy_circuit_dm(
            circ, noise, initial_state=initial_state, use_statevector_backend=True
        )

        # Should be |0><0| after amplitude damping
        expected = torch.zeros((2, 2), dtype=torch.complex64)
        expected[0, 0] = 1.0
        diff = rho - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6

    def test_initial_density_matrix(self):
        """Test providing initial state as density matrix."""
        # Use a circuit with an identity gate so noise is applied
        circ = QuantumCircuit(n_qubits=1)
        circ.add_gate("I", qubits=[0])
        initial_rho = torch.zeros((2, 2), dtype=torch.complex64)
        initial_rho[1, 1] = 1.0  # |1><1|

        noise = NoiseConfig(per_qubit_channels={0: amplitude_damping_channel(gamma=1.0)})
        rho = simulate_noisy_circuit_dm(
            circ, noise, initial_state=initial_rho, use_statevector_backend=False
        )

        # Should be |0><0| after amplitude damping
        expected = torch.zeros((2, 2), dtype=torch.complex64)
        expected[0, 0] = 1.0
        diff = rho - expected
        max_diff = torch.max(torch.abs(diff)).item()
        assert max_diff < 1e-6

