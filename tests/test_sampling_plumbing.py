"""Comprehensive tests for sampling and measurement utilities (G2).

This test module validates:
1. Deterministic sampling from trivial distributions
2. Sampling from |+⟩, |Φ+⟩, and simple mixed states
3. Circuit-based sampling
4. Histogram utilities (bitstring_counts, counts_to_probs, marginals)
5. KL divergence
6. Reproducibility
"""

import pytest
import torch
import math
import qconduit as qc
from qconduit.sampling import (
    sample_from_probs,
    sample_bitstrings_state,
    sample_bitstrings_dm,
    sample_bitstrings_circuit,
    bitstring_counts,
    counts_to_probs,
    kl_divergence,
    marginalize_probs,
)
from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate
from qconduit.backend.density_matrix import dm_from_statevector
from qconduit.circuit import QuantumCircuit


class TestDeterministicSampling:
    """Test deterministic sampling from trivial distributions."""

    def test_sample_from_probs_all_zero(self):
        """Test sampling from probs = [1,0,...,0] always produces all-zero bitstrings."""
        # 1-qubit case
        probs = torch.tensor([1.0, 0.0])
        samples = sample_from_probs(probs, n_qubits=1, n_shots=100)
        assert samples.shape == (100, 1)
        assert samples.dtype == torch.int64
        assert torch.all(samples == 0)

        # 2-qubit case
        probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
        samples = sample_from_probs(probs, n_qubits=2, n_shots=50)
        assert samples.shape == (50, 2)
        assert torch.all(samples == 0)

    def test_sample_from_probs_shape_and_dtype(self):
        """Test sample_from_probs produces correct shape and dtype."""
        probs = torch.tensor([0.5, 0.5])
        samples = sample_from_probs(probs, n_qubits=1, n_shots=10)
        assert samples.shape == (10, 1)
        assert samples.dtype == torch.int64
        assert torch.all((samples == 0) | (samples == 1))


class TestSamplingFromKnownStates:
    """Test sampling from |+⟩, |Φ+⟩, and simple mixed states."""

    def test_sample_bitstrings_state_plus_state(self):
        """Test sampling from |+⟩ with large n_shots."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=1)  # |+⟩

        # Use fixed seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_state(state, n_qubits=1, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 1)
        # Count occurrences
        counts = torch.bincount(samples.squeeze())
        # Should be approximately 50/50
        assert abs(counts[0].item() - 2500) < 200  # Within ~4% tolerance
        assert abs(counts[1].item() - 2500) < 200

    def test_sample_bitstrings_state_bell_state(self):
        """Test sampling from Bell state |Φ+⟩."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)

        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_state(state, n_qubits=2, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 2)
        # Count bitstrings
        counts = bitstring_counts(samples)
        # Bell state should have ~50% |00⟩ and ~50% |11⟩
        assert "00" in counts
        assert "11" in counts
        assert counts["00"] + counts["11"] > 4800  # Should be most of the samples
        # |01⟩ and |10⟩ should be rare
        assert counts.get("01", 0) < 100
        assert counts.get("10", 0) < 100

    def test_sample_bitstrings_dm_plus_state(self):
        """Test sampling from density matrix of |+⟩."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=1)
        rho = dm_from_statevector(state)

        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_dm(rho, n_qubits=1, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 1)
        counts = torch.bincount(samples.squeeze())
        assert abs(counts[0].item() - 2500) < 200
        assert abs(counts[1].item() - 2500) < 200

    def test_sample_bitstrings_dm_bell_state(self):
        """Test sampling from density matrix of Bell state."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=2)
        cnot = qc.CNOT(control_first=True)
        state = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)
        rho = dm_from_statevector(state)

        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_dm(rho, n_qubits=2, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 2)
        counts = bitstring_counts(samples)
        assert counts["00"] + counts["11"] > 4800


class TestCircuitBasedSampling:
    """Test circuit-based sampling."""

    def test_sample_bitstrings_circuit_plus_circuit(self):
        """Test sampling from a |+⟩ circuit."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_circuit(circuit, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 1)
        counts = torch.bincount(samples.squeeze())
        assert abs(counts[0].item() - 2500) < 200
        assert abs(counts[1].item() - 2500) < 200

    def test_sample_bitstrings_circuit_bell_circuit(self):
        """Test sampling from a Bell circuit."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        generator = torch.Generator()
        generator.manual_seed(42)
        samples = sample_bitstrings_circuit(circuit, n_shots=5000, generator=generator)

        assert samples.shape == (5000, 2)
        counts = bitstring_counts(samples)
        assert counts["00"] + counts["11"] > 4800

    def test_sample_bitstrings_circuit_consistency(self):
        """Test that circuit sampling matches statevector sampling."""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.add_gate("H", [0])

        # Sample from circuit
        generator1 = torch.Generator()
        generator1.manual_seed(42)
        samples_circuit = sample_bitstrings_circuit(circuit, n_shots=1000, generator=generator1)

        # Sample from simulated state
        state = circuit.simulate_state()
        generator2 = torch.Generator()
        generator2.manual_seed(42)
        samples_state = sample_bitstrings_state(state, n_qubits=1, n_shots=1000, generator=generator2)

        # Should produce same samples with same seed
        assert torch.allclose(samples_circuit, samples_state)


class TestHistogramUtilities:
    """Test histogram utilities."""

    def test_bitstring_counts(self):
        """Test bitstring_counts on manually constructed samples."""
        # Build samples: 00, 01, 01, 11
        samples = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 1]], dtype=torch.int64)
        counts = bitstring_counts(samples)

        assert counts["00"] == 1
        assert counts["01"] == 2
        assert counts["11"] == 1
        assert "10" not in counts or counts.get("10", 0) == 0

    def test_counts_to_probs(self):
        """Test counts_to_probs normalization."""
        counts = {"00": 2, "01": 3, "11": 5}
        probs = counts_to_probs(counts)

        total = sum(counts.values())
        assert abs(probs["00"] - 2.0 / total) < 1e-6
        assert abs(probs["01"] - 3.0 / total) < 1e-6
        assert abs(probs["11"] - 5.0 / total) < 1e-6
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_marginalize_probs_uniform_2_qubit(self):
        """Test marginalization for 2-qubit uniform distribution."""
        # Uniform distribution over 2 qubits: all probabilities = 0.25
        probs = torch.ones(4) / 4.0
        n_qubits = 2

        # Marginalize over qubit 0
        marg_0 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0])
        assert marg_0.shape == (2,)
        assert torch.allclose(marg_0, torch.tensor([0.5, 0.5]), atol=1e-6)

        # Marginalize over qubit 1
        marg_1 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[1])
        assert marg_1.shape == (2,)
        assert torch.allclose(marg_1, torch.tensor([0.5, 0.5]), atol=1e-6)

    def test_marginalize_probs_bell_state(self):
        """Test marginalization for Bell state distribution."""
        # Bell state: 0.5 for |00⟩, 0.5 for |11⟩
        probs = torch.tensor([0.5, 0.0, 0.0, 0.5])
        n_qubits = 2

        # Marginal over either qubit should be uniform
        marg_0 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[0])
        assert torch.allclose(marg_0, torch.tensor([0.5, 0.5]), atol=1e-6)

        marg_1 = marginalize_probs(probs, n_qubits=n_qubits, qubits_to_keep=[1])
        assert torch.allclose(marg_1, torch.tensor([0.5, 0.5]), atol=1e-6)


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_divergence_same_distribution(self):
        """Test KL(p||p) ≈ 0."""
        p = {"00": 0.5, "01": 0.3, "11": 0.2}
        kl = kl_divergence(p, p)
        assert abs(kl) < 1e-6

    def test_kl_divergence_different_distributions(self):
        """Test KL(p||q) for different distributions."""
        p = {"0": 0.5, "1": 0.5}
        q = {"0": 1.0, "1": 0.0}

        kl = kl_divergence(p, q)
        # KL should be positive and finite
        assert kl > 0
        assert math.isfinite(kl)

        # Manual computation: KL = 0.5 * log(0.5/1.0) + 0.5 * log(0.5/epsilon)
        # With epsilon clamping, this should be approximately log(2) ≈ 0.693
        # But with epsilon clamping, it will be different
        assert kl > 0.1  # Should be positive

    def test_kl_divergence_reference_computation(self):
        """Test KL divergence matches reference computation."""
        import math

        p = {"0": 0.5, "1": 0.5}
        q = {"0": 0.8, "1": 0.2}

        kl = kl_divergence(p, q, epsilon=1e-12)

        # Reference: KL = 0.5 * log(0.5/0.8) + 0.5 * log(0.5/0.2)
        # = 0.5 * log(0.625) + 0.5 * log(2.5)
        ref_kl = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)

        assert abs(kl - ref_kl) < 1e-6


class TestReproducibility:
    """Test reproducibility of sampling."""

    def test_sampling_reproducibility(self):
        """Test that sampling is bit-for-bit reproducible with fixed seed."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=1)

        # First run
        generator1 = torch.Generator()
        generator1.manual_seed(12345)
        samples1 = sample_bitstrings_state(state, n_qubits=1, n_shots=100, generator=generator1)

        # Second run with same seed
        generator2 = torch.Generator()
        generator2.manual_seed(12345)
        samples2 = sample_bitstrings_state(state, n_qubits=1, n_shots=100, generator=generator2)

        # Should be identical
        assert torch.allclose(samples1, samples2)

    def test_sampling_reproducibility_circuit(self):
        """Test circuit sampling reproducibility."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        generator1 = torch.Generator()
        generator1.manual_seed(999)
        samples1 = sample_bitstrings_circuit(circuit, n_shots=50, generator=generator1)

        generator2 = torch.Generator()
        generator2.manual_seed(999)
        samples2 = sample_bitstrings_circuit(circuit, n_shots=50, generator=generator2)

        assert torch.allclose(samples1, samples2)


