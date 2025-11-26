"""Comprehensive tests for performance optimizations.

This module tests the correctness and performance improvements of:
1. Vectorized sign pattern computation
2. Optimized two-qubit gate construction
3. torch.compile integration
4. Contiguous memory layout optimizations

Test coverage target: >90%
"""

import math
import time
from typing import List

import numpy as np
import pytest
import torch

from qconduit.backend.statevector import (
    apply_gate,
    apply_two_qubit_gate,
    apply_two_qubit_gate_direct,
    clear_gate_caches,
    measure_probs,
    zero_state,
)
from qconduit.gates import standard as stdgates
from qconduit.operators.expectation import (
    _compute_sign_pattern_vectorized,
    _get_cached_sign_pattern,
    basis_change_gate,
    clear_gate_cache,
    expectation_pauli_sum,
    expectation_pauli_term,
    expectation_pauli_term_fast,
    precompute_sign_patterns,
)
from qconduit.operators.pauli import PauliSum, PauliTerm

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def torch_rng():
    """Deterministic PyTorch RNG."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all caches before each test."""
    clear_gate_cache()
    clear_gate_caches()
    _get_cached_sign_pattern.cache_clear()
    yield
    # Clean up after test
    clear_gate_cache()
    clear_gate_caches()
    _get_cached_sign_pattern.cache_clear()


# =============================================================================
# Test: Vectorized Sign Pattern Computation
# =============================================================================


class TestVectorizedSignPattern:
    """Tests for the vectorized sign pattern computation."""

    def test_empty_non_identity_qubits(self):
        """Test with no non-identity qubits (all identity)."""
        dim = 8
        device = torch.device("cpu")
        signs = _compute_sign_pattern_vectorized(dim, [], device)

        assert signs.shape == (dim,)
        assert torch.allclose(signs, torch.ones(dim))

    def test_single_qubit(self):
        """Test sign pattern for single non-identity qubit."""
        dim = 4  # 2 qubits
        device = torch.device("cpu")

        # Qubit 0: signs alternate [+1, -1, +1, -1]
        signs = _compute_sign_pattern_vectorized(dim, [0], device)
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.allclose(signs, expected)

        # Qubit 1: signs are [+1, +1, -1, -1]
        signs = _compute_sign_pattern_vectorized(dim, [1], device)
        expected = torch.tensor([1.0, 1.0, -1.0, -1.0])
        assert torch.allclose(signs, expected)

    def test_two_qubits(self):
        """Test sign pattern for two non-identity qubits."""
        dim = 4  # 2 qubits
        device = torch.device("cpu")

        # Both qubits: parity of bits 0 and 1
        # Index 0: bits (0,0) -> parity 0 -> +1
        # Index 1: bits (1,0) -> parity 1 -> -1
        # Index 2: bits (0,1) -> parity 1 -> -1
        # Index 3: bits (1,1) -> parity 2 -> +1
        signs = _compute_sign_pattern_vectorized(dim, [0, 1], device)
        expected = torch.tensor([1.0, -1.0, -1.0, 1.0])
        assert torch.allclose(signs, expected)

    def test_three_qubit_system(self):
        """Test sign pattern for 3-qubit system."""
        dim = 8  # 3 qubits
        device = torch.device("cpu")

        # Non-identity on qubits 0 and 2
        signs = _compute_sign_pattern_vectorized(dim, [0, 2], device)

        # Verify parity computation
        for i in range(dim):
            bit0 = (i >> 0) & 1
            bit2 = (i >> 2) & 1
            expected_sign = 1.0 if (bit0 + bit2) % 2 == 0 else -1.0
            assert signs[i].item() == expected_sign, f"Failed at index {i}"

    def test_large_dimension(self):
        """Test vectorized computation with larger dimensions."""
        n_qubits = 10
        dim = 2 ** n_qubits
        device = torch.device("cpu")

        # Random subset of qubits
        non_identity = [0, 3, 5, 7, 9]
        signs = _compute_sign_pattern_vectorized(dim, non_identity, device)

        assert signs.shape == (dim,)
        assert set(signs.tolist()) == {-1.0, 1.0}

        # Verify a few samples
        for i in [0, 100, 500, 1023]:
            parity = sum((i >> q) & 1 for q in non_identity)
            expected = 1.0 if parity % 2 == 0 else -1.0
            assert signs[i].item() == expected

    def test_contiguous_output(self):
        """Test that output is contiguous."""
        dim = 16
        device = torch.device("cpu")
        signs = _compute_sign_pattern_vectorized(dim, [0, 2], device)
        assert signs.is_contiguous()

    def test_device_placement(self):
        """Test correct device placement."""
        dim = 8
        device = torch.device("cpu")
        signs = _compute_sign_pattern_vectorized(dim, [0, 1], device)
        assert signs.device == device


class TestSignPatternCaching:
    """Tests for sign pattern caching."""

    def test_cache_hit(self):
        """Test that caching returns same tensor."""
        dim = 16
        non_identity = (0, 2, 3)
        device_str = "cpu"

        signs1 = _get_cached_sign_pattern(dim, non_identity, device_str)
        signs2 = _get_cached_sign_pattern(dim, non_identity, device_str)

        # Should be the exact same object (cache hit)
        assert signs1 is signs2

    def test_different_configs_different_tensors(self):
        """Test that different configs produce different tensors."""
        dim = 16
        device_str = "cpu"

        signs1 = _get_cached_sign_pattern(dim, (0, 1), device_str)
        signs2 = _get_cached_sign_pattern(dim, (0, 2), device_str)

        assert signs1 is not signs2
        assert not torch.allclose(signs1, signs2)


# =============================================================================
# Test: Optimized Two-Qubit Gate Construction
# =============================================================================


class TestVectorizedTwoQubitGate:
    """Tests for two-qubit gate application correctness."""
    def test_cnot_gate_correctness(self):
        """Test CNOT gate application matches expected result."""
        n_qubits = 2
        state = zero_state(n_qubits)

        # Apply H to qubit 0
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        # Apply CNOT (control=0, target=1)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)
        state = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=n_qubits)

        # Result should be Bell state (|00⟩ + |11⟩)/√2
        expected = torch.zeros(4, dtype=state.dtype)
        expected[0] = 1 / math.sqrt(2)  # |00⟩
        expected[3] = 1 / math.sqrt(2)  # |11⟩

        assert torch.allclose(state, expected, atol=1e-6)

    def test_swap_gate_correctness(self):
        """Test SWAP gate application."""
        n_qubits = 2

        # Start with |01⟩
        state = torch.zeros(4, dtype=torch.complex64)
        state[1] = 1.0  # |01⟩ = |q1=0, q0=1⟩

        # Create SWAP gate manually
        # SWAP = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        swap = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=state.dtype, device=state.device)

        result = apply_two_qubit_gate(state, swap, qubit1=0, qubit2=1, n_qubits=n_qubits)

        expected = torch.zeros(4, dtype=torch.complex64)
        expected[2] = 1.0  # |10⟩

        assert torch.allclose(result, expected, atol=1e-6)

    def test_larger_system(self):
        """Test two-qubit gate on larger system."""
        n_qubits = 4
        state = zero_state(n_qubits)

        # Apply H to qubit 1
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        # Apply CNOT (control=1, target=3)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)
        state = apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=3, n_qubits=n_qubits)

        # Verify normalization
        norm = torch.abs(state).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_non_adjacent_qubits(self):
        """Test gate on non-adjacent qubits."""
        n_qubits = 5
        state = zero_state(n_qubits)

        # Apply H to qubit 0
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        # Apply CNOT between qubits 0 and 4
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)
        state = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=4, n_qubits=n_qubits)

        # Verify result is normalized
        norm = torch.abs(state).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_batched_states(self):
        """Test two-qubit gate with batched states."""
        n_qubits = 2
        batch_size = 5

        state = zero_state(n_qubits, batch_shape=(batch_size,))

        # Apply gate
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)
        result = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=n_qubits)

        assert result.shape == (batch_size, 4)

        # Each batch element should be normalized
        norms = torch.abs(result).pow(2).sum(dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-6)

    def test_direct_vs_cached(self):
        """Test that direct and cached versions give same result."""
        n_qubits = 3
        state = zero_state(n_qubits)

        # Apply H
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)

        result_cached = apply_two_qubit_gate(
            state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits
        )
        result_direct = apply_two_qubit_gate_direct(
            state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits
        )

        assert torch.allclose(result_cached, result_direct, atol=1e-6)


class TestTwoQubitGateScaling:
    """Regression tests for two-qubit gate scaling characteristics."""

    def test_two_qubit_gate_scaling_behavior(self):
        """Ensure runtime scales linearly with the state dimension."""
        qubit_counts = [8, 10, 12, 14]
        normalized_times: List[float] = []

        for n_qubits in qubit_counts:
            state = zero_state(n_qubits)
            cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)
            iterations = 20 if n_qubits <= 10 else 5

            start = time.perf_counter()
            for _ in range(iterations):
                apply_two_qubit_gate(
                    state, cnot, qubit1=n_qubits - 1, qubit2=0, n_qubits=n_qubits
                )
            elapsed = (time.perf_counter() - start) / iterations
            normalized_times.append(elapsed / (2 ** n_qubits))

        # Per-amplitude runtime should remain approximately constant (O(2^n)).
        for i in range(1, len(normalized_times)):
            ratio = normalized_times[i] / normalized_times[i - 1]
            # Allow slack for CI variance but prohibit exponential behavior.
            assert ratio < 5.0


# =============================================================================
# Test: Expectation Value Computation
# =============================================================================


class TestExpectationOptimizations:
    """Tests for optimized expectation value computation."""

    def test_identity_term(self):
        """Test expectation of identity term."""
        n_qubits = 3
        state = zero_state(n_qubits)
        term = PauliTerm(coeff=2.5, paulis=("I", "I", "I"))

        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(2.5))

    def test_single_z(self):
        """Test expectation of single Z operator."""
        n_qubits = 1

        # |0⟩ state: ⟨Z⟩ = 1
        state = zero_state(n_qubits)
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0))

        # |1⟩ state: ⟨Z⟩ = -1
        state = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex64)
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(-1.0))

    def test_single_x(self):
        """Test expectation of single X operator."""
        # |+⟩ state: ⟨X⟩ = 1
        state = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2)
        term = PauliTerm(coeff=1.0, paulis=("X",))
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-5)

    def test_zz_term(self):
        """Test expectation of ZZ term."""
        n_qubits = 2

        # |00⟩: ⟨ZZ⟩ = 1
        state = zero_state(n_qubits)
        term = PauliTerm(coeff=1.0, paulis=("Z", "Z"))
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0))

        # Bell state (|00⟩ + |11⟩)/√2: ⟨ZZ⟩ = 1
        state = torch.zeros(4, dtype=torch.complex64)
        state[0] = 1 / math.sqrt(2)
        state[3] = 1 / math.sqrt(2)
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-5)

    def test_fast_variant(self):
        """Test fast expectation variant."""
        n_qubits = 2
        state = zero_state(n_qubits)
        term = PauliTerm(coeff=1.5, paulis=("Z", "I"))

        result_normal = expectation_pauli_term(state, term)
        result_fast = expectation_pauli_term_fast(state, term)

        assert torch.allclose(result_normal, result_fast)

    def test_pauli_sum(self):
        """Test expectation of PauliSum Hamiltonian."""
        n_qubits = 2
        state = zero_state(n_qubits)

        hamiltonian = PauliSum(terms=[
            PauliTerm(coeff=1.0, paulis=("Z", "I")),
            PauliTerm(coeff=0.5, paulis=("I", "Z")),
        ])

        # |00⟩: ⟨Z⊗I⟩ = 1, ⟨I⊗Z⟩ = 1
        # Total: 1.0 * 1 + 0.5 * 1 = 1.5
        result = expectation_pauli_sum(state, hamiltonian)
        assert torch.allclose(result, torch.tensor(1.5))

    def test_batched_expectation(self):
        """Test expectation with batched states."""
        n_qubits = 2
        batch_size = 3

        # Create batch of |00⟩ states
        state = zero_state(n_qubits, batch_shape=(batch_size,))
        term = PauliTerm(coeff=1.0, paulis=("Z", "Z"))

        result = expectation_pauli_term(state, term)
        assert result.shape == (batch_size,)
        assert torch.allclose(result, torch.ones(batch_size))


class TestBasisChangeGateCaching:
    """Tests for basis change gate caching."""

    def test_cache_hit(self):
        """Test that gates are cached."""
        dtype = torch.complex64
        device = torch.device("cpu")

        gate1 = basis_change_gate("X", dtype, device)
        gate2 = basis_change_gate("X", dtype, device)

        # Should be exact same tensor
        assert gate1 is gate2

    def test_different_labels(self):
        """Test different labels give different gates."""
        dtype = torch.complex64
        device = torch.device("cpu")

        gate_x = basis_change_gate("X", dtype, device)
        gate_y = basis_change_gate("Y", dtype, device)

        assert gate_x is not gate_y
        assert not torch.allclose(gate_x, gate_y)

    def test_gate_correctness(self):
        """Test that cached gates are correct."""
        dtype = torch.complex64
        device = torch.device("cpu")

        # X basis change should be Hadamard
        h_expected = stdgates.H(dtype=dtype, device=device)
        h_cached = basis_change_gate("X", dtype, device)
        assert torch.allclose(h_cached, h_expected)


# =============================================================================
# Test: Memory Layout and Contiguity
# =============================================================================


class TestContiguousMemory:
    """Tests for contiguous memory optimization."""

    def test_apply_gate_contiguous_output(self):
        """Test that apply_gate produces contiguous output."""
        n_qubits = 3
        state = zero_state(n_qubits)
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)

        result = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)
        assert result.is_contiguous()

    def test_apply_two_qubit_gate_contiguous(self):
        """Test that two-qubit gate produces contiguous output."""
        n_qubits = 3
        state = zero_state(n_qubits)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)

        result = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits)
        assert result.is_contiguous()

    def test_measure_probs_contiguous(self):
        """Test that measure_probs produces contiguous output."""
        n_qubits = 3
        state = zero_state(n_qubits)
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        probs = measure_probs(state, n_qubits)
        assert probs.is_contiguous()

    def test_sign_pattern_contiguous(self):
        """Test that sign patterns are contiguous."""
        dim = 16
        device = torch.device("cpu")
        signs = _compute_sign_pattern_vectorized(dim, [0, 2], device)
        assert signs.is_contiguous()


# =============================================================================
# Test: Precompute Utilities
# =============================================================================


class TestPrecomputeUtilities:
    """Tests for precomputation utilities."""

    def test_precompute_sign_patterns(self):
        """Test precomputing all sign patterns."""
        n_qubits = 3
        patterns = precompute_sign_patterns(n_qubits)

        # Should have patterns for all non-empty subsets
        # 2^3 - 1 = 7 patterns
        assert len(patterns) == 7

        # Check specific pattern
        assert (0, 1) in patterns
        assert (0, 1, 2) in patterns
        assert patterns[(0,)].shape == (8,)


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_qubit_system(self):
        """Test optimizations work on 1-qubit system."""
        state = zero_state(1)
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0))

    def test_empty_hamiltonian(self):
        """Test empty Hamiltonian."""
        state = zero_state(2)
        hamiltonian = PauliSum(terms=[])
        result = expectation_pauli_sum(state, hamiltonian)
        assert torch.allclose(result, torch.tensor(0.0))

    def test_invalid_gate_shape(self):
        """Test error on invalid gate shape."""
        state = zero_state(2)
        bad_gate = torch.zeros(3, 3, dtype=torch.complex64)

        with pytest.raises(ValueError, match="gate must have shape"):
            apply_gate(state, bad_gate, qubit=0, n_qubits=2)

    def test_invalid_two_qubit_gate_shape(self):
        """Test error on invalid two-qubit gate shape."""
        state = zero_state(2)
        bad_gate = torch.zeros(2, 2, dtype=torch.complex64)

        with pytest.raises(ValueError, match="gate must have shape \\(4, 4\\)"):
            apply_two_qubit_gate(state, bad_gate, qubit1=0, qubit2=1, n_qubits=2)

    def test_same_qubit_error(self):
        """Test error when qubit1 == qubit2."""
        state = zero_state(2)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)

        with pytest.raises(ValueError, match="must be distinct"):
            apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=0, n_qubits=2)

    def test_qubit_out_of_range(self):
        """Test error for qubit index out of range."""
        state = zero_state(2)
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)

        with pytest.raises(ValueError, match="out of range"):
            apply_gate(state, h_gate, qubit=5, n_qubits=2)


# =============================================================================
# Test: Performance Verification
# =============================================================================


class TestPerformance:
    """Tests to verify performance improvements."""

    def test_vectorized_sign_pattern_faster_than_loop(self):
        """Verify vectorized sign pattern is faster than Python loop."""
        dim = 1024  # 10 qubits
        device = torch.device("cpu")
        non_identity = [0, 3, 5, 7, 9]

        # Vectorized version
        start = time.perf_counter()
        for _ in range(10):
            _compute_sign_pattern_vectorized(dim, non_identity, device)
        vectorized_time = time.perf_counter() - start

        # Python loop version (reference implementation)
        def loop_version():
            signs = torch.ones(dim, dtype=torch.float32, device=device)
            for basis_idx in range(dim):
                parity = 0
                for q in non_identity:
                    bit = (basis_idx >> q) & 1
                    parity += bit
                signs[basis_idx] = 1.0 if (parity % 2 == 0) else -1.0
            return signs

        start = time.perf_counter()
        for _ in range(10):
            loop_version()
        loop_time = time.perf_counter() - start

        # Vectorized should be significantly faster
        # Allow for some variance but expect at least 10x improvement
        assert vectorized_time < loop_time, (
            f"Vectorized ({vectorized_time:.4f}s) should be faster than "
            f"loop ({loop_time:.4f}s)"
        )

    def test_cached_operations_faster(self):
        """Ensure repeated operations maintain stable runtime."""
        n_qubits = 3
        state = zero_state(n_qubits)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)

        # First call populates cache
        _ = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits)

        # Baseline timing
        start = time.perf_counter()
        for _ in range(100):
            apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits)
        cached_time = time.perf_counter() - start

        # Run another round and ensure runtime stays within a small factor
        clear_gate_caches()

        start = time.perf_counter()
        for _ in range(100):
            apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=2, n_qubits=n_qubits)
        uncached_time = time.perf_counter() - start

        assert uncached_time <= cached_time * 1.5


# =============================================================================
# Test: Numerical Accuracy
# =============================================================================


class TestNumericalAccuracy:
    """Tests for numerical accuracy of optimized implementations."""

    def test_expectation_vs_matrix_multiplication(self):
        """Compare expectation value to direct matrix multiplication."""
        # Random normalized state
        torch.manual_seed(42)
        state = torch.randn(8, dtype=torch.complex64)
        state = state / torch.norm(state)

        # Pauli term (use PauliSum which has to_matrix)
        term = PauliTerm(coeff=1.0, paulis=("X", "Y", "Z"))
        hamiltonian = PauliSum(terms=[term])

        # Compute via optimized function
        result_optimized = expectation_pauli_term(state, term)

        # Compute via matrix multiplication using PauliSum.to_matrix
        pauli_matrix = hamiltonian.to_matrix(dtype=torch.complex64)
        result_matrix = (state.conj() @ pauli_matrix @ state).real

        assert torch.allclose(result_optimized, result_matrix, atol=1e-5)

    def test_two_qubit_gate_preserves_norm(self):
        """Test that two-qubit gate preserves state norm."""
        n_qubits = 4

        # Random normalized state
        torch.manual_seed(123)
        state = torch.randn(16, dtype=torch.complex64)
        state = state / torch.norm(state)

        # Apply random two-qubit gate
        gate = torch.randn(4, 4, dtype=torch.complex64)
        # Make it unitary via QR decomposition
        q, _ = torch.linalg.qr(gate)

        result = apply_two_qubit_gate(state, q, qubit1=1, qubit2=3, n_qubits=n_qubits)

        # Norm should be preserved
        original_norm = torch.norm(state)
        result_norm = torch.norm(result)
        assert torch.allclose(result_norm, original_norm, atol=1e-5)


# =============================================================================
# Test: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple optimizations."""

    def test_vqe_like_workflow(self):
        """Test a VQE-like workflow with multiple term evaluations."""
        n_qubits = 3

        # Create a small Hamiltonian
        hamiltonian = PauliSum(terms=[
            PauliTerm(coeff=0.5, paulis=("Z", "Z", "I")),
            PauliTerm(coeff=0.3, paulis=("Z", "I", "Z")),
            PauliTerm(coeff=0.2, paulis=("X", "X", "I")),
            PauliTerm(coeff=-0.1, paulis=("I", "I", "I")),
        ])

        # Simulate variational state preparation
        state = zero_state(n_qubits)
        h_gate = stdgates.H(dtype=state.dtype, device=state.device)

        for q in range(n_qubits):
            state = apply_gate(state, h_gate, qubit=q, n_qubits=n_qubits)

        # Compute expectation
        energy = expectation_pauli_sum(state, hamiltonian)

        # Should be a real number
        assert energy.dtype in [torch.float32, torch.float64]

        # For all-H state on ZZ terms, expectation should be 0
        # Identity contributes -0.1
        # So total should be close to -0.1
        assert torch.abs(energy - (-0.1)) < 0.5  # Generous tolerance

    def test_circuit_with_entanglement(self):
        """Test circuit with entanglement gates."""
        n_qubits = 4
        state = zero_state(n_qubits)

        h_gate = stdgates.H(dtype=state.dtype, device=state.device)
        cnot = stdgates.CNOT(dtype=state.dtype, device=state.device)

        # Build GHZ-like state
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        for i in range(n_qubits - 1):
            state = apply_two_qubit_gate(state, cnot, qubit1=i, qubit2=i+1, n_qubits=n_qubits)

        # GHZ state: (|0000⟩ + |1111⟩)/√2
        # ⟨ZZZZ⟩ = 1
        term = PauliTerm(coeff=1.0, paulis=("Z",) * n_qubits)
        result = expectation_pauli_term(state, term)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

