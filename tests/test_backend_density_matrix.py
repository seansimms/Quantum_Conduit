"""Tests for density-matrix backend operations."""

import pytest
import torch
import math

from qconduit.backend.density_matrix import (
    zero_dm_state,
    dm_from_statevector,
    apply_kraus_single_qubit,
    measure_probs_dm,
    measure_expectation_z_dm,
)
from qconduit.backend.statevector import zero_state
from qconduit.gates.standard import I, X, Y, Z, H


class TestZeroDMState:
    """Tests for zero_dm_state function."""

    def test_zero_dm_state_shape_1_qubit(self):
        """Test zero_dm_state for 1 qubit has correct shape."""
        rho = zero_dm_state(n_qubits=1)
        assert rho.shape == (2, 2)
        assert rho.dtype == torch.complex64

    def test_zero_dm_state_shape_2_qubits(self):
        """Test zero_dm_state for 2 qubits has correct shape."""
        rho = zero_dm_state(n_qubits=2)
        assert rho.shape == (4, 4)
        assert rho.dtype == torch.complex64

    def test_zero_dm_state_values_1_qubit(self):
        """Test zero_dm_state for 1 qubit has correct values."""
        rho = zero_dm_state(n_qubits=1)
        # Should be |0><0| = [[1, 0], [0, 0]]
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
        assert torch.allclose(rho, expected)

    def test_zero_dm_state_values_2_qubits(self):
        """Test zero_dm_state for 2 qubits has correct values."""
        rho = zero_dm_state(n_qubits=2)
        # Should be |00><00|, so only (0,0) element is 1
        assert torch.allclose(rho[0, 0], torch.tensor(1.0 + 0.0j))
        assert torch.allclose(rho[0, 1:], torch.tensor(0.0 + 0.0j))
        assert torch.allclose(rho[1:, :], torch.tensor(0.0 + 0.0j))

    def test_zero_dm_state_batch_shape(self):
        """Test zero_dm_state with batch dimensions."""
        rho = zero_dm_state(n_qubits=1, batch_shape=(3,))
        assert rho.shape == (3, 2, 2)
        # Each batch element should be |0><0|
        for i in range(3):
            expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
            assert torch.allclose(rho[i], expected)

    def test_zero_dm_state_invalid_n_qubits(self):
        """Test zero_dm_state raises for invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            zero_dm_state(n_qubits=0)


class TestDMFromStatevector:
    """Tests for dm_from_statevector function."""

    def test_dm_from_statevector_zero_state(self):
        """Test dm_from_statevector for |0>."""
        state = zero_state(n_qubits=1)
        rho = dm_from_statevector(state)
        assert rho.shape == (2, 2)
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
        assert torch.allclose(rho, expected)

    def test_dm_from_statevector_one_state(self):
        """Test dm_from_statevector for |1>."""
        state = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho = dm_from_statevector(state)
        assert rho.shape == (2, 2)
        expected = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
        assert torch.allclose(rho, expected)

    def test_dm_from_statevector_plus_state(self):
        """Test dm_from_statevector for |+> = (|0> + |1>)/√2."""
        state = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
        rho = dm_from_statevector(state)
        assert rho.shape == (2, 2)
        # |+><+| = 1/2 * [[1, 1], [1, 1]]
        expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.complex64)
        assert torch.allclose(rho, expected)

    def test_dm_from_statevector_invalid_dtype(self):
        """Test dm_from_statevector raises for non-complex dtype."""
        state = torch.tensor([1.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="state must be complex dtype"):
            dm_from_statevector(state)

    def test_dm_from_statevector_batch(self):
        """Test dm_from_statevector with batch dimensions."""
        state = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
        rho = dm_from_statevector(state)
        assert rho.shape == (2, 2, 2)
        # First batch: |0><0|
        assert torch.allclose(rho[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64))
        # Second batch: |1><1|
        assert torch.allclose(rho[1], torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex64))


class TestApplyKrausSingleQubit:
    """Tests for apply_kraus_single_qubit function."""

    def test_apply_kraus_identity(self):
        """Test applying identity Kraus operator leaves state unchanged."""
        rho = zero_dm_state(n_qubits=1)
        I_op = I(dtype=torch.complex64, device=rho.device)
        kraus_ops = (I_op,)
        rho_out = apply_kraus_single_qubit(rho, kraus_ops, qubit=0, n_qubits=1)
        assert torch.allclose(rho_out, rho)

    def test_apply_kraus_trace_preserving(self):
        """Test that Kraus channel preserves trace."""
        rho = zero_dm_state(n_qubits=1)
        # Apply a proper Kraus channel (depolarizing-like with correct normalization)
        # Use E0 = sqrt(0.9) * I, E1 = sqrt(0.1) * X to ensure sum E_k^dagger E_k = I
        I_op = I(dtype=torch.complex64, device=rho.device)
        X_op = X(dtype=torch.complex64, device=rho.device)
        sqrt_09 = torch.sqrt(torch.tensor(0.9, dtype=torch.complex64, device=rho.device))
        sqrt_01 = torch.sqrt(torch.tensor(0.1, dtype=torch.complex64, device=rho.device))
        kraus_ops = (sqrt_09 * I_op, sqrt_01 * X_op)
        rho_out = apply_kraus_single_qubit(rho, kraus_ops, qubit=0, n_qubits=1)
        trace = rho_out.diagonal().sum().real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-6)

    def test_apply_kraus_invalid_qubit(self):
        """Test apply_kraus_single_qubit raises for invalid qubit index."""
        rho = zero_dm_state(n_qubits=1)
        I_op = I(dtype=torch.complex64, device=rho.device)
        kraus_ops = (I_op,)
        with pytest.raises(ValueError, match="qubit index"):
            apply_kraus_single_qubit(rho, kraus_ops, qubit=1, n_qubits=1)

    def test_apply_kraus_invalid_shape(self):
        """Test apply_kraus_single_qubit raises for invalid Kraus operator shape."""
        rho = zero_dm_state(n_qubits=1)
        invalid_op = torch.eye(3, dtype=torch.complex64, device=rho.device)
        kraus_ops = (invalid_op,)
        with pytest.raises(ValueError, match="must have shape \\(2, 2\\)"):
            apply_kraus_single_qubit(rho, kraus_ops, qubit=0, n_qubits=1)

    def test_apply_kraus_two_qubits(self):
        """Test applying Kraus operator to one qubit of a 2-qubit system."""
        rho = zero_dm_state(n_qubits=2)
        # Apply X to qubit 0
        X_op = X(dtype=torch.complex64, device=rho.device)
        kraus_ops = (X_op,)
        rho_out = apply_kraus_single_qubit(rho, kraus_ops, qubit=0, n_qubits=2)
        # Should flip qubit 0, so |00><00| -> |01><01|
        assert torch.allclose(rho_out[1, 1], torch.tensor(1.0 + 0.0j))
        assert torch.allclose(rho_out[0, 0], torch.tensor(0.0 + 0.0j))


class TestMeasureProbsDM:
    """Tests for measure_probs_dm function."""

    def test_measure_probs_dm_zero_state(self):
        """Test measure_probs_dm for |0><0|."""
        rho = zero_dm_state(n_qubits=1)
        probs = measure_probs_dm(rho)
        assert probs.shape == (2,)
        assert torch.allclose(probs[0], torch.tensor(1.0))
        assert torch.allclose(probs[1], torch.tensor(0.0))
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

    def test_measure_probs_dm_one_state(self):
        """Test measure_probs_dm for |1><1|."""
        state = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho = dm_from_statevector(state)
        probs = measure_probs_dm(rho)
        assert torch.allclose(probs[0], torch.tensor(0.0))
        assert torch.allclose(probs[1], torch.tensor(1.0))
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

    def test_measure_probs_dm_plus_state(self):
        """Test measure_probs_dm for |+><+|."""
        state = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
        rho = dm_from_statevector(state)
        probs = measure_probs_dm(rho)
        # |+> has equal probability for |0> and |1>
        assert torch.allclose(probs[0], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

    def test_measure_probs_dm_batch(self):
        """Test measure_probs_dm with batch dimensions."""
        rho = zero_dm_state(n_qubits=1, batch_shape=(2,))
        probs = measure_probs_dm(rho)
        assert probs.shape == (2, 2)
        for i in range(2):
            assert torch.allclose(probs[i, 0], torch.tensor(1.0))
            assert torch.allclose(probs[i, 1], torch.tensor(0.0))


class TestMeasureExpectationZDM:
    """Tests for measure_expectation_z_dm function."""

    def test_measure_expectation_z_dm_zero_state(self):
        """Test measure_expectation_z_dm for |0><0| gives <Z> = +1."""
        rho = zero_dm_state(n_qubits=1)
        expectation = measure_expectation_z_dm(rho, qubit=0, n_qubits=1)
        assert torch.allclose(expectation, torch.tensor(1.0))

    def test_measure_expectation_z_dm_one_state(self):
        """Test measure_expectation_z_dm for |1><1| gives <Z> = -1."""
        state = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        rho = dm_from_statevector(state)
        expectation = measure_expectation_z_dm(rho, qubit=0, n_qubits=1)
        assert torch.allclose(expectation, torch.tensor(-1.0))

    def test_measure_expectation_z_dm_plus_state(self):
        """Test measure_expectation_z_dm for |+><+| gives <Z> ≈ 0."""
        state = torch.tensor([1.0, 1.0], dtype=torch.complex64) / math.sqrt(2.0)
        rho = dm_from_statevector(state)
        expectation = measure_expectation_z_dm(rho, qubit=0, n_qubits=1)
        assert torch.allclose(expectation, torch.tensor(0.0), atol=1e-6)

    def test_measure_expectation_z_dm_two_qubits(self):
        """Test measure_expectation_z_dm for 2-qubit system."""
        rho = zero_dm_state(n_qubits=2)
        # For |00><00|, qubit 0 should have <Z> = +1
        expectation = measure_expectation_z_dm(rho, qubit=0, n_qubits=2)
        assert torch.allclose(expectation, torch.tensor(1.0))
        # qubit 1 should also have <Z> = +1
        expectation = measure_expectation_z_dm(rho, qubit=1, n_qubits=2)
        assert torch.allclose(expectation, torch.tensor(1.0))

    def test_measure_expectation_z_dm_invalid_qubit(self):
        """Test measure_expectation_z_dm raises for invalid qubit index."""
        rho = zero_dm_state(n_qubits=1)
        with pytest.raises(ValueError, match="qubit index"):
            measure_expectation_z_dm(rho, qubit=1, n_qubits=1)

