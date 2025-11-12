"""Tests for statevector backend operations."""

import pytest
import torch
import math
import qconduit as qc


class TestZeroState:
    """Tests for zero_state function."""

    def test_zero_state_shape_1_qubit(self):
        """Test zero_state for 1 qubit has correct shape."""
        state = qc.zero_state(n_qubits=1)
        assert state.shape == (2,)
        assert state.dtype == torch.complex64

    def test_zero_state_shape_2_qubits(self):
        """Test zero_state for 2 qubits has correct shape."""
        state = qc.zero_state(n_qubits=2)
        assert state.shape == (4,)
        assert state.dtype == torch.complex64

    def test_zero_state_amplitude(self):
        """Test zero_state has |0...0⟩ amplitude = 1."""
        state = qc.zero_state(n_qubits=2)
        assert torch.allclose(state[0], torch.tensor(1.0 + 0.0j))
        assert torch.allclose(state[1:], torch.tensor(0.0 + 0.0j))

    def test_zero_state_batch_shape(self):
        """Test zero_state with batch dimensions."""
        state = qc.zero_state(n_qubits=1, batch_shape=(3, 4))
        assert state.shape == (3, 4, 2)
        # Each batch element should be |0⟩
        for i in range(3):
            for j in range(4):
                assert torch.allclose(state[i, j, 0], torch.tensor(1.0 + 0.0j))
                assert torch.allclose(state[i, j, 1], torch.tensor(0.0 + 0.0j))

    def test_zero_state_invalid_n_qubits(self):
        """Test zero_state raises for invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            qc.zero_state(n_qubits=0)

    def test_zero_state_device_parameter(self):
        """Test zero_state accepts device parameter."""
        state = qc.zero_state(n_qubits=1, device="sv_cpu")
        assert state.device.type == "cpu"


class TestApplyGate:
    """Tests for apply_gate function."""

    def test_apply_x_to_zero_yields_one(self):
        """Test applying X to |0⟩ yields |1⟩."""
        state = qc.zero_state(n_qubits=1)
        x_gate = qc.X()
        result = qc.apply_gate(state, x_gate, qubit=0, n_qubits=1)

        expected = torch.zeros(2, dtype=torch.complex64)
        expected[1] = 1.0  # |1⟩
        assert torch.allclose(result, expected)

    def test_apply_h_to_zero(self):
        """Test applying H to |0⟩ yields (|0⟩ + |1⟩)/√2."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        result = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex64)
        assert torch.allclose(result, expected)

    def test_apply_gate_2_qubits_qubit0(self):
        """Test applying gate to qubit 0 in 2-qubit state."""
        state = qc.zero_state(n_qubits=2)  # |00⟩
        x_gate = qc.X()
        result = qc.apply_gate(state, x_gate, qubit=0, n_qubits=2)

        # Should yield |01⟩ (qubit 0 is LSB)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[1] = 1.0  # |01⟩
        assert torch.allclose(result, expected)

    def test_apply_gate_2_qubits_qubit1(self):
        """Test applying gate to qubit 1 in 2-qubit state."""
        state = qc.zero_state(n_qubits=2)  # |00⟩
        x_gate = qc.X()
        result = qc.apply_gate(state, x_gate, qubit=1, n_qubits=2)

        # Should yield |10⟩ (qubit 1 is MSB)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[2] = 1.0  # |10⟩
        assert torch.allclose(result, expected)

    def test_apply_gate_batch(self):
        """Test apply_gate works with batched states."""
        state = qc.zero_state(n_qubits=1, batch_shape=(3,))
        x_gate = qc.X()
        result = qc.apply_gate(state, x_gate, qubit=0, n_qubits=1)

        assert result.shape == (3, 2)
        # Each batch should be |1⟩
        expected = torch.zeros(2, dtype=torch.complex64)
        expected[1] = 1.0
        for i in range(3):
            assert torch.allclose(result[i], expected)

    def test_apply_gate_invalid_gate_shape(self):
        """Test apply_gate raises for invalid gate shape."""
        state = qc.zero_state(n_qubits=1)
        invalid_gate = torch.eye(3, dtype=torch.complex64)
        with pytest.raises(ValueError, match="gate must have shape"):
            qc.apply_gate(state, invalid_gate, qubit=0, n_qubits=1)

    def test_apply_gate_invalid_qubit_index(self):
        """Test apply_gate raises for invalid qubit index."""
        state = qc.zero_state(n_qubits=1)
        x_gate = qc.X()
        with pytest.raises(ValueError, match="qubit index"):
            qc.apply_gate(state, x_gate, qubit=1, n_qubits=1)

    def test_apply_gate_infers_n_qubits(self):
        """Test apply_gate infers n_qubits from state dimension."""
        state = qc.zero_state(n_qubits=2)
        x_gate = qc.X()
        result = qc.apply_gate(state, x_gate, qubit=0)
        # Should work without explicit n_qubits
        assert result.shape == (4,)


class TestApplyTwoQubitGate:
    """Tests for apply_two_qubit_gate function."""

    def test_apply_cnot_control_first(self):
        """Test CNOT with control on qubit 1, target on qubit 0."""
        # Start with |10⟩ (index 2)
        state = torch.zeros(4, dtype=torch.complex64)
        state[2] = 1.0  # |10⟩

        cnot = qc.CNOT(control_first=True)
        result = qc.apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=0, n_qubits=2)

        # Should yield |11⟩ (index 3)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[3] = 1.0  # |11⟩
        assert torch.allclose(result, expected)

    def test_apply_cnot_control_second(self):
        """Test CNOT with control on qubit 0, target on qubit 1."""
        # Start with |01⟩ (index 1)
        state = torch.zeros(4, dtype=torch.complex64)
        state[1] = 1.0  # |01⟩

        cnot = qc.CNOT(control_first=False)
        result = qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=2)

        # Should yield |11⟩ (index 3)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[3] = 1.0  # |11⟩
        assert torch.allclose(result, expected)

    def test_apply_cnot_swapped_qubits(self):
        """Test CNOT works when qubit1 > qubit2."""
        # Start with |01⟩ (index 1), apply CNOT with qubit1=1, qubit2=0
        state = torch.zeros(4, dtype=torch.complex64)
        state[1] = 1.0  # |01⟩

        cnot = qc.CNOT(control_first=True)
        # qubit1=1, qubit2=0 means control=1, target=0
        result = qc.apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=0, n_qubits=2)

        # |01⟩ with control=1 (MSB) and target=0 (LSB) should stay |01⟩ (control is 0)
        expected = torch.zeros(4, dtype=torch.complex64)
        expected[1] = 1.0  # |01⟩
        assert torch.allclose(result, expected)

    def test_apply_two_qubit_gate_invalid_gate_shape(self):
        """Test apply_two_qubit_gate raises for invalid gate shape."""
        state = qc.zero_state(n_qubits=2)
        invalid_gate = torch.eye(2, dtype=torch.complex64)
        with pytest.raises(ValueError, match="gate must have shape"):
            qc.apply_two_qubit_gate(state, invalid_gate, qubit1=0, qubit2=1, n_qubits=2)

    def test_apply_two_qubit_gate_same_qubits(self):
        """Test apply_two_qubit_gate raises for same qubit indices."""
        state = qc.zero_state(n_qubits=2)
        cnot = qc.CNOT()
        with pytest.raises(ValueError, match="must be distinct"):
            qc.apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=0, n_qubits=2)

    def test_apply_two_qubit_gate_batch(self):
        """Test apply_two_qubit_gate works with batched states."""
        state = qc.zero_state(n_qubits=2, batch_shape=(2,))
        # Set first batch to |10⟩, second to |00⟩
        state[0, 2] = 1.0
        state[0, 0] = 0.0

        cnot = qc.CNOT(control_first=True)
        result = qc.apply_two_qubit_gate(state, cnot, qubit1=1, qubit2=0, n_qubits=2)

        assert result.shape == (2, 4)
        # First batch should be |11⟩
        assert torch.allclose(result[0, 3], torch.tensor(1.0 + 0.0j))
        # Second batch should remain |00⟩
        assert torch.allclose(result[1, 0], torch.tensor(1.0 + 0.0j))


class TestMeasureExpectationZ:
    """Tests for measure_expectation_z function."""

    def test_measure_z_zero_state(self):
        """Test ⟨Z⟩ for |0⟩ is +1."""
        state = qc.zero_state(n_qubits=1)
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(1.0))

    def test_measure_z_one_state(self):
        """Test ⟨Z⟩ for |1⟩ is -1."""
        state = qc.zero_state(n_qubits=1)
        x_gate = qc.X()
        state = qc.apply_gate(state, x_gate, qubit=0, n_qubits=1)  # |1⟩
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(-1.0))

    def test_measure_z_plus_state(self):
        """Test ⟨Z⟩ for (|0⟩ + |1⟩)/√2 is 0."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)  # (|0⟩ + |1⟩)/√2
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert torch.allclose(z_exp, torch.tensor(0.0), atol=1e-6)

    def test_measure_z_2_qubits(self):
        """Test ⟨Z⟩ for specific qubit in 2-qubit state."""
        state = qc.zero_state(n_qubits=2)  # |00⟩
        # Apply X to qubit 0: |00⟩ -> |01⟩
        x_gate = qc.X()
        state = qc.apply_gate(state, x_gate, qubit=0, n_qubits=2)

        z_exp_0 = qc.measure_expectation_z(state, qubit=0, n_qubits=2)
        z_exp_1 = qc.measure_expectation_z(state, qubit=1, n_qubits=2)

        # Qubit 0 is |1⟩, qubit 1 is |0⟩
        assert torch.allclose(z_exp_0, torch.tensor(-1.0))
        assert torch.allclose(z_exp_1, torch.tensor(1.0))

    def test_measure_z_batch(self):
        """Test measure_expectation_z works with batched states."""
        state = qc.zero_state(n_qubits=1, batch_shape=(3,))
        z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
        assert z_exp.shape == (3,)
        assert torch.allclose(z_exp, torch.tensor(1.0))

    def test_measure_z_invalid_qubit_index(self):
        """Test measure_expectation_z raises for invalid qubit index."""
        state = qc.zero_state(n_qubits=1)
        with pytest.raises(ValueError, match="qubit index"):
            qc.measure_expectation_z(state, qubit=1, n_qubits=1)


class TestMeasureProbs:
    """Tests for measure_probs function."""

    def test_measure_probs_zero_state(self):
        """Test probabilities for |0⟩."""
        state = qc.zero_state(n_qubits=1)
        probs = qc.measure_probs(state, n_qubits=1)
        assert probs.shape == (2,)
        assert torch.allclose(probs[0], torch.tensor(1.0))
        assert torch.allclose(probs[1], torch.tensor(0.0))

    def test_measure_probs_hadamard_state(self):
        """Test probabilities for H|0⟩."""
        state = qc.zero_state(n_qubits=1)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)
        probs = qc.measure_probs(state, n_qubits=1)

        assert probs.shape == (2,)
        # Should be approximately 0.5, 0.5
        assert torch.allclose(probs[0], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(probs[1], torch.tensor(0.5), atol=1e-6)

    def test_measure_probs_normalized(self):
        """Test probabilities sum to 1."""
        state = qc.zero_state(n_qubits=2)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=2)
        probs = qc.measure_probs(state, n_qubits=2)

        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_measure_probs_batch(self):
        """Test measure_probs works with batched states."""
        state = qc.zero_state(n_qubits=1, batch_shape=(2,))
        probs = qc.measure_probs(state, n_qubits=1)
        assert probs.shape == (2, 2)
        # Each batch should have probability 1 at index 0
        assert torch.allclose(probs[:, 0], torch.ones(2))
        assert torch.allclose(probs[:, 1], torch.zeros(2))

    def test_measure_probs_2_qubits(self):
        """Test probabilities for 2-qubit state."""
        state = qc.zero_state(n_qubits=2)  # |00⟩
        probs = qc.measure_probs(state, n_qubits=2)
        assert probs.shape == (4,)
        assert torch.allclose(probs[0], torch.tensor(1.0))
        assert torch.allclose(probs[1:], torch.tensor(0.0))

