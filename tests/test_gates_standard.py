"""Tests for standard quantum gates."""

import pytest
import torch
import math
import qconduit as qc
from qconduit.gates.standard import is_unitary


class TestStaticGates:
    """Tests for static single-qubit gates."""

    def test_i_gate_shape_and_dtype(self):
        """Test I gate has correct shape and dtype."""
        gate = qc.I()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_i_gate_matrix(self):
        """Test I gate is identity matrix."""
        gate = qc.I()
        expected = torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_x_gate_shape_and_dtype(self):
        """Test X gate has correct shape and dtype."""
        gate = qc.X()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_x_gate_matrix(self):
        """Test X gate has correct matrix."""
        gate = qc.X()
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_y_gate_shape_and_dtype(self):
        """Test Y gate has correct shape and dtype."""
        gate = qc.Y()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_y_gate_matrix(self):
        """Test Y gate has correct matrix."""
        gate = qc.Y()
        expected = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_z_gate_shape_and_dtype(self):
        """Test Z gate has correct shape and dtype."""
        gate = qc.Z()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_z_gate_matrix(self):
        """Test Z gate has correct matrix."""
        gate = qc.Z()
        expected = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_h_gate_shape_and_dtype(self):
        """Test H gate has correct shape and dtype."""
        gate = qc.H()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_h_gate_matrix(self):
        """Test H gate has correct matrix."""
        gate = qc.H()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        expected = torch.tensor(
            [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=torch.complex64
        )
        assert torch.allclose(gate, expected)

    def test_s_gate_shape_and_dtype(self):
        """Test S gate has correct shape and dtype."""
        gate = qc.S()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_s_gate_matrix(self):
        """Test S gate has correct matrix."""
        gate = qc.S()
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_t_gate_shape_and_dtype(self):
        """Test T gate has correct shape and dtype."""
        gate = qc.T()
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_t_gate_matrix(self):
        """Test T gate has correct matrix."""
        import cmath
        gate = qc.T()
        exp_i_pi_4 = cmath.exp(1.0j * math.pi / 4.0)
        expected = torch.tensor([[1.0, 0.0], [0.0, exp_i_pi_4]], dtype=torch.complex64)
        assert torch.allclose(gate, expected)

    def test_all_static_gates_unitary(self):
        """Test all static gates are unitary."""
        gates = [qc.I(), qc.X(), qc.Y(), qc.Z(), qc.H(), qc.S(), qc.T()]
        for gate in gates:
            assert is_unitary(gate), f"Gate {gate} is not unitary"

    def test_gate_device_and_dtype_parameters(self):
        """Test gates accept device and dtype parameters."""
        gate = qc.X(dtype=torch.complex128, device=torch.device("cpu"))
        assert gate.dtype == torch.complex128
        assert gate.device.type == "cpu"


class TestCNOT:
    """Tests for CNOT gate."""

    def test_cnot_shape_and_dtype(self):
        """Test CNOT gate has correct shape and dtype."""
        gate = qc.CNOT()
        assert gate.shape == (4, 4)
        assert gate.dtype == torch.complex64

    def test_cnot_control_first_matrix(self):
        """Test CNOT with control_first=True has correct matrix."""
        gate = qc.CNOT(control_first=True)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.complex64,
        )
        assert torch.allclose(gate, expected)

    def test_cnot_control_second_matrix(self):
        """Test CNOT with control_first=False has correct matrix."""
        gate = qc.CNOT(control_first=False)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.complex64,
        )
        assert torch.allclose(gate, expected)

    def test_cnot_unitary(self):
        """Test CNOT gate is unitary."""
        gate = qc.CNOT()
        assert is_unitary(gate)

    def test_cnot_acts_on_basis_states(self):
        """Test CNOT acts correctly on computational basis states."""
        # Test CNOT|10⟩ = |11⟩ (control first, target second)
        # |10⟩ is index 2 in little-endian (qubit 0 = LSB)
        state = torch.zeros(4, dtype=torch.complex64)
        state[2] = 1.0  # |10⟩

        cnot = qc.CNOT(control_first=True)
        result = cnot @ state

        expected = torch.zeros(4, dtype=torch.complex64)
        expected[3] = 1.0  # |11⟩
        assert torch.allclose(result, expected)


class TestRotationGates:
    """Tests for parametric rotation gates."""

    def test_rx_shape_and_dtype(self):
        """Test RX gate has correct shape and dtype."""
        gate = qc.RX(0.5)
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_ry_shape_and_dtype(self):
        """Test RY gate has correct shape and dtype."""
        gate = qc.RY(0.5)
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_rz_shape_and_dtype(self):
        """Test RZ gate has correct shape and dtype."""
        gate = qc.RZ(0.5)
        assert gate.shape == (2, 2)
        assert gate.dtype == torch.complex64

    def test_rx_zero_angle(self):
        """Test RX(0) equals identity."""
        gate = qc.RX(0.0)
        identity = qc.I()
        assert torch.allclose(gate, identity, atol=1e-6)

    def test_ry_zero_angle(self):
        """Test RY(0) equals identity."""
        gate = qc.RY(0.0)
        identity = qc.I()
        assert torch.allclose(gate, identity, atol=1e-6)

    def test_rz_zero_angle(self):
        """Test RZ(0) equals identity."""
        gate = qc.RZ(0.0)
        identity = qc.I()
        assert torch.allclose(gate, identity, atol=1e-6)

    def test_rx_pi_angle(self):
        """Test RX(π) is approximately -iX (up to global phase)."""
        gate = qc.RX(math.pi)
        x_gate = qc.X()
        # RX(π) = -iX, so |RX(π)| should match |X|
        assert torch.allclose(torch.abs(gate), torch.abs(x_gate), atol=1e-6)

    def test_ry_pi_angle(self):
        """Test RY(π) is approximately Y (up to global phase)."""
        gate = qc.RY(math.pi)
        y_gate = qc.Y()
        # RY(π) = -iY, so |RY(π)| should match |Y|
        assert torch.allclose(torch.abs(gate), torch.abs(y_gate), atol=1e-6)

    def test_rz_pi_angle(self):
        """Test RZ(π) is approximately Z (up to global phase)."""
        gate = qc.RZ(math.pi)
        z_gate = qc.Z()
        # RZ(π) = -iZ, so |RZ(π)| should match |Z|
        assert torch.allclose(torch.abs(gate), torch.abs(z_gate), atol=1e-6)

    def test_rotation_gates_unitary(self):
        """Test all rotation gates are unitary."""
        angles = [0.1, 0.5, 1.0, math.pi / 2, math.pi]
        for angle in angles:
            assert is_unitary(qc.RX(angle)), f"RX({angle}) is not unitary"
            assert is_unitary(qc.RY(angle)), f"RY({angle}) is not unitary"
            assert is_unitary(qc.RZ(angle)), f"RZ({angle}) is not unitary"

    def test_rotation_gates_accept_tensor(self):
        """Test rotation gates accept tensor input."""
        theta = torch.tensor(0.5)
        gate = qc.RX(theta)
        assert gate.shape == (2, 2)


class TestIsUnitary:
    """Tests for is_unitary helper function."""

    def test_identity_is_unitary(self):
        """Test identity matrix is unitary."""
        identity = torch.eye(3, dtype=torch.complex64)
        assert is_unitary(identity)

    def test_non_square_not_unitary(self):
        """Test non-square matrix is not unitary."""
        matrix = torch.randn(2, 3, dtype=torch.complex64)
        assert not is_unitary(matrix)

    def test_random_matrix_not_unitary(self):
        """Test random matrix is not unitary."""
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        assert not is_unitary(matrix)

    def test_batched_unitary(self):
        """Test is_unitary works with batched matrices."""
        # Create batch of identity matrices
        batch = torch.eye(2, dtype=torch.complex64).unsqueeze(0).repeat(3, 1, 1)
        assert is_unitary(batch)

