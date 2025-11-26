"""Tests for QASM parsing edge cases."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.circuit import QuantumCircuit
from qconduit.io import parse_qasm_file, parse_qasm_string
from qconduit.io.utils import (
    angle_str_to_float,
    float_to_angle_str,
    gate_name_normalize,
    safe_int_list_from_str,
)


def _compare_states_up_to_global_phase(
    state1: torch.Tensor,
    state2: torch.Tensor,
    atol: float = 1e-8,
) -> None:
    """Compare two quantum states up to a global phase."""
    inner = (state1.conj() * state2).sum()
    if inner.abs() < 1e-10:
        assert torch.allclose(state1, state2, atol=atol)
    else:
        global_phase = inner / inner.abs()
        phased = state2 * global_phase.conj()
        assert torch.allclose(state1, phased, atol=atol)


class TestAngleParsing:
    """Tests for angle expression parsing."""

    def test_pi_over_2(self):
        """Test parsing pi/2."""
        angle = angle_str_to_float("pi/2")
        assert abs(angle - math.pi / 2.0) < 1e-10

    def test_pi_over_4(self):
        """Test parsing pi/4."""
        angle = angle_str_to_float("pi/4")
        assert abs(angle - math.pi / 4.0) < 1e-10

    def test_negative_pi(self):
        """Test parsing -pi."""
        angle = angle_str_to_float("-pi")
        assert abs(angle + math.pi) < 1e-10

    def test_multiple_pi(self):
        """Test parsing 2*pi/3."""
        angle = angle_str_to_float("2*pi/3")
        assert abs(angle - 2.0 * math.pi / 3.0) < 1e-10

    def test_decimal_angle(self):
        """Test parsing decimal angle."""
        angle = angle_str_to_float("0.78539816339")
        # This is approximately pi/4
        assert abs(angle - math.pi / 4.0) < 1e-3

    def test_complex_expression(self):
        """Test parsing complex expression."""
        angle = angle_str_to_float("3*pi/4")
        assert abs(angle - 3.0 * math.pi / 4.0) < 1e-10

    def test_parentheses(self):
        """Test parsing expression with parentheses."""
        angle = angle_str_to_float("(pi/2)")
        assert abs(angle - math.pi / 2.0) < 1e-10

    def test_invalid_expression(self):
        """Test that invalid expressions raise errors."""
        with pytest.raises(ValueError):
            angle_str_to_float("sin(pi/2)")  # Function calls not allowed

        with pytest.raises(ValueError):
            angle_str_to_float("pi**2")  # Exponentiation not allowed


class TestAngleFormatting:
    """Tests for angle formatting to QASM strings."""

    def test_pi_over_2_formatting(self):
        """Test formatting pi/2."""
        s = float_to_angle_str(math.pi / 2.0)
        assert s == "pi/2"

    def test_pi_formatting(self):
        """Test formatting pi."""
        s = float_to_angle_str(math.pi)
        assert s == "pi"

    def test_negative_pi_formatting(self):
        """Test formatting -pi."""
        s = float_to_angle_str(-math.pi)
        assert s == "-pi"

    def test_decimal_formatting(self):
        """Test formatting arbitrary decimal."""
        s = float_to_angle_str(0.123456789)
        # Should be a decimal string
        assert isinstance(s, str)
        # Parse back and verify
        parsed = float(s)
        assert abs(parsed - 0.123456789) < 1e-10


class TestMultipleRegisters:
    """Tests for multiple register handling."""

    def test_multiple_qregs(self):
        """Test parsing QASM with multiple qreg declarations."""
        qasm = """OPENQASM 2.0;
qreg a[1];
qreg b[1];
h a[0];
cx a[0],b[0];
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

        # Verify gates are applied to correct qubits
        # a[0] should map to global index 0, b[0] to index 1
        assert circuit.ops[0].name == "H"
        assert circuit.ops[0].qubits == (0,)
        assert circuit.ops[1].name == "CNOT"
        assert circuit.ops[1].qubits == (0, 1)

        # Verify correctness by simulation
        hand_circuit = QuantumCircuit(2)
        hand_circuit.add_gate("H", [0])
        hand_circuit.add_gate("CNOT", [0, 1])

        state_parsed = circuit.simulate_state()
        state_hand = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state_parsed, state_hand)

    def test_larger_registers(self):
        """Test parsing with larger registers."""
        qasm = """OPENQASM 2.0;
qreg q1[2];
qreg q2[2];
h q1[0];
cx q1[1],q2[0];
x q2[1];
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 4

        # q1[0] = 0, q1[1] = 1, q2[0] = 2, q2[1] = 3
        assert circuit.ops[0].qubits == (0,)
        assert circuit.ops[1].qubits == (1, 2)
        assert circuit.ops[2].qubits == (3,)


class TestCommentsAndWhitespace:
    """Tests for comment and whitespace tolerance."""

    def test_line_comments(self):
        """Test parsing QASM with line comments."""
        qasm = """OPENQASM 2.0;
// This is a comment
qreg q[2];
h q[0]; // Apply Hadamard
cx q[0],q[1]; // CNOT gate
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

    def test_block_comments(self):
        """Test parsing QASM with block comments."""
        qasm = """OPENQASM 2.0;
/* This is a block comment */
qreg q[2];
/* Another comment
   spanning multiple lines */
h q[0];
cx q[0],q[1];
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

    def test_whitespace_tolerance(self):
        """Test parsing QASM with various whitespace."""
        qasm = """OPENQASM 2.0;
qreg q[2];
h    q[0]   ;
cx   q[0]  ,  q[1]  ;
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

    def test_multiline_statements(self):
        """Test parsing multiline statements."""
        qasm = """OPENQASM 2.0;
qreg q[2];
h
q[0]
;
cx
q[0]
,
q[1]
;
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2


class TestGateVariations:
    """Tests for various gate name variations."""

    def test_lowercase_gates(self):
        """Test parsing lowercase gate names."""
        qasm = """OPENQASM 2.0;
qreg q[2];
h q[0];
x q[1];
y q[0];
z q[1];
"""

        circuit = parse_qasm_string(qasm)
        assert len(circuit.ops) == 4
        assert circuit.ops[0].name == "H"
        assert circuit.ops[1].name == "X"
        assert circuit.ops[2].name == "Y"
        assert circuit.ops[3].name == "Z"

    def test_sdg_tdg_gates(self):
        """Test parsing S dagger and T dagger gates."""
        qasm = """OPENQASM 2.0;
qreg q[1];
sdg q[0];
tdg q[0];
"""

        circuit = parse_qasm_string(qasm)
        assert len(circuit.ops) == 2
        # sdg should be RZ(-π/2), tdg should be RZ(-π/4)
        assert circuit.ops[0].name == "RZ"
        assert abs(circuit.ops[0].params[0] + math.pi / 2.0) < 1e-8
        assert circuit.ops[1].name == "RZ"
        assert abs(circuit.ops[1].params[0] + math.pi / 4.0) < 1e-8


class TestUtilsHelpers:
    """Tests for utility helper functions."""

    def test_gate_name_normalize(self):
        """Test gate name normalization."""
        assert gate_name_normalize("h") == "H"
        assert gate_name_normalize("CNOT") == "CNOT"
        assert gate_name_normalize("cx") == "CNOT"
        assert gate_name_normalize("rx") == "RX"

    def test_safe_int_list_from_str(self):
        """Test parsing integer lists from strings."""
        assert safe_int_list_from_str("[0,1,2]") == [0, 1, 2]
        assert safe_int_list_from_str("0,1,2") == [0, 1, 2]
        assert safe_int_list_from_str("[0]") == [0]

        with pytest.raises(ValueError):
            safe_int_list_from_str("[0,not_a_number]")

    def test_parse_qasm_file(self):
        """Test parsing QASM from file."""
        import os
        import tempfile

        qasm_content = """OPENQASM 2.0;
qreg q[1];
h q[0];
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
            temp_path = f.name
            f.write(qasm_content)

        try:
            circuit = parse_qasm_file(temp_path)
            assert circuit.n_qubits == 1
            assert len(circuit.ops) == 1
            assert circuit.ops[0].name == "H"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_parse_qasm_file_not_found(self):
        """Test that parse_qasm_file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_qasm_file("/nonexistent/path/to/file.qasm")

    def test_invalid_qreg_size(self):
        """Test that invalid qreg size raises error."""
        qasm = """OPENQASM 2.0;
qreg q[0];
"""

        with pytest.raises(ValueError, match="Invalid qreg size"):
            parse_qasm_string(qasm)

    def test_unknown_register(self):
        """Test that unknown register raises error."""
        qasm = """OPENQASM 2.0;
qreg q[1];
h unknown[0];
"""

        with pytest.raises(ValueError, match="Unknown register"):
            parse_qasm_string(qasm)

    def test_qubit_index_out_of_range(self):
        """Test that qubit index out of range raises error."""
        qasm = """OPENQASM 2.0;
qreg q[1];
h q[5];
"""

        with pytest.raises(ValueError, match="out of range"):
            parse_qasm_string(qasm)

