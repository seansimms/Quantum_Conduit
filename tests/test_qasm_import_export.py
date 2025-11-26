"""Tests for OpenQASM 2.0 import and export."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.circuit import QuantumCircuit
from qconduit.io import export_circuit_to_qasm, parse_qasm_string


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


class TestBasicImport:
    """Tests for basic QASM import functionality."""

    def test_basic_bell_circuit(self):
        """Test parsing a simple Bell state circuit."""
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""

        circuit = parse_qasm_string(qasm)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

        # Verify gates
        assert circuit.ops[0].name == "H"
        assert circuit.ops[0].qubits == (0,)
        assert circuit.ops[1].name == "CNOT"
        assert circuit.ops[1].qubits == (0, 1)

        # Simulate and verify state
        # Note: In this backend, qubit 0 is LSB, so H on q[0] gives (|00⟩ + |01⟩)/sqrt(2)
        # CNOT(0,1) with control=LSB and target=MSB gives (|00⟩ + |01⟩)/sqrt(2) (no change)
        # The actual behavior depends on CNOT implementation
        state = circuit.simulate_state()

        # Compare with hand-constructed equivalent (tested separately)
        hand_circuit = QuantumCircuit(2)
        hand_circuit.add_gate("H", [0])
        hand_circuit.add_gate("CNOT", [0, 1])
        expected = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state, expected)

    def test_hand_constructed_equivalent(self):
        """Test that parsed circuit matches hand-constructed circuit."""
        qasm = """OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0],q[1];
"""

        parsed_circuit = parse_qasm_string(qasm)

        # Hand-construct equivalent circuit
        hand_circuit = QuantumCircuit(2)
        hand_circuit.add_gate("H", [0])
        hand_circuit.add_gate("CNOT", [0, 1])

        # Compare states
        state_parsed = parsed_circuit.simulate_state()
        state_hand = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state_parsed, state_hand)


class TestUGateMapping:
    """Tests for u1/u2/u3 gate decomposition."""

    def test_u1_mapping(self):
        """Test u1(λ) → RZ(λ) decomposition."""
        qasm = """OPENQASM 2.0;
qreg q[1];
u1(pi/2) q[0];
"""

        circuit = parse_qasm_string(qasm)
        assert len(circuit.ops) == 1
        assert circuit.ops[0].name == "RZ"
        assert circuit.ops[0].params is not None
        assert abs(circuit.ops[0].params[0] - math.pi / 2.0) < 1e-8

        # Verify unitary equivalence
        hand_circuit = QuantumCircuit(1)
        hand_circuit.add_gate("RZ", [0], [math.pi / 2.0])

        state_parsed = circuit.simulate_state()
        state_hand = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state_parsed, state_hand)

    def test_u2_mapping(self):
        """Test u2(φ,λ) → RZ(φ) RY(π/2) RZ(λ) decomposition."""
        qasm = """OPENQASM 2.0;
qreg q[1];
u2(pi/4,pi/2) q[0];
"""

        circuit = parse_qasm_string(qasm)
        assert len(circuit.ops) == 3
        assert circuit.ops[0].name == "RZ"
        assert circuit.ops[1].name == "RY"
        assert circuit.ops[2].name == "RZ"

        # Verify parameters
        assert abs(circuit.ops[0].params[0] - math.pi / 4.0) < 1e-8
        assert abs(circuit.ops[1].params[0] - math.pi / 2.0) < 1e-8
        assert abs(circuit.ops[2].params[0] - math.pi / 2.0) < 1e-8

        # Verify unitary equivalence with hand-constructed decomposition
        hand_circuit = QuantumCircuit(1)
        hand_circuit.add_gate("RZ", [0], [math.pi / 4.0])
        hand_circuit.add_gate("RY", [0], [math.pi / 2.0])
        hand_circuit.add_gate("RZ", [0], [math.pi / 2.0])

        state_parsed = circuit.simulate_state()
        state_hand = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state_parsed, state_hand)

    def test_u3_mapping(self):
        """Test u3(θ,φ,λ) → RZ(φ) RY(θ) RZ(λ) decomposition."""
        qasm = """OPENQASM 2.0;
qreg q[1];
u3(pi/3,pi/4,pi/6) q[0];
"""

        circuit = parse_qasm_string(qasm)
        assert len(circuit.ops) == 3
        assert circuit.ops[0].name == "RZ"
        assert circuit.ops[1].name == "RY"
        assert circuit.ops[2].name == "RZ"

        # Verify parameters
        assert abs(circuit.ops[0].params[0] - math.pi / 4.0) < 1e-8
        assert abs(circuit.ops[1].params[0] - math.pi / 3.0) < 1e-8
        assert abs(circuit.ops[2].params[0] - math.pi / 6.0) < 1e-8

        # Verify unitary equivalence
        hand_circuit = QuantumCircuit(1)
        hand_circuit.add_gate("RZ", [0], [math.pi / 4.0])
        hand_circuit.add_gate("RY", [0], [math.pi / 3.0])
        hand_circuit.add_gate("RZ", [0], [math.pi / 6.0])

        state_parsed = circuit.simulate_state()
        state_hand = hand_circuit.simulate_state()

        _compare_states_up_to_global_phase(state_parsed, state_hand)


class TestExportRoundTrip:
    """Tests for export and round-trip functionality."""

    def test_export_basic_circuit(self):
        """Test exporting a basic circuit to QASM."""
        circuit = QuantumCircuit(2)
        circuit.add_gate("H", [0])
        circuit.add_gate("CNOT", [0, 1])

        qasm = export_circuit_to_qasm(circuit)

        # Check header
        assert "OPENQASM 2.0;" in qasm
        assert 'include "qelib1.inc";' in qasm
        assert "qreg q[2];" in qasm

        # Check gates (order may vary, so check both are present)
        assert "h q[0];" in qasm or "h q[1];" in qasm
        assert "cx" in qasm.lower()

    def test_round_trip_basic(self):
        """Test round-trip: circuit → QASM → circuit."""
        original = QuantumCircuit(2)
        original.add_gate("H", [0])
        original.add_gate("CNOT", [0, 1])
        original.add_gate("X", [1])

        qasm = export_circuit_to_qasm(original)
        reconstructed = parse_qasm_string(qasm)

        # Compare states
        state_orig = original.simulate_state()
        state_recon = reconstructed.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_recon)

    def test_round_trip_with_rotations(self):
        """Test round-trip with rotation gates."""
        original = QuantumCircuit(2)
        original.add_gate("RX", [0], [math.pi / 4.0])
        original.add_gate("RY", [1], [math.pi / 3.0])
        original.add_gate("RZ", [0], [math.pi / 6.0])
        original.add_gate("CNOT", [0, 1])

        qasm = export_circuit_to_qasm(original)
        reconstructed = parse_qasm_string(qasm)

        state_orig = original.simulate_state()
        state_recon = reconstructed.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_recon)


class TestUnsupportedConstructs:
    """Tests for error handling of unsupported constructs."""

    def test_unsupported_if_statement(self):
        """Test that if statements raise clear errors."""
        qasm = """OPENQASM 2.0;
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
if(c==1) x q[0];
"""

        # Parser should detect if statements and raise clear error
        with pytest.raises(ValueError, match="if statements are not supported"):
            parse_qasm_string(qasm)

    def test_unsupported_custom_gate(self):
        """Test that custom gate definitions raise errors."""
        qasm = """OPENQASM 2.0;
qreg q[1];
gate custom_gate(theta) q { rx(theta) q; }
custom_gate(pi/2) q[0];
"""

        # Custom gate definitions are not supported
        # The parser will try to parse "gate" as a gate application and fail
        with pytest.raises(ValueError):
            parse_qasm_string(qasm)

    def test_invalid_syntax(self):
        """Test that invalid syntax raises errors."""
        qasm = """OPENQASM 2.0;
qreg q[1];
h q[0  // missing closing bracket
"""

        with pytest.raises(ValueError):
            parse_qasm_string(qasm)

