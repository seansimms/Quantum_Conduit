"""Tests for JSON IR round-trip functionality."""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch

from qconduit.circuit import QuantumCircuit
from qconduit.io import (
    circuit_to_json,
    dump_json_circuit,
    json_circuit_schema,
    json_to_circuit,
    load_json_circuit,
    validate_json_circuit,
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


class TestRoundTripJSON:
    """Tests for JSON round-trip functionality."""

    def test_basic_round_trip(self):
        """Test basic round-trip: circuit → JSON → circuit."""
        original = QuantumCircuit(2)
        original.add_gate("H", [0])
        original.add_gate("CNOT", [0, 1])
        original.add_gate("X", [1])

        json_obj = circuit_to_json(original)
        reconstructed = json_to_circuit(json_obj)

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

        json_obj = circuit_to_json(original)
        reconstructed = json_to_circuit(json_obj)

        state_orig = original.simulate_state()
        state_recon = reconstructed.simulate_state()

        _compare_states_up_to_global_phase(state_orig, state_recon)

    def test_file_round_trip(self):
        """Test round-trip via file I/O."""
        original = QuantumCircuit(3)
        original.add_gate("H", [0])
        original.add_gate("RY", [1], [math.pi / 4.0])
        original.add_gate("CNOT", [0, 2])
        original.add_gate("RZ", [1], [math.pi / 3.0])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            dump_json_circuit(original, temp_path)
            loaded = load_json_circuit(temp_path)

            state_orig = original.simulate_state()
            state_loaded = loaded.simulate_state()

            _compare_states_up_to_global_phase(state_orig, state_loaded)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_metadata_preservation(self):
        """Test that metadata is preserved in JSON."""
        metadata = {
            "producer": "test_suite",
            "timestamp": "2024-01-01T00:00:00Z",
            "notes": "Test circuit",
        }

        circuit = QuantumCircuit(1)
        circuit.add_gate("H", [0])

        json_obj = circuit_to_json(circuit, metadata=metadata)

        assert "metadata" in json_obj
        assert json_obj["metadata"]["producer"] == "test_suite"
        assert json_obj["metadata"]["timestamp"] == "2024-01-01T00:00:00Z"
        assert json_obj["metadata"]["notes"] == "Test circuit"

        # Metadata doesn't affect circuit reconstruction
        reconstructed = json_to_circuit(json_obj)
        assert reconstructed.n_qubits == 1
        assert len(reconstructed.ops) == 1


class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_valid_circuit(self):
        """Test that valid JSON passes validation."""
        valid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 2,
            "gates": [
                {"name": "H", "targets": [0]},
                {"name": "CNOT", "targets": [1], "controls": [0]},
            ],
        }

        # Should not raise
        validate_json_circuit(valid_json)

    def test_missing_n_qubits(self):
        """Test that missing n_qubits raises error."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "gates": [{"name": "H", "targets": [0]}],
        }

        with pytest.raises(ValueError, match="n_qubits"):
            validate_json_circuit(invalid_json)

    def test_missing_gates(self):
        """Test that missing gates raises error."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
        }

        with pytest.raises(ValueError, match="gates"):
            validate_json_circuit(invalid_json)

    def test_invalid_gate_structure(self):
        """Test that invalid gate structure raises error."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [
                {"name": "H"},  # Missing targets
            ],
        }

        with pytest.raises(ValueError, match="targets"):
            validate_json_circuit(invalid_json)

    def test_qubit_out_of_range(self):
        """Test that qubit indices out of range raise errors."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [
                {"name": "H", "targets": [5]},  # Out of range
            ],
        }

        with pytest.raises(ValueError, match="out of range"):
            validate_json_circuit(invalid_json)

    def test_invalid_params_type(self):
        """Test that invalid parameter types raise errors."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [
                {"name": "RX", "targets": [0], "params": ["not a number"]},
            ],
        }

        with pytest.raises(ValueError, match="params"):
            validate_json_circuit(invalid_json)

    def test_file_not_found(self):
        """Test that load_json_circuit raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_json_circuit("/nonexistent/path/to/file.json")

    def test_invalid_json_file(self):
        """Test that load_json_circuit handles invalid JSON."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            f.write("{ invalid json }")

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_json_circuit(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_unsupported_gate_in_json(self):
        """Test that unsupported gates raise errors."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [
                {"name": "UNSUPPORTED_GATE", "targets": [0]},
            ],
        }

        with pytest.raises(ValueError, match="Unsupported gate"):
            json_to_circuit(invalid_json)

    def test_controls_with_non_cnot(self):
        """Test that controls with non-CNOT gates raise errors."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 2,
            "gates": [
                {"name": "H", "targets": [1], "controls": [0]},
            ],
        }

        with pytest.raises(ValueError, match="Controlled gates other than CNOT"):
            json_to_circuit(invalid_json)

    def test_json_circuit_schema(self):
        """Test that json_circuit_schema returns a valid schema dict."""
        schema = json_circuit_schema()
        assert isinstance(schema, dict)
        assert "version" in schema
        assert "n_qubits" in schema
        assert "gates" in schema

    def test_validate_invalid_endian(self):
        """Test that invalid endian value raises error."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [],
            "endian": "invalid",
        }

        with pytest.raises(ValueError, match="endian"):
            validate_json_circuit(invalid_json)

    def test_validate_invalid_metadata_type(self):
        """Test that invalid metadata type raises error."""
        invalid_json = {
            "version": "qconduit-json-1.0",
            "n_qubits": 1,
            "gates": [],
            "metadata": "not a dict",
        }

        with pytest.raises(ValueError, match="metadata"):
            validate_json_circuit(invalid_json)

