"""JSON IR schema definition and validation for quantum circuits.

This module defines a canonical JSON interchange format for quantum circuits.
The schema is designed to be compact, portable, and sufficient for representing
textbook quantum circuits with standard gates.

Schema Structure:
    {
        "version": "qconduit-json-1.0",
        "n_qubits": <integer>,
        "gates": [
            {
                "name": <string>,
                "targets": [<integer>, ...],
                "controls": [<integer>, ...],  # optional, for controlled gates
                "params": [<float>, ...],      # optional, angles in radians
                "label": <string>,              # optional, user metadata
            },
            ...
        ],
        "metadata": {                           # optional
            "producer": <string>,
            "timestamp": <string>,
            "notes": <string>,
            ...
        },
        "endian": <string>                     # optional, default "little"
    }

Qubit ordering convention:
    - Default endianness is "little" (qubit 0 is LSB)
    - This matches the statevector backend convention
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def json_circuit_schema() -> dict:
    """
    Return the JSON schema (as a Python dict) for the JSON IR format.

    This is a structural schema description, not a full JSON Schema validator.
    The schema defines the expected structure for circuit interchange.

    Returns
    -------
    dict
        Schema description with field definitions and constraints.
    """
    return {
        "version": {
            "type": "string",
            "description": "Schema version identifier, e.g., 'qconduit-json-1.0'",
            "required": True,
        },
        "n_qubits": {
            "type": "integer",
            "description": "Number of qubits in the circuit",
            "required": True,
            "min": 1,
        },
        "gates": {
            "type": "list",
            "description": "List of gate operations",
            "required": True,
            "items": {
                "name": {
                    "type": "string",
                    "description": "Gate name (e.g., 'H', 'RX', 'CNOT')",
                    "required": True,
                },
                "targets": {
                    "type": "list",
                    "description": "Target qubit indices (0-based)",
                    "required": True,
                    "items": {"type": "integer", "min": 0},
                },
                "controls": {
                    "type": "list",
                    "description": "Control qubit indices (optional, for controlled gates)",
                    "required": False,
                    "items": {"type": "integer", "min": 0},
                },
                "params": {
                    "type": "list",
                    "description": "Gate parameters (angles in radians)",
                    "required": False,
                    "items": {"type": "number"},
                },
                "label": {
                    "type": "string",
                    "description": "Optional user label/metadata",
                    "required": False,
                },
            },
        },
        "metadata": {
            "type": "dict",
            "description": "Optional metadata (producer, timestamp, notes, etc.)",
            "required": False,
        },
        "endian": {
            "type": "string",
            "description": "Qubit ordering convention ('little' or 'big')",
            "required": False,
            "default": "little",
        },
    }


def validate_json_circuit(obj: dict) -> None:
    """
    Validate a JSON circuit object against the schema.

    Performs structural validation: checks required fields, types, and basic
    constraints. Raises ValueError with a descriptive message if validation fails.

    Parameters
    ----------
    obj : dict
        JSON object to validate.

    Raises
    ------
    ValueError
        If the object does not conform to the schema.
    """
    if not isinstance(obj, dict):
        raise ValueError("JSON circuit must be a dictionary object.")

    # Check required fields
    if "version" not in obj:
        raise ValueError("JSON circuit missing required field 'version'.")
    if not isinstance(obj["version"], str):
        raise ValueError("Field 'version' must be a string.")

    if "n_qubits" not in obj:
        raise ValueError("JSON circuit missing required field 'n_qubits'.")
    if not isinstance(obj["n_qubits"], int):
        raise ValueError("Field 'n_qubits' must be an integer.")
    if obj["n_qubits"] < 1:
        raise ValueError(f"Field 'n_qubits' must be >= 1, got {obj['n_qubits']}.")

    if "gates" not in obj:
        raise ValueError("JSON circuit missing required field 'gates'.")
    if not isinstance(obj["gates"], list):
        raise ValueError("Field 'gates' must be a list.")

    # Validate each gate
    for i, gate in enumerate(obj["gates"]):
        if not isinstance(gate, dict):
            raise ValueError(f"Gate at index {i} must be a dictionary object.")

        if "name" not in gate:
            raise ValueError(f"Gate at index {i} missing required field 'name'.")
        if not isinstance(gate["name"], str):
            raise ValueError(f"Gate at index {i}: field 'name' must be a string.")

        if "targets" not in gate:
            raise ValueError(f"Gate at index {i} missing required field 'targets'.")
        if not isinstance(gate["targets"], list):
            raise ValueError(f"Gate at index {i}: field 'targets' must be a list.")
        for j, q in enumerate(gate["targets"]):
            if not isinstance(q, int):
                raise ValueError(
                    f"Gate at index {i}: targets[{j}] must be an integer, got {type(q).__name__}."
                )
            if q < 0:
                raise ValueError(
                    f"Gate at index {i}: targets[{j}] must be >= 0, got {q}."
                )
            if q >= obj["n_qubits"]:
                raise ValueError(
                    f"Gate at index {i}: targets[{j}] = {q} is out of range "
                    f"[0, {obj['n_qubits']})."
                )

        # Validate optional fields
        if "controls" in gate:
            if not isinstance(gate["controls"], list):
                raise ValueError(f"Gate at index {i}: field 'controls' must be a list.")
            for j, q in enumerate(gate["controls"]):
                if not isinstance(q, int):
                    raise ValueError(
                        f"Gate at index {i}: controls[{j}] must be an integer, "
                        f"got {type(q).__name__}."
                    )
                if q < 0:
                    raise ValueError(
                        f"Gate at index {i}: controls[{j}] must be >= 0, got {q}."
                    )
                if q >= obj["n_qubits"]:
                    raise ValueError(
                        f"Gate at index {i}: controls[{j}] = {q} is out of range "
                        f"[0, {obj['n_qubits']})."
                    )

        if "params" in gate:
            if not isinstance(gate["params"], list):
                raise ValueError(f"Gate at index {i}: field 'params' must be a list.")
            for j, p in enumerate(gate["params"]):
                if not isinstance(p, (int, float)):
                    raise ValueError(
                        f"Gate at index {i}: params[{j}] must be a number, "
                        f"got {type(p).__name__}."
                    )

        if "label" in gate:
            if not isinstance(gate["label"], str):
                raise ValueError(f"Gate at index {i}: field 'label' must be a string.")

    # Validate optional metadata
    if "metadata" in obj:
        if not isinstance(obj["metadata"], dict):
            raise ValueError("Field 'metadata' must be a dictionary.")

    # Validate optional endian
    if "endian" in obj:
        if not isinstance(obj["endian"], str):
            raise ValueError("Field 'endian' must be a string.")
        if obj["endian"] not in ("little", "big"):
            raise ValueError(
                f"Field 'endian' must be 'little' or 'big', got {obj['endian']!r}."
            )


__all__ = ["json_circuit_schema", "validate_json_circuit"]

