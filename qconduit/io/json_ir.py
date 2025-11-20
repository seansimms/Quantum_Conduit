"""JSON IR import and export for quantum circuits.

This module provides functions to convert QuantumCircuit objects to/from
a canonical JSON interchange format. The JSON format is compact, portable,
and sufficient for representing textbook quantum circuits.

See schema.py for the JSON IR schema specification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from qconduit.circuit import QuantumCircuit

from .schema import validate_json_circuit
from .utils import gate_name_normalize


def circuit_to_json(
    circuit: QuantumCircuit, metadata: Optional[dict] = None
) -> dict:
    """
    Convert a QuantumCircuit to JSON IR format.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to convert.
    metadata : dict, optional
        Optional metadata dictionary (producer, timestamp, notes, etc.).
        Must be JSON-serializable.

    Returns
    -------
    dict
        JSON IR object following the schema defined in schema.py.
    """
    gates_list = []

    for op in circuit.ops:
        gate_obj: Dict[str, Any] = {
            "name": op.name,
            "targets": list(op.qubits),
        }

        # Add parameters if present
        if op.params is not None and len(op.params) > 0:
            gate_obj["params"] = [float(p) for p in op.params]

        # For CNOT, separate control and target
        if op.name.upper() == "CNOT" and len(op.qubits) == 2:
            gate_obj["controls"] = [op.qubits[0]]
            gate_obj["targets"] = [op.qubits[1]]

        gates_list.append(gate_obj)

    result: Dict[str, Any] = {
        "version": "qconduit-json-1.0",
        "n_qubits": circuit.n_qubits,
        "gates": gates_list,
    }

    if metadata:
        result["metadata"] = metadata

    result["endian"] = "little"  # Default endianness

    return result


def json_to_circuit(obj: dict) -> QuantumCircuit:
    """
    Convert a JSON IR object to a QuantumCircuit.

    Parameters
    ----------
    obj : dict
        JSON IR object following the schema defined in schema.py.

    Returns
    -------
    QuantumCircuit
        Reconstructed circuit.

    Raises
    ------
    ValueError
        If the JSON object is invalid or contains unsupported gates.
    """
    # Validate schema
    validate_json_circuit(obj)

    n_qubits = obj["n_qubits"]
    circuit = QuantumCircuit(n_qubits)

    # Process gates
    for gate_obj in obj["gates"]:
        name = gate_name_normalize(gate_obj["name"])
        targets = gate_obj["targets"]

        # Handle controlled gates
        if "controls" in gate_obj:
            controls = gate_obj["controls"]
            # For CNOT, combine controls and targets
            if name == "CNOT":
                if len(controls) != 1 or len(targets) != 1:
                    raise ValueError(
                        f"CNOT gate must have exactly 1 control and 1 target, "
                        f"got {len(controls)} controls and {len(targets)} targets."
                    )
                qubits = controls + targets
                circuit.add_gate("CNOT", qubits)
                continue
            else:
                # Other controlled gates not yet supported
                raise ValueError(
                    f"Controlled gates other than CNOT are not yet supported: {name}"
                )

        # Handle parameters
        params = None
        if "params" in gate_obj and gate_obj["params"]:
            params = [float(p) for p in gate_obj["params"]]

        # Validate qubit indices
        for q in targets:
            if q < 0 or q >= n_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {n_qubits}) "
                    f"in gate '{name}'."
                )

        # Validate gate name (check against supported gates)
        supported_gates = {
            "H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "I"
        }
        if name not in supported_gates:
            raise ValueError(
                f"Unsupported gate '{name}' in JSON circuit. "
                f"Supported gates: {sorted(supported_gates)}."
            )

        # Add gate
        circuit.add_gate(name, targets, params)

    return circuit


def dump_json_circuit(circuit: QuantumCircuit, path: str) -> None:
    """
    Write a QuantumCircuit to a JSON file.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to write.
    path : str
        Path to output JSON file.
    """
    obj = circuit_to_json(circuit)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json_circuit(path: str) -> QuantumCircuit:
    """
    Load a QuantumCircuit from a JSON file.

    Parameters
    ----------
    path : str
        Path to input JSON file.

    Returns
    -------
    QuantumCircuit
        Loaded circuit.

    Raises
    ------
    ValueError
        If the file is invalid or contains unsupported gates.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON circuit file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}: {e}")

    return json_to_circuit(obj)


__all__ = [
    "circuit_to_json",
    "json_to_circuit",
    "dump_json_circuit",
    "load_json_circuit",
]

