"""OpenQASM 2.0 parser and exporter for quantum circuits.

This module provides functions to parse OpenQASM 2.0 files/strings into
QuantumCircuit objects and export QuantumCircuit objects to OpenQASM 2.0 format.

Supported QASM 2.0 subset:
    - OPENQASM 2.0; header
    - include "qelib1.inc"; (optional, ignored)
    - qreg declarations (single or multiple registers)
    - creg declarations (parsed but ignored)
    - Standard gates: h, x, y, z, s, t, sdg, tdg, rx, ry, rz
    - Controlled gates: cx (CNOT)
    - U gates: u1(λ), u2(φ,λ), u3(θ,φ,λ) - decomposed to RZ/RY/RX
    - measure statements (parsed as annotations, not executable gates)
    - Comments: // and /* */

Unsupported features:
    - OpenQASM 3.0
    - Custom gate definitions
    - Classical control flow (if statements)
    - Parameterized gates (only constant angles)
    - Opaque gates
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from qconduit.circuit import QuantumCircuit

from .utils import angle_str_to_float, float_to_angle_str, gate_name_normalize


def parse_qasm_string(qasm: str) -> QuantumCircuit:
    """
    Parse an OpenQASM 2.0 string into a QuantumCircuit.

    Parameters
    ----------
    qasm : str
        OpenQASM 2.0 source code.

    Returns
    -------
    QuantumCircuit
        Parsed circuit.

    Raises
    ------
    ValueError
        If the QASM contains unsupported constructs or syntax errors.
    """
    # Normalize line endings and strip comments
    lines = _preprocess_qasm(qasm)

    # Parse statements
    statements = _tokenize_statements(lines)

    # Track register declarations
    qregs: Dict[str, Tuple[int, int]] = {}  # name -> (start_index, size)
    cregs: Dict[str, int] = {}  # name -> size (for validation, not used)
    total_qubits = 0

    # First pass: collect register declarations
    for stmt in statements:
        if stmt.startswith("OPENQASM"):
            continue  # Header, ignore version check
        if stmt.startswith('include "qelib1.inc"'):
            continue  # Standard library include, ignore
        if stmt.startswith("qreg "):
            name, size = _parse_qreg_declaration(stmt)
            qregs[name] = (total_qubits, size)
            total_qubits += size
        elif stmt.startswith("creg "):
            name, size = _parse_creg_declaration(stmt)
            cregs[name] = size

    if total_qubits == 0:
        raise ValueError("No qreg declarations found in QASM file.")

    # Create circuit
    circuit = QuantumCircuit(total_qubits)

    # Second pass: parse gate applications
    for stmt in statements:
        if stmt.startswith(("OPENQASM", "include", "qreg ", "creg ", "barrier")):
            continue

        if stmt.startswith("measure "):
            # Parse measure but don't add executable gate (just annotation)
            # For now, we'll skip measure statements
            continue

        # Check for unsupported constructs (must check before gate parsing)
        stmt_stripped = stmt.strip()
        if stmt_stripped.startswith("if ") or stmt_stripped.lower().startswith("if("):
            raise ValueError(
                f"Unsupported construct: if statements are not supported. "
                f"Offending line: {stmt!r}"
            )
        if stmt_stripped.startswith("gate ") or stmt_stripped.lower().startswith("gate "):
            raise ValueError(
                f"Unsupported construct: custom gate definitions are not supported. "
                f"Offending line: {stmt!r}"
            )

        # Parse gate statement
        _parse_gate_statement(circuit, stmt, qregs)

    return circuit


def parse_qasm_file(path: str) -> QuantumCircuit:
    """
    Parse an OpenQASM 2.0 file into a QuantumCircuit.

    Parameters
    ----------
    path : str
        Path to the QASM file.

    Returns
    -------
    QuantumCircuit
        Parsed circuit.

    Raises
    ------
    ValueError
        If the file cannot be read or contains unsupported constructs.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"QASM file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error reading QASM file {path}: {e}")

    return parse_qasm_string(content)


def export_circuit_to_qasm(
    circuit: QuantumCircuit, include_qelib: bool = True
) -> str:
    """
    Export a QuantumCircuit to OpenQASM 2.0 format.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to export.
    include_qelib : bool
        Whether to include the qelib1.inc header.

    Returns
    -------
    str
        OpenQASM 2.0 source code.

    Raises
    ------
    ValueError
        If the circuit contains gates that cannot be represented in QASM 2.0.
    """
    lines = ["OPENQASM 2.0;"]

    if include_qelib:
        lines.append('include "qelib1.inc";')

    lines.append(f"qreg q[{circuit.n_qubits}];")
    lines.append("")

    # Export gates
    for op in circuit.ops:
        qasm_line = _gate_to_qasm(op)
        if qasm_line:
            lines.append(qasm_line)

    return "\n".join(lines)


def _preprocess_qasm(qasm: str) -> List[str]:
    """Remove comments and normalize whitespace."""
    lines = []
    in_block_comment = False

    for line in qasm.split("\n"):
        # Handle block comments
        if "/*" in line:
            if "*/" in line:
                # Block comment on same line
                line = re.sub(r"/\*.*?\*/", "", line)
            else:
                # Start of block comment
                line = line[: line.index("/*")]
                in_block_comment = True

        if in_block_comment:
            if "*/" in line:
                # End of block comment
                line = line[line.index("*/") + 2 :]
                in_block_comment = False
            else:
                # Still in block comment
                continue

        # Remove line comments
        if "//" in line:
            line = line[: line.index("//")]

        # Strip whitespace
        line = line.strip()

        if line:
            lines.append(line)

    return lines


def _tokenize_statements(lines: List[str]) -> List[str]:
    """Tokenize lines into semicolon-terminated statements."""
    statements = []
    current = []

    for line in lines:
        current.append(line)
        # Check if line ends with semicolon
        if line.endswith(";"):
            stmt = " ".join(current).rstrip(";").strip()
            if stmt:
                statements.append(stmt)
            current = []

    # Handle last statement if no trailing semicolon
    if current:
        stmt = " ".join(current).strip()
        if stmt:
            statements.append(stmt)

    return statements


def _parse_qreg_declaration(stmt: str) -> Tuple[str, int]:
    """Parse qreg declaration: qreg name[size];"""
    # Pattern: qreg name[size];
    match = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", stmt)
    if not match:
        raise ValueError(f"Invalid qreg declaration: {stmt!r}")
    name = match.group(1)
    size = int(match.group(2))
    if size <= 0:
        raise ValueError(f"Invalid qreg size: {size}")
    return name, size


def _parse_creg_declaration(stmt: str) -> Tuple[str, int]:
    """Parse creg declaration: creg name[size];"""
    match = re.match(r"creg\s+(\w+)\s*\[\s*(\d+)\s*\]", stmt)
    if not match:
        raise ValueError(f"Invalid creg declaration: {stmt!r}")
    name = match.group(1)
    size = int(match.group(2))
    if size <= 0:
        raise ValueError(f"Invalid creg size: {size}")
    return name, size


def _parse_gate_statement(
    circuit: QuantumCircuit, stmt: str, qregs: Dict[str, Tuple[int, int]]
) -> None:
    """Parse a gate application statement and add to circuit."""
    # Pattern: gate_name(params?) qubit_list;
    # Examples:
    #   h q[0];
    #   cx q[0],q[1];
    #   u3(pi/2,pi/4,0) q[0];
    #   rx(pi/2) q[0];

    # Extract gate name and parameters
    # Match: name(params?) or just name
    gate_match = re.match(r"(\w+)\s*(\([^)]*\))?\s+(.+)", stmt)
    if not gate_match:
        raise ValueError(f"Invalid gate statement: {stmt!r}")

    gate_name = gate_match.group(1).lower()
    params_str = gate_match.group(2)  # Includes parentheses or None
    qubits_str = gate_match.group(3).strip().rstrip(";").strip()

    # Parse parameters
    params: List[float] = []
    if params_str:
        # Remove parentheses
        params_str = params_str.strip("()")
        if params_str:
            # Split by comma and parse each parameter
            param_parts = [p.strip() for p in params_str.split(",")]
            for p in param_parts:
                if p:
                    params.append(angle_str_to_float(p))

    # Parse qubit list
    qubit_indices = _parse_qubit_list(qubits_str, qregs)

    # Map gate to circuit API
    _apply_gate_to_circuit(circuit, gate_name, params, qubit_indices)


def _parse_qubit_list(
    qubits_str: str, qregs: Dict[str, Tuple[int, int]]
) -> List[int]:
    """Parse qubit list like 'q[0],q[1]' or 'q[0]' into global indices."""
    # Split by comma
    parts = [p.strip() for p in qubits_str.split(",")]

    indices = []
    for part in parts:
        # Pattern: reg_name[index]
        match = re.match(r"(\w+)\s*\[\s*(\d+)\s*\]", part)
        if not match:
            raise ValueError(f"Invalid qubit reference: {part!r}")

        reg_name = match.group(1)
        local_index = int(match.group(2))

        if reg_name not in qregs:
            raise ValueError(
                f"Unknown register '{reg_name}' in qubit reference: {part!r}"
            )

        start_index, size = qregs[reg_name]
        if local_index >= size:
            raise ValueError(
                f"Qubit index {local_index} out of range for register "
                f"'{reg_name}' (size {size})"
            )

        global_index = start_index + local_index
        indices.append(global_index)

    return indices


def _apply_gate_to_circuit(
    circuit: QuantumCircuit,
    gate_name: str,
    params: List[float],
    qubit_indices: List[int],
) -> None:
    """Apply a parsed gate to the circuit, handling decompositions."""
    gate_name_lower = gate_name.lower()

    # Handle U gates (decompose to RZ/RY/RX)
    if gate_name_lower == "u1":
        if len(params) != 1:
            raise ValueError(f"u1 gate requires 1 parameter, got {len(params)}")
        if len(qubit_indices) != 1:
            raise ValueError(f"u1 gate requires 1 qubit, got {len(qubit_indices)}")
        # u1(λ) = RZ(λ)
        circuit.add_gate("RZ", [qubit_indices[0]], [params[0]])

    elif gate_name_lower == "u2":
        if len(params) != 2:
            raise ValueError(f"u2 gate requires 2 parameters, got {len(params)}")
        if len(qubit_indices) != 1:
            raise ValueError(f"u2 gate requires 1 qubit, got {len(qubit_indices)}")
        # u2(φ,λ) = RZ(φ) RY(π/2) RZ(λ)
        phi, lam = params[0], params[1]
        circuit.add_gate("RZ", [qubit_indices[0]], [phi])
        circuit.add_gate("RY", [qubit_indices[0]], [math.pi / 2.0])
        circuit.add_gate("RZ", [qubit_indices[0]], [lam])

    elif gate_name_lower == "u3":
        if len(params) != 3:
            raise ValueError(f"u3 gate requires 3 parameters, got {len(params)}")
        if len(qubit_indices) != 1:
            raise ValueError(f"u3 gate requires 1 qubit, got {len(qubit_indices)}")
        # u3(θ,φ,λ) = RZ(φ) RY(θ) RZ(λ)
        theta, phi, lam = params[0], params[1], params[2]
        circuit.add_gate("RZ", [qubit_indices[0]], [phi])
        circuit.add_gate("RY", [qubit_indices[0]], [theta])
        circuit.add_gate("RZ", [qubit_indices[0]], [lam])

    # Handle standard gates
    elif gate_name_lower in ("h", "x", "y", "z", "s", "t"):
        if len(qubit_indices) != 1:
            raise ValueError(
                f"{gate_name} gate requires 1 qubit, got {len(qubit_indices)}"
            )
        if params:
            raise ValueError(f"{gate_name} gate does not take parameters")
        circuit.add_gate(gate_name.upper(), [qubit_indices[0]])

    elif gate_name_lower in ("sdg", "tdg"):
        if len(qubit_indices) != 1:
            raise ValueError(
                f"{gate_name} gate requires 1 qubit, got {len(qubit_indices)}"
            )
        if params:
            raise ValueError(f"{gate_name} gate does not take parameters")
        # sdg = S^dagger = RZ(-π/2), tdg = T^dagger = RZ(-π/4)
        if gate_name_lower == "sdg":
            circuit.add_gate("RZ", [qubit_indices[0]], [-math.pi / 2.0])
        else:  # tdg
            circuit.add_gate("RZ", [qubit_indices[0]], [-math.pi / 4.0])

    elif gate_name_lower in ("rx", "ry", "rz"):
        if len(params) != 1:
            raise ValueError(f"{gate_name} gate requires 1 parameter, got {len(params)}")
        if len(qubit_indices) != 1:
            raise ValueError(f"{gate_name} gate requires 1 qubit, got {len(qubit_indices)}")
        circuit.add_gate(gate_name.upper(), [qubit_indices[0]], [params[0]])

    elif gate_name_lower in ("cx", "cnot"):
        if len(qubit_indices) != 2:
            raise ValueError(f"cx gate requires 2 qubits, got {len(qubit_indices)}")
        if params:
            raise ValueError("cx gate does not take parameters")
        circuit.add_gate("CNOT", qubit_indices)

    else:
        raise ValueError(
            f"Unsupported gate '{gate_name}' in QASM. "
            "Supported gates: h, x, y, z, s, t, sdg, tdg, rx, ry, rz, "
            "u1, u2, u3, cx."
        )


def _gate_to_qasm(op) -> Optional[str]:
    """Convert a GateOp to QASM line."""
    # Access the gate operation attributes
    from qconduit.circuit import GateOp
    
    if not isinstance(op, GateOp):
        raise TypeError(f"Expected GateOp, got {type(op)}")
    
    name = op.name.upper()
    qubits = op.qubits
    params = op.params

    # Map gate names to QASM equivalents
    if name == "H":
        if len(qubits) != 1:
            return None
        return f"h q[{qubits[0]}];"

    elif name == "X":
        if len(qubits) != 1:
            return None
        return f"x q[{qubits[0]}];"

    elif name == "Y":
        if len(qubits) != 1:
            return None
        return f"y q[{qubits[0]}];"

    elif name == "Z":
        if len(qubits) != 1:
            return None
        return f"z q[{qubits[0]}];"

    elif name == "S":
        if len(qubits) != 1:
            return None
        if params and len(params) > 0:
            # Parameterized S gate - use rz
            return f"rz({float_to_angle_str(params[0])}) q[{qubits[0]}];"
        return f"s q[{qubits[0]}];"

    elif name == "T":
        if len(qubits) != 1:
            return None
        if params and len(params) > 0:
            # Parameterized T gate - use rz
            return f"rz({float_to_angle_str(params[0])}) q[{qubits[0]}];"
        return f"t q[{qubits[0]}];"

    elif name == "RX":
        if len(qubits) != 1 or not params or len(params) != 1:
            return None
        return f"rx({float_to_angle_str(params[0])}) q[{qubits[0]}];"

    elif name == "RY":
        if len(qubits) != 1 or not params or len(params) != 1:
            return None
        return f"ry({float_to_angle_str(params[0])}) q[{qubits[0]}];"

    elif name == "RZ":
        if len(qubits) != 1 or not params or len(params) != 1:
            return None
        # RZ can be represented as u1 in QASM
        angle = params[0]
        # Check if it's a simple multiple of pi/4 for readability
        # Otherwise use u1
        return f"u1({float_to_angle_str(angle)}) q[{qubits[0]}];"

    elif name == "CNOT":
        if len(qubits) != 2:
            return None
        return f"cx q[{qubits[0]}],q[{qubits[1]}];"

    else:
        # Unsupported gate - raise error
        raise ValueError(
            f"Cannot export gate '{name}' to QASM 2.0. "
            "Supported gates: H, X, Y, Z, S, T, RX, RY, RZ, CNOT."
        )


__all__ = ["parse_qasm_string", "parse_qasm_file", "export_circuit_to_qasm"]

