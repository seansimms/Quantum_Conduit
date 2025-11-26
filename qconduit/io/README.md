# Quantum Conduit I/O Module

This module provides import/export functionality for quantum circuits in two formats:

1. **OpenQASM 2.0** - A subset of the OpenQASM 2.0 specification for quantum circuit description
2. **JSON IR** - A canonical JSON interchange format for quantum circuits

## Supported Features

### OpenQASM 2.0 Subset

The parser supports a safe, educational subset of OpenQASM 2.0:

**Supported constructs:**
- `OPENQASM 2.0;` header
- `include "qelib1.inc";` (optional, ignored)
- `qreg` declarations (single or multiple registers)
- `creg` declarations (parsed but ignored for circuit structure)
- Standard gates: `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`
- Rotation gates: `rx`, `ry`, `rz` with angle parameters
- Controlled gates: `cx` (CNOT)
- U gates: `u1(λ)`, `u2(φ,λ)`, `u3(θ,φ,λ)` - automatically decomposed to RZ/RY/RX
- `measure` statements (parsed as annotations, not executable gates)
- Comments: `//` (line) and `/* */` (block)
- Whitespace-tolerant parsing

**Unsupported features:**
- OpenQASM 3.0
- Custom gate definitions (`gate` keyword)
- Classical control flow (`if` statements)
- Parameterized gates (only constant angles supported)
- Opaque gates

**U gate decompositions:**
- `u1(λ)` → `RZ(λ)`
- `u2(φ,λ)` → `RZ(φ) RY(π/2) RZ(λ)`
- `u3(θ,φ,λ)` → `RZ(φ) RY(θ) RZ(λ)`

### JSON IR Format

The JSON IR provides a compact, portable representation of quantum circuits:

```json
{
  "version": "qconduit-json-1.0",
  "n_qubits": 2,
  "gates": [
    {
      "name": "H",
      "targets": [0]
    },
    {
      "name": "CNOT",
      "targets": [1],
      "controls": [0]
    },
    {
      "name": "RX",
      "targets": [0],
      "params": [1.57079632679]
    }
  ],
  "metadata": {
    "producer": "qconduit",
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "endian": "little"
}
```

**Schema fields:**
- `version`: Schema version string (required)
- `n_qubits`: Number of qubits (required, ≥1)
- `gates`: List of gate objects (required)
  - `name`: Gate name string (required)
  - `targets`: List of target qubit indices (required)
  - `controls`: List of control qubit indices (optional, for CNOT)
  - `params`: List of parameters in radians (optional)
  - `label`: User metadata string (optional)
- `metadata`: Optional metadata dictionary
- `endian`: Qubit ordering convention ("little" or "big", default "little")

**Supported gates:**
- Single-qubit: `H`, `X`, `Y`, `Z`, `S`, `T`, `I`
- Rotations: `RX`, `RY`, `RZ` (with one parameter)
- Two-qubit: `CNOT`

## Usage Examples

### OpenQASM Import/Export

```python
from qconduit.io import parse_qasm_string, export_circuit_to_qasm
from qconduit.circuit import QuantumCircuit

# Parse QASM string
qasm = """OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0],q[1];
"""

circuit = parse_qasm_string(qasm)

# Export circuit to QASM
qasm_output = export_circuit_to_qasm(circuit)
print(qasm_output)
```

### JSON IR Import/Export

```python
from qconduit.io import circuit_to_json, json_to_circuit, dump_json_circuit, load_json_circuit
from qconduit.circuit import QuantumCircuit

# Create circuit
circuit = QuantumCircuit(2)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])

# Convert to JSON
json_obj = circuit_to_json(circuit, metadata={"producer": "my_tool"})

# Convert back to circuit
reconstructed = json_to_circuit(json_obj)

# File I/O
dump_json_circuit(circuit, "circuit.json")
loaded = load_json_circuit("circuit.json")
```

### Schema Validation

```python
from qconduit.io import validate_json_circuit, json_circuit_schema

# Get schema definition
schema = json_circuit_schema()

# Validate JSON object
json_obj = {
    "version": "qconduit-json-1.0",
    "n_qubits": 1,
    "gates": [{"name": "H", "targets": [0]}]
}

validate_json_circuit(json_obj)  # Raises ValueError if invalid
```

## Angle Expression Parsing

The QASM parser supports angle expressions composed of:
- Decimal numbers: `1.5`, `0.785`
- Pi multiples: `pi`, `pi/2`, `3*pi/4`, `-pi`
- Arithmetic: `*`, `/`, parentheses, unary minus

Examples:
- `pi/2` → π/2 radians
- `3*pi/4` → 3π/4 radians
- `-pi` → -π radians
- `0.78539816339` → decimal value

## Qubit Ordering Convention

The default endianness is **"little"**, meaning qubit 0 is the least significant bit (LSB) in the computational basis index. This matches the statevector backend convention.

For example, in a 2-qubit system:
- `|00⟩` = index 0 (qubit 0 = 0, qubit 1 = 0)
- `|01⟩` = index 1 (qubit 0 = 1, qubit 1 = 0)
- `|10⟩` = index 2 (qubit 0 = 0, qubit 1 = 1)
- `|11⟩` = index 3 (qubit 0 = 1, qubit 1 = 1)

## Multiple Register Handling

When multiple `qreg` declarations are present, registers are flattened into a single global index space in declaration order:

```qasm
qreg a[2];  # a[0] → global index 0, a[1] → global index 1
qreg b[2];  # b[0] → global index 2, b[1] → global index 3
```

## Error Handling

The parser provides clear error messages for unsupported constructs:

```python
qasm = "if(c==1) x q[0];"
# Raises ValueError: "Unsupported construct: if statements are not supported."
```

Invalid JSON structures also raise descriptive errors:

```python
invalid_json = {"version": "qconduit-json-1.0"}  # Missing n_qubits
# Raises ValueError: "JSON circuit missing required field 'n_qubits'."
```

## Limitations and Extensions

**Current limitations:**
- Only constant angle parameters (no parameterized gates)
- No classical control flow
- No custom gate definitions
- Limited to standard single- and two-qubit gates

**Extending support:**
To add support for additional gates or features:
1. Update `_apply_gate_to_circuit()` in `qasm2.py` for QASM parsing
2. Update `_gate_to_qasm()` in `qasm2.py` for QASM export
3. Update gate validation in `json_to_circuit()` in `json_ir.py`
4. Add corresponding tests

## Round-Trip Guarantees

For supported gates, the following round-trips preserve unitary equivalence (up to global phase):
- `circuit → QASM → circuit'`
- `circuit → JSON → circuit'`

Equivalence is verified using statevector simulation with tolerance 1e-8.

## Testing

Comprehensive tests cover:
- Basic import/export functionality
- U gate decomposition correctness
- Round-trip equivalence
- Edge cases (angle parsing, multiple registers, comments)
- Error handling (unsupported constructs, invalid syntax)
- Schema validation

Run tests with:
```bash
pytest tests/test_qasm_import_export.py tests/test_json_ir_roundtrip.py tests/test_qasm_edgecases.py
```



