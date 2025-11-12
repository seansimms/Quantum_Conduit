# Quantum Conduit

Quantum Conduit is a PyTorch-native quantum statevector plumbing library designed for quantum machine learning applications. It provides clean, minimal abstractions for quantum operations that integrate seamlessly with PyTorch's computational graph.

## Features

- **Statevector Backend**: Pure quantum state operations with batch support
- **Standard Gate Library**: Common single- and two-qubit gates (Pauli, Hadamard, rotations, CNOT)
- **Device Abstraction**: CPU and CUDA device management
- **QuantumModule**: Base class for quantum layers compatible with `torch.nn.Module`

## Installation

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import qconduit as qc

# Create a 1-qubit zero state
state = qc.zero_state(n_qubits=1)

# Apply Hadamard gate
h_gate = qc.H()
state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)

# Compute probabilities
probs = qc.measure_probs(state, n_qubits=1)
print(f"Probabilities: {probs}")  # Should be approximately [0.5, 0.5]

# Measure Z expectation
z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
print(f"<Z>: {z_exp}")  # Should be approximately 0.0
```

## Running Tests

```bash
pytest
```

## License

MIT License - see LICENSE file for details.

