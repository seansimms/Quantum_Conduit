# Quantum Conduit

<div align="center">

**The world's first PyTorch-native quantum statevector plumbing library for quantum machine learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/yourusername/Quantum_Conduit)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#api-reference) • [Examples](#examples)

</div>

---

Quantum Conduit is a minimal, PyTorch-native quantum statevector library designed specifically for quantum machine learning applications. Unlike high-level quantum frameworks, Quantum Conduit provides clean, low-level abstractions that integrate seamlessly with PyTorch's computational graph, enabling native autograd support and batch processing for quantum operations.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Comparison with Alternatives](#comparison-with-alternatives)
- [Contributing](#contributing)
- [License](#license)

## Features

Quantum Conduit provides a comprehensive set of quantum computing primitives optimized for machine learning:

### Core Capabilities

- **🔬 Statevector Backend**: Pure quantum state operations with full batch support
- **🌊 Density Matrix Backend**: Mixed quantum states for noise modeling (optimized for small systems)
- **⚛️ Standard Gate Library**: Complete set of single- and two-qubit gates (I, X, Y, Z, H, S, T, CNOT, RX, RY, RZ)
- **🖥️ Device Abstraction**: Seamless CPU and CUDA support with automatic device management
- **🧩 QuantumModule**: PyTorch-native module system compatible with `torch.nn.Module`

### Quantum Machine Learning

- **📊 Parametric Ansätze**: Hardware-efficient and custom ansätze for variational algorithms
- **🔍 VQE Algorithm**: Built-in Variational Quantum Eigensolver for ground-state energy estimation
- **🤖 Hybrid Quantum-Classical**: Seamless integration with PyTorch neural networks
- **📈 Parameter-Shift Gradients**: Quantum-aware gradient computation via parameter-shift rule
- **🔄 Full Autograd Support**: Native PyTorch differentiation throughout the stack

### Advanced Features

- **🎯 Pauli Operators**: Complete support for Pauli-term and Pauli-sum Hamiltonians
- **🌪️ Noise Models**: Standard quantum channels (depolarizing, amplitude damping, phase damping)
- **📦 Batch Processing**: Efficient batch operations for training quantum models
- **🎨 Extensible Design**: Clean abstractions for custom gates, ansätze, and algorithms

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.1 or higher

### Basic Installation

```bash
pip install -e .
```

### Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

### CUDA Support

CUDA support is automatically available if PyTorch was installed with CUDA support. No additional configuration is required.

## Quick Start

### Example 1: Basic Quantum Operations

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
print(f"Probabilities: {probs}")  # [0.5, 0.5]

# Measure Z expectation
z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=1)
print(f"<Z>: {z_exp}")  # ~0.0
```

### Example 2: Variational Quantum Eigensolver (VQE)

```python
import torch
import qconduit as qc
from qconduit.algorithms import VQE
from qconduit.layers import HardwareEfficientAnsatz

# Define a 2-qubit Hamiltonian (diagonal)
hamiltonian = torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32)

# Create ansatz
ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)

# Create VQE instance (hamiltonian can be tensor or PauliSum)
vqe = VQE(ansatz=ansatz, hamiltonian=hamiltonian)

# Initialize parameters
params = torch.nn.Parameter(0.1 * torch.randn(ansatz.num_parameters))

# Optimize
optimizer = torch.optim.Adam([params], lr=0.1)
for step in range(50):
    optimizer.zero_grad()
    energy = vqe.energy(params)
    energy.backward()
    optimizer.step()
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}: energy = {energy.item():.6f}")
```

### Example 3: Hybrid Quantum-Classical Model

```python
import torch
import torch.nn as nn
from qconduit.layers import QuantumBlock

class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Quantum block: 2 qubits, depth 1, 2 input features
        self.quantum = QuantumBlock(n_qubits=2, depth=1, in_features=2)
        # Classical head
        self.head = nn.Linear(2, 2)  # 2 classes
    
    def forward(self, x):
        q_features = self.quantum(x)  # Quantum expectations
        return self.head(q_features)  # Classical classification

# Use like any PyTorch model
model = HybridClassifier()
x = torch.randn(32, 2)  # Batch of 32 samples
logits = model(x)
```

### Example 4: Noise Modeling

```python
import torch
import qconduit as qc
from qconduit.noise import DepolarizingChannel
from qconduit.layers import HardwareEfficientAnsatz

# Create a noisy quantum circuit
ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=2)
params = torch.randn(ansatz.num_parameters)

# Build state
state = ansatz(params)

# Apply noise model
noise = DepolarizingChannel(p=0.1)  # 10% depolarizing noise
rho = noise.apply_statevector(state, n_qubits=2)

# Compute noisy expectation values
from qconduit.backend.density_matrix import measure_probs_dm
probs = measure_probs_dm(rho)
print(f"Noisy probabilities: {probs}")
```

## Architecture

### Design Principles

Quantum Conduit is built on three core principles:

1. **PyTorch-Native**: All operations integrate seamlessly with PyTorch's autograd system, enabling end-to-end differentiation of quantum-classical hybrid models.

2. **Minimal Abstractions**: The library provides "plumbing" rather than high-level abstractions, giving you direct control over quantum states and operations.

3. **Batch-First**: All operations support batched inputs, enabling efficient training of quantum models on classical data.

### Library Structure

```
qconduit/
├── core/           # Core abstractions (Device, QuantumModule)
├── backend/        # Statevector and density matrix backends
├── gates/          # Standard quantum gates
├── layers/         # Parametric ansätze and hybrid blocks
├── algorithms/     # Quantum algorithms (VQE)
├── operators/      # Pauli operators and expectations
├── grad/           # Gradient computation (parameter-shift)
└── noise/          # Noise models and quantum channels
```

### Key Components

- **Device Abstraction**: Unified interface for CPU and CUDA operations
- **Statevector Backend**: Efficient pure-state simulation with O(2^n) memory
- **Density Matrix Backend**: Mixed-state simulation for noise (O(4^n) memory, small systems)
- **Gate Library**: Standard gates with gradient-preserving implementations
- **Module System**: `QuantumModule` base class compatible with PyTorch's module system

## Examples

The `examples/` directory contains complete, runnable examples:

- **`vqe_h2.py`**: Variational Quantum Eigensolver for finding ground-state energy
- **`hybrid_classifier.py`**: Hybrid quantum-classical neural network for classification

Run examples directly:

```bash
python examples/vqe_h2.py
python examples/hybrid_classifier.py
```

## API Reference

### Core Abstractions

```python
import qconduit as qc

# Device management
device = qc.device("sv_cpu")  # or "sv_cuda"
default = qc.default_device()

# Quantum module base class
class MyQuantumLayer(qc.QuantumModule):
    def forward(self, x):
        # Your quantum operations
        pass
```

### Backend Operations

```python
# State creation
state = qc.zero_state(n_qubits=2, batch_shape=(10,))  # Batched states

# Gate application
gate = qc.H()
state = qc.apply_gate(state, gate, qubit=0, n_qubits=2)
state = qc.apply_two_qubit_gate(state, qc.CNOT(), qubit1=0, qubit2=1, n_qubits=2)

# Measurements
probs = qc.measure_probs(state, n_qubits=2)
z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=2)
```

### Gates

```python
# Single-qubit gates
I_gate = qc.I()
X_gate = qc.X()
Y_gate = qc.Y()
Z_gate = qc.Z()
H_gate = qc.H()
S_gate = qc.S()
T_gate = qc.T()

# Parametric gates (preserve gradients)
theta = torch.tensor(0.5, requires_grad=True)
RX_gate = qc.RX(theta)
RY_gate = qc.RY(theta)
RZ_gate = qc.RZ(theta)

# Two-qubit gates
cnot = qc.CNOT(control_first=True)

# Utility
is_unitary = qc.is_unitary(gate_matrix)
```

### Layers and Ansätze

```python
from qconduit.layers import HardwareEfficientAnsatz, QuantumBlock, ParametricAnsatz

# Hardware-efficient ansatz
ansatz = HardwareEfficientAnsatz(n_qubits=4, depth=3)
params = torch.randn(ansatz.num_parameters)
state = ansatz(params)  # Forward pass

# Hybrid quantum-classical block
quantum_block = QuantumBlock(n_qubits=2, depth=2, in_features=10)
classical_features = torch.randn(32, 10)  # Batch of 32
quantum_features = quantum_block(classical_features)  # Shape: (32, 2)

# Custom ansatz
class MyAnsatz(ParametricAnsatz):
    def forward(self, params):
        state = qc.zero_state(n_qubits=self.n_qubits)
        # Your custom circuit
        return state
```

### Algorithms

```python
from qconduit.algorithms import VQE
from qconduit.operators import PauliTerm, PauliSum

# VQE with diagonal Hamiltonian
hamiltonian_diag = torch.tensor([0.0, 0.5, 0.5, 1.0])
vqe = VQE(ansatz=ansatz, hamiltonian=hamiltonian_diag)

# VQE with Pauli-sum Hamiltonian
pauli_ham = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "I")),
    PauliTerm(0.5, ("X", "X")),
])
vqe = VQE(ansatz=ansatz, hamiltonian=pauli_ham)

# Compute energy
energy = vqe.energy(params)  # Differentiable
```

### Operators

```python
from qconduit.operators import PauliTerm, PauliSum, expectation_pauli_term, expectation_pauli_sum

# Create Pauli terms
term1 = PauliTerm(1.0, ("Z", "I"))
term2 = PauliTerm(0.5, ("X", "Y"))

# Create Pauli-sum Hamiltonian
hamiltonian = PauliSum.from_terms([term1, term2])
hamiltonian = hamiltonian.simplify()  # Combine like terms

# Compute expectations
exp_val = expectation_pauli_term(state, term1)
total_exp = expectation_pauli_sum(state, hamiltonian)

# Convert to matrix (small systems only)
matrix = hamiltonian.to_matrix()  # (2^n, 2^n) complex tensor
```

### Gradients

```python
from qconduit.grad import param_shift_energy

# Parameter-shift gradients (alternative to autograd)
params = torch.tensor([0.1, 0.2], requires_grad=True)
energy = param_shift_energy(ansatz, hamiltonian, params)
energy.backward()  # Gradients computed via parameter-shift rule
```

### Noise Models

```python
from qconduit.noise import DepolarizingChannel, AmplitudeDampingChannel, PhaseDampingChannel

# Depolarizing noise
noise = DepolarizingChannel(p=0.1)  # 10% depolarizing probability
rho = noise.apply_statevector(state, n_qubits=2)

# Amplitude damping
amp_damp = AmplitudeDampingChannel(gamma=0.05)  # 5% damping
rho = amp_damp.apply_density_matrix(rho, n_qubits=2)

# Phase damping
phase_damp = PhaseDampingChannel(gamma=0.03)  # 3% dephasing
rho = phase_damp.apply_density_matrix(rho, n_qubits=2)
```

### Density Matrix Backend

```python
from qconduit.backend.density_matrix import (
    zero_dm_state,
    dm_from_statevector,
    measure_probs_dm,
    measure_expectation_z_dm,
)

# Create density matrix
rho = zero_dm_state(n_qubits=2)
# Or convert from statevector
rho = dm_from_statevector(state)

# Measurements
probs = measure_probs_dm(rho)
z_exp = measure_expectation_z_dm(rho, qubit=0, n_qubits=2)
```

## Performance Considerations

### Memory Complexity

- **Statevector Backend**: O(2^n) memory for n qubits
  - 1 qubit: 8 bytes (complex64)
  - 10 qubits: ~8 KB
  - 20 qubits: ~8 MB
  - 30 qubits: ~8 GB

- **Density Matrix Backend**: O(4^n) memory for n qubits
  - Intended for small systems (typically n ≤ 4)
  - 4 qubits: ~512 bytes
  - 8 qubits: ~128 MB

### Batch Processing

All operations support batched inputs, enabling efficient processing of multiple quantum states simultaneously. Batch dimensions are preserved throughout operations, making it easy to train quantum models on classical datasets.

### CUDA Acceleration

CUDA support is available when PyTorch is installed with CUDA. Simply use `device("sv_cuda")` to enable GPU acceleration. Quantum operations benefit from GPU parallelization, especially for large batch sizes.

### Optimization Tips

1. **Use statevector backend** for pure states (most common case)
2. **Use density matrix backend** only when noise modeling is required
3. **Leverage batch processing** for training on datasets
4. **Use CUDA** for large-scale simulations and batch processing
5. **Consider parameter-shift gradients** for specific use cases where autograd may be inefficient

## Comparison with Alternatives

### Quantum Conduit vs. Other Frameworks

| Feature | Quantum Conduit | Qiskit | PennyLane | Cirq |
|---------|----------------|--------|-----------|------|
| **PyTorch Integration** | ✅ Native | ❌ | ✅ Plugin | ❌ |
| **Autograd Support** | ✅ Full | ❌ | ✅ Plugin | ❌ |
| **Batch Processing** | ✅ Built-in | ❌ | ⚠️ Limited | ❌ |
| **Abstraction Level** | Low (plumbing) | High | Medium | Low |
| **Noise Models** | ✅ Standard | ✅ Advanced | ✅ Plugin | ✅ |
| **ML Focus** | ✅ Primary | ❌ | ✅ Primary | ❌ |
| **Learning Curve** | Low (PyTorch users) | Medium | Medium | Medium |

### Unique Value Proposition

Quantum Conduit is the **only** quantum library designed from the ground up as PyTorch-native plumbing. This means:

- **Zero friction** when integrating quantum layers into PyTorch models
- **Native autograd** without plugin layers or wrappers
- **Batch-first design** optimized for ML workloads
- **Minimal abstractions** giving you direct control

If you're building quantum machine learning models and already know PyTorch, Quantum Conduit provides the most natural integration.

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Quantum_Conduit.git
   cd Quantum_Conduit
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Run linter:
   ```bash
   ruff check .
   ```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and modular

### Testing

- Add tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code is linted
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the quantum machine learning community**

[Report Bug](https://github.com/yourusername/Quantum_Conduit/issues) • [Request Feature](https://github.com/yourusername/Quantum_Conduit/issues) • [Documentation](https://github.com/yourusername/Quantum_Conduit)

</div>
