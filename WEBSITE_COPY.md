# Quantum Conduit - Website Copy

## Hero Section

**Headline:**
# PyTorch-Native Quantum Computing for Machine Learning

**Subheadline:**
Quantum Conduit is the world's first PyTorch-native quantum statevector library. Built from the ground up for quantum machine learning, it provides clean, low-level abstractions that integrate seamlessly with PyTorch's computational graph.

**Key Value Proposition:**
Native autograd support. Full batch processing. Zero friction integration with PyTorch neural networks.

---

## What is Quantum Conduit?

Quantum Conduit is a minimal quantum computing library designed specifically for researchers and developers building quantum machine learning models. Unlike high-level quantum frameworks, Quantum Conduit operates at the "plumbing" level—giving you direct control over quantum states and operations while maintaining full compatibility with PyTorch's ecosystem.

**Key Differentiators:**
- **PyTorch-Native**: Built as a first-class PyTorch library, not a wrapper
- **Autograd Support**: Full automatic differentiation throughout the quantum stack
- **Batch Processing**: Efficient batch operations optimized for ML workloads
- **Low-Level Control**: Direct access to quantum statevectors and operations

---

## Features

**Core Capabilities**
- Statevector and density matrix backends with full batch support
- Complete standard gate library (I, X, Y, Z, H, S, T, CNOT, RX, RY, RZ)
- Seamless CPU and CUDA support with automatic device management

**Quantum Machine Learning**
- Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA)
- Parameter-shift gradient computation
- Native integration with PyTorch neural networks
- Complete training infrastructure with callbacks and history tracking

**Advanced Features**
- Adiabatic quantum computing
- Fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev)
- Quantum state tomography
- Circuit transpilation and decomposition
- Comprehensive noise models and quantum channels

---

## Why Quantum Conduit?

If you're building quantum machine learning models and already know PyTorch, Quantum Conduit provides the most natural integration. No plugin layers. No wrappers. Just quantum computing primitives that work exactly like PyTorch tensors.

**Perfect for:**
- Researchers developing quantum ML algorithms
- Developers building hybrid quantum-classical models
- Teams already using PyTorch who need quantum capabilities

---

## Get Started

```bash
pip install qconduit
```

**Quick Example:**
```python
import torch
import qconduit as qc

# Create a quantum state
state = qc.zero_state(n_qubits=2)

# Apply gates with full autograd support
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=2)
state = qc.apply_gate(state, qc.CNOT(), qubits=[0, 1], n_qubits=2)

# Measure expectations
expectation = qc.measure_expectation_z(state, qubit=0, n_qubits=2)
```

**Learn More:**
- [PyPI Package](https://pypi.org/project/qconduit/)
- [Documentation](https://pypi.org/project/qconduit/)
- [Project Details](https://pypi.org/project/qconduit/)

---

## Open Source

Quantum Conduit is open source and available under the MIT License. We welcome contributions from the quantum machine learning community.

**Citation:**
If you use Quantum Conduit in your research, please cite our [Zenodo publication](https://doi.org/10.5281/zenodo.17599984).

---

## Call to Action

**Ready to build quantum ML models with PyTorch?**

```bash
pip install qconduit
```

[View on PyPI](https://pypi.org/project/qconduit/) • [Install Now](https://pypi.org/project/qconduit/) • [Cite in Research](https://doi.org/10.5281/zenodo.17599984)

