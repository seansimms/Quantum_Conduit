# Quantum Conduit

<div align="center">

**The world's first PyTorch-native quantum statevector plumbing library for quantum machine learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/qconduit.svg)](https://badge.fury.io/py/qconduit)
[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/seansimms/Quantum_Conduit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17599984.svg)](https://doi.org/10.5281/zenodo.17599984)

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
- [Use Cases](#use-cases)
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
- **🔌 Circuit IR**: Structured circuit representation with simulation and visualization
- **🔍 Diagnostics**: State validation, fidelity computation, and debugging tools

### Quantum Machine Learning

- **📊 Parametric Ansätze**: Hardware-efficient and custom ansätze for variational algorithms
- **🔍 VQE Algorithm**: Built-in Variational Quantum Eigensolver for ground-state energy estimation
- **🎯 QAOA Algorithm**: Quantum Approximate Optimization Algorithm for MaxCut/Ising problems
- **🌡️ Adiabatic Evolution**: Adiabatic quantum computing with configurable schedules and circuit building
- **🎛️ Variational Scaffolding**: High-level APIs for running VQE and QAOA with result objects
- **🤖 Hybrid Quantum-Classical**: Seamless integration with PyTorch neural networks
- **📈 Parameter-Shift Gradients**: Quantum-aware gradient computation via parameter-shift rule
- **🔄 Full Autograd Support**: Native PyTorch differentiation throughout the stack
- **🏋️ Training Infrastructure**: Complete VQE training loop with callbacks and history tracking

### Advanced Features

- **🎯 Pauli Operators**: Complete support for Pauli-term and Pauli-sum Hamiltonians
- **🌪️ Noise Models**: Standard quantum channels, enhanced Kraus channels, and circuit-level noise simulation
- **📦 Batch Processing**: Efficient batch operations for training quantum models
- **🎨 Extensible Design**: Clean abstractions for custom gates, ansätze, and algorithms
- **🐛 Debug Mode**: Built-in debugging with normalization checks and validation
- **🎲 Sampling Utilities**: Bitstring sampling and probability distribution analysis
- **⏱️ Time Evolution**: Trotterization and exact Hamiltonian time evolution (dual APIs)
- **⚙️ Optimizer Factory**: Convenient optimizer creation utilities
- **🔬 Experimental Tools**: Parameter sweep utilities for algorithm exploration
- **🔬 Exact Solvers**: Exact diagonalization for benchmarking and validation (small systems)
- **🏗️ Pre-built Models**: Standard quantum many-body models (spin chains, chemistry models)
- **🧬 Fermion-to-Qubit Mappings**: Jordan-Wigner and Bravyi-Kitaev transforms for quantum chemistry
- **🔬 Quantum State Tomography**: Density matrix reconstruction from Pauli measurements
- **⚙️ Circuit Transpilation**: Gate decomposition and basis set conversion for hardware compatibility

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.1 or higher

### Install from PyPI (Recommended)

```bash
pip install qconduit
```

### Install from Source

For the latest development version:

```bash
git clone https://github.com/seansimms/Quantum_Conduit.git
cd Quantum_Conduit
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
import qconduit as qc
probs = qc.measure_probs_dm(rho)
print(f"Noisy probabilities: {probs}")
```

### Example 5: Circuit IR

```python
from qconduit.circuit import QuantumCircuit

# Create and simulate a Bell state circuit
circuit = QuantumCircuit(n_qubits=2)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [1, 0])

# Simulate the circuit
state = circuit.simulate_state()

# Visualize the circuit
print(circuit.to_text_diagram())
# Output:
# q0: ─H──⊕─
# q1: ────●─

# Analyze circuit properties
print(f"Depth: {circuit.depth()}")  # 2
print(f"Gate counts: {circuit.gate_counts()}")  # {'H': 1, 'CNOT': 1}
print(f"Number of gates: {circuit.num_gates()}")  # 2

# Parametric gates
circuit2 = QuantumCircuit(n_qubits=1)
circuit2.add_gate("RX", [0], params=[0.5])  # Rotation gate with angle
state2 = circuit2.simulate_state()
```

### Example 6: Diagnostics and Debug Mode

```python
import qconduit as qc
from qconduit.diagnostics import state_norm, fidelity, bloch_vector, assert_normalized

# Check state normalization
state = qc.zero_state(n_qubits=1)
norm = state_norm(state)
print(f"State norm: {norm}")  # 1.0
assert_normalized(state)  # Validates norm ≈ 1

# Compute fidelity between states
state1 = qc.zero_state(n_qubits=1)
h_gate = qc.H()
state2 = qc.apply_gate(state1, h_gate, qubit=0, n_qubits=1)
f = fidelity(state1, state2)
print(f"Fidelity: {f}")  # 0.5 (states are orthogonal)

# Compute Bloch vector for single-qubit state
bloch = bloch_vector(state2)
print(f"Bloch vector (x, y, z): {bloch}")  # [1.0, 0.0, 0.0] for |+⟩

# Enable debug mode for validation
qc.set_debug_enabled(True)
# Operations now include automatic normalization checks
# This helps catch bugs during development

# Use context manager for temporary debug mode
with qc.debug_context(True):
    # Debug checks enabled here
    state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=1)
# Debug mode restored to previous state
```

### Example 7: QAOA for MaxCut

```python
from qconduit.algorithms import QAOAAnsatz, ising_maxcut_hamiltonian, Edge, VQE
import torch

# Define a graph (triangle: 3 nodes, 3 edges)
edges = [Edge(0, 1), Edge(1, 2), Edge(2, 0)]
hamiltonian = ising_maxcut_hamiltonian(num_nodes=3, edges=edges)

# Create QAOA ansatz (p is the number of QAOA layers)
qaoa = QAOAAnsatz(n_qubits=3, problem_hamiltonian=hamiltonian, p=2)

# Optimize with VQE
vqe = VQE(ansatz=qaoa, hamiltonian=hamiltonian)
params = torch.nn.Parameter(0.1 * torch.randn(qaoa.num_parameters))

optimizer = torch.optim.Adam([params], lr=0.1)
for step in range(50):
    optimizer.zero_grad()
    energy = vqe.energy(params)
    energy.backward()
    optimizer.step()
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}: energy = {energy.item():.6f}")
```

### Example 8: VQE Training with Callbacks

```python
from qconduit.training import VQETrainer, TrainingCallback, EarlyStoppingConfig
from qconduit.algorithms import VQE
import torch

# Define callback for logging
class LoggingCallback(TrainingCallback):
    def __call__(self, info):
        if info.step % 10 == 0:
            print(f"Step {info.step}: energy = {info.energy:.6f}")

# Configure early stopping
early_stop = EarlyStoppingConfig(patience=10, min_delta=1e-6)

# Create VQE and optimizer
vqe = VQE(ansatz=ansatz, hamiltonian=hamiltonian)
params = torch.nn.Parameter(0.1 * torch.randn(ansatz.num_parameters))
optimizer = torch.optim.Adam([params], lr=0.1)

# Train VQE with callbacks and early stopping
trainer = VQETrainer(vqe, optimizer=optimizer)
history = trainer.train(
    params,
    max_steps=100,
    callbacks=[LoggingCallback()],
    early_stopping=early_stop,
)

print(f"Best energy: {history.best_energy():.6f}")
print(f"Final energy: {history.final_energy():.6f}")
```

### Example 9: Sampling and Analysis

```python
from qconduit.sampling import sample_bitstrings_state, bitstring_counts, kl_divergence
import qconduit as qc

# Create a quantum state
state = qc.zero_state(n_qubits=3)
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=3)
state = qc.apply_gate(state, qc.H(), qubit=1, n_qubits=3)

# Sample bitstrings from the state
samples = sample_bitstrings_state(state, n_qubits=3, n_shots=1000)

# Count occurrences
counts = bitstring_counts(samples)
print(f"Sample counts: {counts}")

# Compare probability distributions using KL divergence
probs1 = qc.measure_probs(state, n_qubits=3)
probs2 = qc.measure_probs(qc.zero_state(n_qubits=3), n_qubits=3)
# Convert probability tensors to dictionaries for kl_divergence
probs1_dict = {format(i, f'0{3}b'): float(probs1[i].item()) for i in range(len(probs1))}
probs2_dict = {format(i, f'0{3}b'): float(probs2[i].item()) for i in range(len(probs2))}
kl = kl_divergence(probs1_dict, probs2_dict)
print(f"KL divergence: {kl:.6f}")
```

### Example 10: Time Evolution

```python
from qconduit.time_evolution import time_evolve_state, build_trotter_circuit
from qconduit.operators import PauliTerm, PauliSum
import qconduit as qc

# Create a simple Hamiltonian (transverse field Ising model)
hamiltonian = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "Z")),  # Interaction
    PauliTerm(0.5, ("X", "I")),  # Transverse field
    PauliTerm(0.5, ("I", "X")),
])

# Evolve state under the Hamiltonian
state = qc.zero_state(n_qubits=2)
evolved_state = time_evolve_state(
    state, hamiltonian, t=0.5, n_steps=10, n_qubits=2
)

# Build Trotter circuit for the same evolution
circuit = build_trotter_circuit(
    hamiltonian, t=0.5, n_steps=10, n_qubits=2, order=1  # First-order Trotter
)
state_from_circuit = circuit.simulate_state()

print("Time evolution complete")
```

### Example 11: Exact Diagonalization

```python
from qconduit.exact import exact_eigensystem, exact_ground_state, paulisum_to_dense
from qconduit.operators import PauliTerm, PauliSum
import torch

# Create a simple Hamiltonian
hamiltonian = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "Z")),
    PauliTerm(0.5, ("X", "I")),
    PauliTerm(0.5, ("I", "X")),
])

# Convert to dense matrix
dense_matrix = paulisum_to_dense(hamiltonian, num_qubits=2)
print(f"Dense matrix shape: {dense_matrix.shape}")  # (4, 4)

# Compute full eigensystem
eigenvalues, eigenvectors = exact_eigensystem(hamiltonian, num_qubits=2)
print(f"Eigenvalues: {eigenvalues}")

# Get just the ground state
ground_energy, ground_state = exact_ground_state(hamiltonian, num_qubits=2)
print(f"Ground state energy: {ground_energy.item():.6f}")
```

### Example 12: Pre-built Models

```python
from qconduit.models import (
    transverse_field_ising_chain,
    heisenberg_xxz_chain,
    ising_zz_chain,
    two_qubit_generic_chemistry_like,
    diagonal_z_field,
)

# Transverse field Ising model (TFIM)
tfim = transverse_field_ising_chain(
    num_sites=4,
    j_coupling=1.0,
    h_field=0.5,
    periodic=True  # Periodic boundary conditions
)

# Heisenberg XXZ chain
heisenberg = heisenberg_xxz_chain(
    num_sites=3,
    j_coupling=1.0,
    delta=0.5,  # Anisotropy parameter
    periodic=False
)

# Ising ZZ chain (no transverse field)
ising = ising_zz_chain(
    num_sites=4,
    j_coupling=1.0,
    periodic=True
)

# Two-qubit chemistry-like model
chemistry_ham = two_qubit_generic_chemistry_like(
    c_i=0.0,      # Identity coefficient
    c_z0=0.5,     # Z⊗I coefficient
    c_z1=0.3,     # I⊗Z coefficient
    c_z0z1=0.1,   # Z⊗Z coefficient
    c_xx=0.0,     # X⊗X coefficient
    c_yy=0.0      # Y⊗Y coefficient
)

# Diagonal Z field
z_field = diagonal_z_field(num_qubits=3, local_fields=[0.5, 0.5, 0.5])

# Use with VQE or exact diagonalization
from qconduit.exact import exact_ground_state
energy, state = exact_ground_state(tfim, num_qubits=4)
print(f"TFIM ground energy: {energy.item():.6f}")
```

### Example 13: Adiabatic Evolution

```python
from qconduit.adiabatic import (
    AdiabaticConfig,
    linear_schedule,
    polynomial_schedule,
    adiabatic_evolve_state,
    build_adiabatic_circuit,
    build_x_mixer_hamiltonian,
    interpolate_paulisum,
)
from qconduit.operators import PauliSum, PauliTerm
import qconduit as qc
import torch

# Define initial (mixer) and final (problem) Hamiltonians
h_mixer = build_x_mixer_hamiltonian(num_qubits=3)  # -sum_i X_i
h_problem = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "Z", "I")),
    PauliTerm(1.0, ("I", "Z", "Z")),
])

# Create schedule (linear interpolation)
num_steps = 20
schedule = linear_schedule(num_steps)

# Configure adiabatic evolution
config = AdiabaticConfig(
    total_time=1.0,
    num_steps=num_steps,
    schedule=schedule,
    trotter_steps_per_interval=5
)

# Evolve state adiabatically
initial_state = qc.zero_state(n_qubits=3)
# Prepare |+⟩^⊗n state
for i in range(3):
    initial_state = qc.apply_gate(initial_state, qc.H(), qubit=i, n_qubits=3)

final_state = adiabatic_evolve_state(
    initial_state,
    h_mixer,
    h_problem,
    config
)

# Build adiabatic circuit for visualization
circuit = build_adiabatic_circuit(
    n_qubits=3, h_mixer=h_mixer, h_problem=h_problem, config=config
)
print(circuit.to_text_diagram())
```

### Example 14: Fermion-to-Qubit Mappings

```python
from qconduit.fermion import (
    FermionOperator,
    FermionTerm,
    FermionOpSymbol,
    jordan_wigner,
    bravyi_kitaev,
)

# Create a fermionic operator (e.g., a^†_0 a_1 + a^†_1 a_0)
term1 = FermionTerm(
    coeff=1.0,
    operators=((0, "+"), (1, "-"))  # a^†_0 a_1
)
term2 = FermionTerm(
    coeff=1.0,
    operators=((1, "+"), (0, "-"))  # a^†_1 a_0
)

fermion_op = FermionOperator([term1, term2])

# Map to qubits using Jordan-Wigner transform
jw_hamiltonian = jordan_wigner(fermion_op, n_spin_orbitals=2)
print(f"Jordan-Wigner: {len(jw_hamiltonian.terms)} Pauli terms")

# Map to qubits using Bravyi-Kitaev transform
bk_hamiltonian = bravyi_kitaev(fermion_op, n_spin_orbitals=2)
print(f"Bravyi-Kitaev: {len(bk_hamiltonian.terms)} Pauli terms")

# Use the mapped Hamiltonian with VQE or exact diagonalization
from qconduit.exact import exact_ground_state
energy, state = exact_ground_state(jw_hamiltonian, num_qubits=2)
print(f"Ground energy: {energy.item():.6f}")
```

### Example 15: Noisy Circuit Simulation

```python
from qconduit.noise import NoiseConfig, simulate_noisy_circuit_dm, sample_noisy_circuit_dm
from qconduit.circuit import QuantumCircuit
from qconduit.noise import DepolarizingChannel
import qconduit as qc

# Create a quantum circuit
circuit = QuantumCircuit(n_qubits=2)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])
circuit.add_gate("RX", [1], params=[0.5])

# Configure noise: depolarizing noise on qubit 0, amplitude damping on qubit 1
from qconduit.noise import AmplitudeDampingChannel
noise_config = NoiseConfig(
    per_qubit_channels={
        0: DepolarizingChannel(p=0.01),  # 1% depolarizing on qubit 0
        1: AmplitudeDampingChannel(gamma=0.05),  # Amplitude damping on qubit 1
    }
)

# Simulate noisy circuit (returns density matrix)
rho = simulate_noisy_circuit_dm(circuit, noise=noise_config)
print(f"Density matrix shape: {rho.shape}")  # (4, 4)

# Sample bitstrings from noisy circuit
samples = sample_noisy_circuit_dm(
    circuit,
    noise=noise_config,
    n_shots=1000
)

# Analyze results
from qconduit.sampling import bitstring_counts
counts = bitstring_counts(samples)
print(f"Sample distribution: {counts}")
```

### Example 16: Quantum State Tomography

```python
from qconduit.measurement import (
    single_qubit_pauli_expectations_from_statevector,
    reconstruct_single_qubit_density_from_pauli,
    two_qubit_pauli_expectations_from_statevector,
    reconstruct_two_qubit_density_from_pauli,
    pauli_expectation_from_statevector,
)
import qconduit as qc

# Create a quantum state
state = qc.zero_state(n_qubits=1)
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=1)
state = qc.apply_gate(state, qc.RY(0.5), qubit=0, n_qubits=1)

# Measure Pauli expectations
ex_x, ex_y, ex_z = single_qubit_pauli_expectations_from_statevector(state)
print(f"Pauli expectations: X={ex_x:.4f}, Y={ex_y:.4f}, Z={ex_z:.4f}")

# Reconstruct density matrix from measurements
rho_reconstructed = reconstruct_single_qubit_density_from_pauli(ex_x, ex_y, ex_z)

# Compare with actual density matrix
rho_actual = qc.dm_from_statevector(state)
fidelity = qc.fidelity(rho_actual, rho_reconstructed)
print(f"Reconstruction fidelity: {fidelity.item():.6f}")

# Two-qubit tomography
state_2q = qc.zero_state(n_qubits=2)
state_2q = qc.apply_gate(state_2q, qc.H(), qubit=0, n_qubits=2)
state_2q = qc.apply_two_qubit_gate(state_2q, qc.CNOT(), qubit1=0, qubit2=1, n_qubits=2)

# Get all two-qubit Pauli expectations
pauli_expectations = two_qubit_pauli_expectations_from_statevector(state_2q)
rho_2q = reconstruct_two_qubit_density_from_pauli(pauli_expectations)
```

### Example 17: Variational Algorithm Scaffolding

```python
from qconduit.variational import (
    VariationalAnsatz,
    HardwareEfficientAnsatz,
    LayeredEntanglerAnsatz,
    run_vqe,
    run_qaoa,
    VQEResult,
    QAOAResult,
)
from qconduit.operators import PauliSum, PauliTerm
import torch

# Create a Hamiltonian
hamiltonian = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "Z")),
    PauliTerm(0.5, ("X", "I")),
    PauliTerm(0.5, ("I", "X")),
])

# High-level VQE API
ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=2)
initial_params = torch.randn(ansatz.num_parameters)

result = run_vqe(
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    initial_params=initial_params,
    max_iterations=100,
    learning_rate=0.1,
)

print(f"Ground state energy: {result.optimal_value:.6f}")
print(f"Optimal parameters: {result.optimal_params}")
print(f"Converged: {result.converged}")

# High-level QAOA API
qaoa_result = run_qaoa(
    cost_hamiltonian=hamiltonian,
    num_qubits=2,
    depth=2,
    max_iterations=100,
    learning_rate=0.05,
)

print(f"QAOA optimal energy: {qaoa_result.optimal_value:.6f}")
```

### Example 18: Circuit Transpilation

```python
from qconduit.transpile import (
    decompose_h_to_rz_rx_rz,
    transpile_to_rx_rz_cx_basis,
    transpile_to_clifford_t,
    summarize_gate_counts,
    estimate_circuit_depth,
    GateCountSummary,
)
from qconduit.circuit import QuantumCircuit
import qconduit as qc

# Create a circuit with various gates
circuit = QuantumCircuit(n_qubits=3)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])
circuit.add_gate("T", [1])
circuit.add_gate("S", [2])
circuit.add_gate("RX", [0], params=[0.5])

# Transpile to RX, RZ, CNOT basis (common hardware basis)
transpiled = transpile_to_rx_rz_cx_basis(circuit)
print("Transpiled circuit:")
print(transpiled.to_text_diagram())

# Transpile to Clifford+T basis
clifford_t = transpile_to_clifford_t(circuit)
print("\nClifford+T circuit:")
print(clifford_t.to_text_diagram())

# Analyze gate counts
summary = summarize_gate_counts(clifford_t)
print(f"\nGate counts: {summary.counts}")
print(f"T-count: {summary.t_count}")
print(f"Clifford count: {summary.clifford_count}")

# Estimate circuit depth
depth = estimate_circuit_depth(transpiled)
print(f"Circuit depth: {depth}")
```

### Example 19: Enhanced Kraus Channels

```python
from qconduit.noise import (
    KrausChannel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
    compose_kraus_channels,
    apply_kraus_channel_to_statevector,
    apply_kraus_channel_to_density_matrix,
)
import qconduit as qc
import torch

# Create various noise channels
bit_flip = bit_flip_channel(p=0.01)  # 1% bit flip probability
phase_flip = phase_flip_channel(p=0.02)  # 2% phase flip probability
bit_phase_flip = bit_phase_flip_channel(p=0.005)  # 0.5% bit-phase flip

# Generalized amplitude damping (with temperature)
amp_damp = generalized_amplitude_damping_channel(
    gamma=0.1,  # Damping rate
    n_th=0.1,   # Thermal population
)

# Two-qubit depolarizing channel
two_qubit_depol = two_qubit_depolarizing_channel(p=0.01)

# Compose channels (apply sequentially)
combined = compose_kraus_channels(bit_flip, phase_flip)

# Apply to statevector
state = qc.zero_state(n_qubits=1)
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=1)
noisy_state = apply_kraus_channel_to_statevector(
    state, bit_flip, qubit=0, n_qubits=1
)

# Apply to density matrix
rho = qc.dm_from_statevector(state)
noisy_rho = apply_kraus_channel_to_density_matrix(
    rho, phase_flip, qubit=0, n_qubits=1
)

# Check channel properties
print(f"Bit flip channel is trace-preserving: {bit_flip.is_trace_preserving()}")
print(f"Kraus operators: {len(bit_flip.kraus_ops)}")
```

### Example 20: Exact Time Evolution

```python
from qconduit.evolution import (
    exact_time_evolution_statevector,
    TrotterOrder,
    TrotterSchedule,
    evolve_state_trotter,
)
from qconduit.operators import PauliSum, PauliTerm
import qconduit as qc

# Create a Hamiltonian
hamiltonian = PauliSum.from_terms([
    PauliTerm(1.0, ("Z", "Z")),
    PauliTerm(0.5, ("X", "I")),
    PauliTerm(0.5, ("I", "X")),
])

# Exact time evolution (for small systems)
state = qc.zero_state(n_qubits=2)
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=2)

evolved_exact = exact_time_evolution_statevector(
    state, hamiltonian, time=0.5
)

# Trotterized evolution (for larger systems)
from qconduit.evolution import TrotterOrder, TrotterSchedule

schedule = TrotterSchedule(
    num_steps=10,     # Number of steps
    total_time=0.5,   # Total evolution time
    order=1,          # TrotterOrder.FIRST (1) or TrotterOrder.SECOND (2)
)

evolved_trotter = evolve_state_trotter(
    state,
    hamiltonian,
    schedule,
)

# Compare results
fidelity = qc.fidelity(
    qc.dm_from_statevector(evolved_exact),
    qc.dm_from_statevector(evolved_trotter)
)
print(f"Fidelity between exact and Trotter: {fidelity.item():.6f}")
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
├── circuit/        # Circuit IR (GateOp, QuantumCircuit)
├── layers/         # Parametric ansätze and hybrid blocks
├── algorithms/     # Quantum algorithms (VQE, QAOA)
├── operators/      # Pauli operators and expectations
├── grad/           # Gradient computation (parameter-shift)
├── noise/          # Noise models and quantum channels
├── diagnostics/    # State validation and debugging tools
├── training/       # Training loops and utilities
├── sampling/       # Bitstring sampling and analysis
├── time_evolution/ # Trotterization and time evolution
├── evolution/      # Alternative evolution API (exact + enhanced Trotter)
├── optim/          # Optimizer factory utilities
├── experiments/    # Parameter sweep utilities
├── exact/          # Exact diagonalization for small systems
├── models/         # Pre-built quantum many-body models
├── adiabatic/      # Adiabatic quantum computing
├── fermion/        # Fermion-to-qubit mappings
├── measurement/    # Measurement and quantum state tomography
├── variational/    # Variational algorithm scaffolding
└── transpile/      # Gate decomposition and circuit transpilation
```

### Key Components

- **Device Abstraction**: Unified interface for CPU and CUDA operations
- **Statevector Backend**: Efficient pure-state simulation with O(2^n) memory
- **Density Matrix Backend**: Mixed-state simulation for noise (O(4^n) memory, small systems)
- **Gate Library**: Standard gates with gradient-preserving implementations
- **Module System**: `QuantumModule` base class compatible with PyTorch's module system
- **Circuit IR**: Structured circuit representation with simulation and visualization
- **Diagnostics**: State validation, fidelity computation, and debug mode integration
- **Training Infrastructure**: Complete training loops with callbacks and history tracking
- **Sampling**: Bitstring sampling and probability distribution analysis
- **Time Evolution**: Trotterization for Hamiltonian simulation
- **Evolution**: Alternative evolution API with exact and enhanced Trotter methods
- **Optimizers**: Factory utilities for optimizer creation
- **Experiments**: Parameter sweep utilities for algorithm exploration
- **Exact Solvers**: Exact diagonalization for benchmarking and validation
- **Pre-built Models**: Standard quantum many-body models (spin chains, chemistry)
- **Adiabatic Evolution**: Adiabatic quantum computing with configurable schedules
- **Fermion-to-Qubit Mappings**: Jordan-Wigner and Bravyi-Kitaev transforms
- **Measurement/Tomography**: Quantum state tomography and Pauli expectation measurements
- **Variational Scaffolding**: High-level APIs for VQE and QAOA algorithms
- **Transpilation**: Gate decomposition and basis set conversion for hardware

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
# Statevector operations
state = qc.zero_state(n_qubits=2, batch_shape=(10,))  # Batched states

# Gate application
gate = qc.H()
state = qc.apply_gate(state, gate, qubit=0, n_qubits=2)
state = qc.apply_two_qubit_gate(state, qc.CNOT(), qubit1=0, qubit2=1, n_qubits=2)

# Measurements
probs = qc.measure_probs(state, n_qubits=2)
z_exp = qc.measure_expectation_z(state, qubit=0, n_qubits=2)

# Density matrix operations (for noise modeling)
rho = qc.zero_dm_state(n_qubits=2)  # Create |00><00|
rho = qc.dm_from_statevector(state)  # Convert statevector to density matrix

# Apply Kraus operators (for noise channels)
kraus_ops = (E0, E1, E2)  # Tuple of 2x2 matrices
rho = qc.apply_kraus_single_qubit(rho, kraus_ops, qubit=0, n_qubits=2)

# Density matrix measurements
probs_dm = qc.measure_probs_dm(rho)
z_exp_dm = qc.measure_expectation_z_dm(rho, qubit=0, n_qubits=2)
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

### Circuit IR

```python
from qconduit.circuit import QuantumCircuit, GateOp

# Create a circuit
circuit = QuantumCircuit(n_qubits=3)

# Add gates
circuit.add_gate("H", [0])  # Hadamard on qubit 0
circuit.add_gate("CNOT", [0, 1])  # CNOT with control=0, target=1
circuit.add_gate("RX", [2], params=[0.5])  # Parametric rotation

# Circuit properties
n_gates = circuit.num_gates()  # Number of gates
gate_counts = circuit.gate_counts()  # Dict: {"H": 1, "CNOT": 1, "RX": 1}
depth = circuit.depth()  # Circuit depth (parallel gate scheduling)

# Simulate circuit
state = circuit.simulate_state()  # Returns statevector

# Visualize circuit
diagram = circuit.to_text_diagram()
print(diagram)
# q0: ─H──●───────
# q1: ────⊕───────
# q2: ────────R───

# Copy circuit
circuit_copy = circuit.copy()

# Access operations
for op in circuit.ops:
    print(f"{op.name} on qubits {op.qubits} with params {op.params}")
```

### Diagnostics

```python
import qconduit as qc
from qconduit.diagnostics import (
    state_norm,
    assert_normalized,
    is_hermitian,
    assert_hermitian,
    fidelity,
    bloch_vector,
    is_debug_enabled,
    set_debug_enabled,
    debug_context,
)

# State validation
state = qc.zero_state(n_qubits=2)
norm = state_norm(state)  # Compute L2 norm
assert_normalized(state, atol=1e-5)  # Assert norm ≈ 1

# Matrix validation
matrix = qc.H()  # Get a gate matrix
is_herm = is_hermitian(matrix)  # Check if Hermitian
assert_hermitian(matrix, atol=1e-6)  # Assert Hermitian

# Fidelity computation
state1 = qc.zero_state(n_qubits=1)
state2 = qc.apply_gate(state1, qc.H(), qubit=0, n_qubits=1)
f = fidelity(state1, state2)  # |<state1|state2>|²

# Bloch vector (single-qubit only)
bloch = bloch_vector(state2)  # Returns (x, y, z) components

# Debug mode management
is_enabled = is_debug_enabled()  # Check current status
set_debug_enabled(True)  # Enable globally

# Context manager for temporary debug mode
with debug_context(True):
    # Debug checks enabled here
    state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=1)
# Debug mode restored to previous state

# Environment variable: QCONDUIT_DEBUG=1 enables debug mode at startup
```

### QAOA

```python
from qconduit.algorithms import QAOAAnsatz, ising_maxcut_hamiltonian, Edge

# Define graph edges
edges = [Edge(0, 1, weight=1.0), Edge(1, 2, weight=0.5)]
# Or use tuples: edges = [(0, 1), (1, 2)]

# Build MaxCut Ising Hamiltonian
hamiltonian = ising_maxcut_hamiltonian(
    num_nodes=3,
    edges=edges,
    include_constant=True  # Include constant term in Hamiltonian
)

# Create QAOA ansatz (p is the number of QAOA layers)
qaoa = QAOAAnsatz(n_qubits=3, problem_hamiltonian=hamiltonian, p=2)
params = torch.randn(qaoa.num_parameters)
state = qaoa(params)  # Forward pass

# Use with VQE for optimization
from qconduit.algorithms import VQE
vqe = VQE(ansatz=qaoa, hamiltonian=hamiltonian)
energy = vqe.energy(params)
```

### Training

```python
from qconduit.training import (
    VQETrainer,
    TrainingHistory,
    TrainingCallback,
    TrainingStepInfo,
    EarlyStoppingConfig,
)

# Create trainer
trainer = VQETrainer(vqe, optimizer=optimizer)

# Define custom callback
class MyCallback(TrainingCallback):
    def __call__(self, info: TrainingStepInfo):
        # Access step, epoch, energy, loss, grad_norm, param_norm
        if info.step % 10 == 0:
            print(f"Step {info.step}: energy={info.energy:.6f}")

# Configure early stopping
early_stop = EarlyStoppingConfig(
    patience=10,      # Stop if no improvement for 10 steps
    min_delta=1e-6,   # Minimum change to count as improvement
)

# Train with callbacks and early stopping
history = trainer.train(
    params,
    max_steps=100,
    callbacks=[MyCallback()],
    early_stopping=early_stop,
)

# Access training history
best_energy = history.best_energy()
final_energy = history.final_energy()
num_steps = history.num_steps()

# Access individual steps
for step_info in history.steps:
    print(f"Step {step_info.step}: energy={step_info.energy}")
```

### Sampling

```python
from qconduit.sampling import (
    sample_bitstrings_state,
    sample_bitstrings_dm,
    sample_bitstrings_circuit,
    sample_from_probs,
    bitstring_counts,
    counts_to_probs,
    kl_divergence,
    marginalize_probs,
)

# Sample from statevector
samples = sample_bitstrings_state(
    state, n_qubits=3, n_shots=1000, qubits=None  # None = all qubits
)

# Sample from density matrix
samples_dm = sample_bitstrings_dm(rho, n_qubits=3, n_shots=1000)

# Sample from circuit
from qconduit.circuit import QuantumCircuit
circuit = QuantumCircuit(n_qubits=2)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])
samples_circuit = sample_bitstrings_circuit(circuit, n_shots=1000)

# Sample from probability distribution
probs = qc.measure_probs(state, n_qubits=3)
samples = sample_from_probs(probs, n_qubits=3, n_shots=1000)

# Count bitstring occurrences
counts = bitstring_counts(samples)
# Returns dict: {"000": 250, "001": 250, ...}

# Convert counts to probabilities
probs_from_counts = counts_to_probs(counts)

# Compute KL divergence between distributions
# Convert probability tensors to dictionaries
probs1_dict = {format(i, f'0{3}b'): float(probs1[i].item()) for i in range(len(probs1))}
probs2_dict = {format(i, f'0{3}b'): float(probs2[i].item()) for i in range(len(probs2))}
kl = kl_divergence(probs1_dict, probs2_dict)

# Marginalize probabilities (sum over some qubits)
marginal = marginalize_probs(probs, n_qubits=3, qubits_to_keep=[0, 1])  # Keep qubits 0,1
```

### Time Evolution

```python
from qconduit.time_evolution import (
    time_evolve_state,
    trotter_step_pauli_sum,
    build_trotter_step_circuit,
    build_trotter_circuit,
    OrderLiteral,
)

# Evolve state under Hamiltonian
evolved_state = time_evolve_state(
    state,
    hamiltonian,  # PauliSum
    t=0.5,        # Total evolution time
    n_steps=10,   # Number of Trotter steps
    n_qubits=2,   # Number of qubits
)

# Single Trotter step
state_after_step = trotter_step_pauli_sum(
    state, hamiltonian, dt=0.05, n_qubits=2, order=1  # order: 1 or 2
)

# Build Trotter circuit (for visualization or reuse)
circuit = build_trotter_circuit(
    hamiltonian,
    t=0.5,
    n_steps=10,
    n_qubits=2,
    order=1,  # or OrderLiteral.FIRST (1 or 2)
)

# Build single Trotter step circuit
step_circuit = build_trotter_step_circuit(
    hamiltonian, dt=0.05, n_qubits=2, order=1
)
```

### Optimizers

```python
from qconduit.optim import OptimConfig, create_optimizer

# Create optimizer configuration
config = OptimConfig(
    name="adam",      # Optimizer name: "adam", "sgd", "rmsprop", etc.
    lr=0.01,          # Learning rate
    weight_decay=0.0, # Weight decay (L2 regularization)
    # Additional optimizer-specific kwargs
    betas=(0.9, 0.999),  # For Adam
)

# Create optimizer from parameters
params = [torch.nn.Parameter(torch.randn(5))]
optimizer = create_optimizer(config, params)

# Use with training
for step in range(100):
    optimizer.zero_grad()
    loss = compute_loss(params)
    loss.backward()
    optimizer.step()
```

### Experiments

```python
from qconduit.experiments import (
    run_1d_sweep,
    run_2d_sweep,
    sweep_vqe_1d,
    sweep_vqe_2d,
    SweepResult1D,
    SweepResult2D,
)

# Generic 1D parameter sweep
def objective(params):
    return params[0] ** 2

result_1d = run_1d_sweep(
    objective,
    points=torch.linspace(0, 1, 50),
    base_params=torch.tensor([0.0]),  # Base parameter tensor
    index=0,  # Index of parameter to sweep
    metadata={"param_name": "x", "x_label": "Parameter"},
)

# Access results
print(f"Best value: {result_1d.values.min()}")
print(f"Best point: {result_1d.points[result_1d.values.argmin()]}")

# VQE-specific 1D sweep
vqe_result = sweep_vqe_1d(
    vqe,
    points=torch.linspace(0, 2 * torch.pi, 50),
    base_params=params_template,  # Template parameter tensor
    index=0,  # Index of parameter to sweep
)

# 2D parameter sweep
def objective_2d(params):
    return params[0] ** 2 + params[1] ** 2

result_2d = run_2d_sweep(
    objective_2d,
    x_points=torch.linspace(0, 1, 20),
    y_points=torch.linspace(0, 1, 20),
    metadata={"x_label": "X", "y_label": "Y"},
)

# Access 2D results
print(f"Values shape: {result_2d.values.shape}")  # (20, 20)
print(f"Best value: {result_2d.values.min()}")
```

### Exact Solvers

```python
from qconduit.exact import (
    paulisum_to_dense,
    exact_eigensystem,
    exact_ground_state,
)

# Convert PauliSum to dense matrix
dense_matrix = paulisum_to_dense(
    hamiltonian,  # PauliSum
    num_qubits=3,
    device=None,  # Optional, defaults to default_device()
    dtype=torch.complex128,  # Complex dtype
)

# Compute full eigensystem
eigenvalues, eigenvectors = exact_eigensystem(
    hamiltonian,
    num_qubits=3,
    k=None,  # Reserved for future use (subset of eigenpairs)
    device=None,
    dtype=torch.complex128,
)
# eigenvalues: shape (2**n_qubits,)
# eigenvectors: shape (2**n_qubits, 2**n_qubits), columns are eigenvectors

# Get just the ground state
ground_energy, ground_state = exact_ground_state(
    hamiltonian,
    num_qubits=3,
    device=None,
    dtype=torch.complex128,
)
# ground_energy: scalar tensor
# ground_state: shape (2**n_qubits,)
```

### Pre-built Models

```python
from qconduit.models import (
    transverse_field_ising_chain,
    heisenberg_xxz_chain,
    ising_zz_chain,
    two_qubit_generic_chemistry_like,
    diagonal_z_field,
)

# Transverse field Ising model (TFIM)
# H = -J * sum_{<i,j>} Z_i Z_j - h * sum_i X_i
tfim = transverse_field_ising_chain(
    num_sites=4,      # Number of spins
    j_coupling=1.0,   # ZZ coupling strength
    h_field=0.5,      # Transverse field strength
    periodic=True,    # Periodic boundary conditions
)

# Heisenberg XXZ chain
# H = J * sum_{<i,j>} (X_i X_j + Y_i Y_j + Δ Z_i Z_j)
heisenberg = heisenberg_xxz_chain(
    num_sites=3,
    j_coupling=1.0,   # Overall coupling
    delta=0.5,        # Anisotropy parameter
    periodic=False,   # Open chain
)

# Ising ZZ chain (no transverse field)
# H = -J * sum_{<i,j>} Z_i Z_j
ising = ising_zz_chain(
    num_sites=4,
    j_coupling=1.0,
    periodic=True,
)

# Two-qubit chemistry-like model
# Generic two-qubit Hamiltonian for chemistry applications
chemistry_ham = two_qubit_generic_chemistry_like(
    c_i=0.0,      # Identity coefficient
    c_z0=0.5,     # Z⊗I coefficient
    c_z1=0.3,     # I⊗Z coefficient
    c_z0z1=0.1,   # Z⊗Z coefficient
    c_xx=0.0,     # X⊗X coefficient
    c_yy=0.0      # Y⊗Y coefficient
)

# Diagonal Z field
# H = sum_i h_i Z_i where h_i are the local field strengths
z_field = diagonal_z_field(
    num_qubits=3,
    local_fields=[0.5, 0.5, 0.5],  # Field strength for each qubit
)
```

### Adiabatic Evolution

```python
from qconduit.adiabatic import (
    ScheduleFn,
    linear_schedule,
    polynomial_schedule,
    sample_schedule,
    AdiabaticConfig,
    interpolate_paulisum,
    adiabatic_evolve_state,
    build_adiabatic_circuit,
    build_x_mixer_hamiltonian,
    adiabatic_x_mixer_to_problem_state,
)

# Create schedules
schedule_linear = linear_schedule(num_steps=20)  # Linear s(t) = t/T
schedule_poly = polynomial_schedule(num_steps=20, power=2)  # Polynomial s(t) = (t/T)^p

# Custom schedule function
def custom_schedule(num_steps: int) -> torch.Tensor:
    # Return 1D tensor of shape (num_steps,) with values in [0, 1]
    return torch.linspace(0, 1, num_steps) ** 0.5

# Build X mixer Hamiltonian: H_mixer = -sum_i X_i
h_mixer = build_x_mixer_hamiltonian(num_qubits=3)

# Interpolate between two Hamiltonians
h_interpolated = interpolate_paulisum(
    h_initial,  # Initial Hamiltonian
    h_final,    # Final Hamiltonian
    s=0.5,      # Interpolation parameter in [0, 1]
)

# Configure adiabatic evolution
config = AdiabaticConfig(
    total_time=1.0,              # Total evolution time
    num_steps=20,                # Number of discrete steps
    schedule=linear_schedule(20), # Schedule function
    trotter_steps_per_interval=5, # Trotter steps per interval
)

# Evolve state adiabatically
final_state = adiabatic_evolve_state(
    initial_state,  # Initial statevector
    h_mixer,        # Initial (mixer) Hamiltonian
    h_problem,      # Final (problem) Hamiltonian
    config          # AdiabaticConfig
)

# Build adiabatic circuit
circuit = build_adiabatic_circuit(
    n_qubits=3,
    h_mixer=h_mixer,
    h_problem=h_problem,
    config=config
)

# Prepare ground state of X mixer (|+⟩^⊗n)
initial_state = qc.zero_state(n_qubits=3)
for i in range(3):
    initial_state = qc.apply_gate(initial_state, qc.H(), qubit=i, n_qubits=3)
ground_state = adiabatic_x_mixer_to_problem_state(
    initial_state,
    h_problem,
    config
)
```

### Fermion-to-Qubit Mappings

```python
from qconduit.fermion import (
    FermionOpSymbol,
    FermionTerm,
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
)

# Create fermionic operators
# FermionOpSymbol: (mode_index, op_type) where op_type is "+" (creation) or "-" (annihilation)
term1 = FermionTerm(
    coeff=1.0,
    operators=((0, "+"), (1, "-"))  # a^†_0 a_1
)

term2 = FermionTerm(
    coeff=0.5,
    operators=((1, "+"), (0, "-"), (2, "+"), (2, "-"))  # 0.5 * a^†_1 a_0 a^†_2 a_2
)

# Build FermionOperator (sum of terms)
fermion_op = FermionOperator([term1, term2])

# Map to qubits using Jordan-Wigner transform
jw_hamiltonian = jordan_wigner(
    fermion_op,
    n_spin_orbitals=3,  # Number of fermionic modes (spin-orbitals)
)

# Map to qubits using Bravyi-Kitaev transform
bk_hamiltonian = bravyi_kitaev(
    fermion_op,
    n_spin_orbitals=3,
)

# Both return PauliSum that can be used with VQE, exact diagonalization, etc.
```

### Evolution Module (Alternative API)

```python
from qconduit.evolution import (
    exact_time_evolution_statevector,
    TrotterOrder,
    TrotterSchedule,
    evolve_state_trotter,
    build_trotter_step_circuit,
    build_trotter_circuit,
)

# Exact time evolution (for small systems, uses dense matrix exponentiation)
evolved = exact_time_evolution_statevector(
    state,        # Initial statevector
    hamiltonian,  # PauliSum Hamiltonian
    time=0.5,     # Evolution time
    device=None,  # Optional device
)

# Enhanced Trotter evolution with schedule
schedule = TrotterSchedule(
    num_steps=10,     # Number of Trotter steps
    total_time=0.5,   # Total evolution time
    order=1,          # TrotterOrder.FIRST (1) or TrotterOrder.SECOND (2)
)

evolved_trotter = evolve_state_trotter(
    state,
    hamiltonian,
    schedule,
)

# Build Trotter circuits
step_circuit = build_trotter_step_circuit(hamiltonian, schedule.step_time, schedule.order, num_qubits=2)
full_circuit = build_trotter_circuit(hamiltonian, schedule, num_qubits=2)
```

### Measurement and Quantum State Tomography

```python
from qconduit.measurement import (
    # Sampling utilities
    basis_probabilities_from_statevector,
    sample_bitstrings_from_probabilities,
    sample_bitstrings_from_statevector,
    bitstring_counts,
    empirical_probabilities_from_bitstrings,
    estimate_pauli_z_expectation_from_samples,
    # Pauli expectation values
    pauli_matrix_from_label,
    pauli_expectation_from_statevector,
    single_qubit_pauli_expectations_from_statevector,
    two_qubit_pauli_expectations_from_statevector,
    # State tomography
    reconstruct_single_qubit_density_from_pauli,
    reconstruct_two_qubit_density_from_pauli,
)

# Get basis probabilities
probs = basis_probabilities_from_statevector(state)

# Sample bitstrings
samples = sample_bitstrings_from_statevector(state, n_shots=1000)

# Compute Pauli expectation values
ex_x = pauli_expectation_from_statevector(state, "X")
ex_zz = pauli_expectation_from_statevector(state, "ZZ")

# Single-qubit tomography
ex_x, ex_y, ex_z = single_qubit_pauli_expectations_from_statevector(state)
rho = reconstruct_single_qubit_density_from_pauli(ex_x, ex_y, ex_z)

# Two-qubit tomography
pauli_expectations = two_qubit_pauli_expectations_from_statevector(state_2q)
rho_2q = reconstruct_two_qubit_density_from_pauli(pauli_expectations)

# Estimate expectation from samples
z_expectation, std_error = estimate_pauli_z_expectation_from_samples(samples, qubit_index=0)
```

### Variational Algorithm Scaffolding

```python
from qconduit.variational import (
    VariationalAnsatz,
    HardwareEfficientAnsatz,
    LayeredEntanglerAnsatz,
    QAOAAnsatz,
    run_vqe,
    run_qaoa,
    VQEResult,
    QAOAResult,
    evaluate_expectation_value,
)

# High-level VQE API
from qconduit.variational import HardwareEfficientAnsatz
import torch

ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=2)
initial_params = torch.randn(ansatz.num_parameters)

result = run_vqe(
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    initial_params=initial_params,
    optimizer_name="adam",  # or "sgd"
    max_iterations=200,
    learning_rate=0.1,
    tol_rel=1e-6,  # Relative tolerance for convergence
    device=None,
)

# Access results
print(f"Optimal energy: {result.optimal_value}")
print(f"Optimal parameters: {result.optimal_params}")
print(f"Converged: {result.converged}")
print(f"Number of iterations: {result.num_iterations}")

# High-level QAOA API
qaoa_result = run_qaoa(
    cost_hamiltonian=hamiltonian,
    num_qubits=3,
    depth=2,
    initial_params=None,
    optimizer_name="adam",
    max_iterations=200,
    learning_rate=0.05,
    tol_rel=1e-6,
    device=None,
)

# Evaluate expectation value for custom ansatz
ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=2)
params = torch.randn(ansatz.num_parameters)
energy = evaluate_expectation_value(ansatz, params, hamiltonian)
```

### Circuit Transpilation

```python
from qconduit.transpile import (
    # Gate decomposition
    decompose_h_to_rz_rx_rz,
    decompose_x_to_rx,
    decompose_y_to_ry,
    decompose_z_to_rz,
    decompose_rz_to_clifford_t,
    decompose_gate_to_basis,
    # Basis transpilation
    transpile_to_basis,
    transpile_to_rx_rz_cx_basis,
    transpile_to_clifford_t,
    # Circuit analysis
    GateCountSummary,
    summarize_gate_counts,
    estimate_circuit_depth,
)

# Decompose individual gates
circuit = QuantumCircuit(n_qubits=2)
circuit.add_gate("H", [0])
decompose_h_to_rz_rx_rz(circuit, qubit=0)  # Modifies circuit in-place

# Transpile to specific basis
rx_rz_cx = transpile_to_rx_rz_cx_basis(circuit)  # Returns new circuit
clifford_t = transpile_to_clifford_t(circuit)  # Returns new circuit

# Transpile to custom basis
custom_basis = transpile_to_basis(circuit, basis_gates=["RX", "RZ", "CNOT"])

# Analyze circuits
summary = summarize_gate_counts(circuit)
print(f"Gate counts: {summary.counts}")
print(f"T-count: {summary.t_count}")
print(f"Clifford count: {summary.clifford_count}")
print(f"Total gates: {summary.total_gates}")

# Estimate depth
depth = estimate_circuit_depth(circuit)
print(f"Circuit depth: {depth}")
```

### Enhanced Kraus Channels

```python
from qconduit.noise import (
    KrausChannel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
    to_density_matrix,
    apply_kraus_channel_to_density_matrix,
    apply_kraus_channel_to_statevector,
    compose_kraus_channels,
)

# Create standard noise channels
bit_flip = bit_flip_channel(p=0.01)  # p in [0, 1]
phase_flip = phase_flip_channel(p=0.02)
bit_phase_flip = bit_phase_flip_channel(p=0.005)

# Generalized amplitude damping (with thermal population)
amp_damp = generalized_amplitude_damping_channel(
    gamma=0.1,  # Damping rate
    n_th=0.1,   # Thermal population (0 = zero temperature)
)

# Two-qubit depolarizing channel
two_qubit_depol = two_qubit_depolarizing_channel(p=0.01)

# Create custom Kraus channel
kraus_ops = (K0, K1, K2)  # Tuple of Kraus operators
custom_channel = KrausChannel(
    name="custom",
    kraus_ops=kraus_ops,
    num_qubits=1,
)

# Compose multiple channels
combined = compose_kraus_channels([bit_flip, phase_flip])

# Apply to statevector
noisy_state = apply_kraus_channel_to_statevector(
    state, channel, qubit=0, n_qubits=2
)

# Apply to density matrix
noisy_rho = apply_kraus_channel_to_density_matrix(
    rho, channel, qubit=0, n_qubits=2
)

# Convert statevector to density matrix
rho = to_density_matrix(state)

# Check channel properties
is_tp = channel.is_trace_preserving()  # Check trace-preserving property
```

### Enhanced Noise Models

```python
from qconduit.noise import (
    NoiseConfig,
    simulate_noisy_circuit_dm,
    sample_noisy_circuit_dm,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)
from qconduit.circuit import QuantumCircuit

# Configure per-qubit noise channels
noise_config = NoiseConfig(
    per_qubit_channels={
        0: DepolarizingChannel(p=0.01),      # 1% depolarizing on qubit 0
        1: AmplitudeDampingChannel(gamma=0.05), # Amplitude damping on qubit 1
        2: PhaseDampingChannel(gamma=0.02),     # Phase damping on qubit 2
    }
)

# Simulate noisy circuit (returns density matrix)
circuit = QuantumCircuit(n_qubits=3)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])

rho = simulate_noisy_circuit_dm(
    circuit,
    noise=noise_config,
)
# Returns density matrix of shape (2**n_qubits, 2**n_qubits)

# Sample bitstrings from noisy circuit
samples = sample_noisy_circuit_dm(
    circuit,
    noise=noise_config,
    n_shots=1000,  # Number of samples
)
# Returns tensor of shape (n_samples, n_qubits) with bitstrings
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

## Use Cases

### When to Use Circuit IR vs Direct Gate Application

**Use Circuit IR (`QuantumCircuit`)** when:
- You need to **visualize** circuits with `to_text_diagram()`
- You want to **analyze** circuit properties (depth, gate counts)
- You're building circuits **dynamically** or from external specifications
- You need to **copy** or **modify** circuits before simulation
- You're working with **circuit optimization** or compilation

**Use direct gate application** when:
- You need **maximum performance** (no IR overhead)
- You're building circuits **statically** in code
- You want **direct control** over state manipulation
- You're working with **batched operations** (Circuit IR doesn't support batching yet)

**Example: Circuit IR for visualization**
```python
from qconduit.circuit import QuantumCircuit

circuit = QuantumCircuit(n_qubits=3)
circuit.add_gate("H", [0])
circuit.add_gate("CNOT", [0, 1])
circuit.add_gate("CNOT", [1, 2])
print(circuit.to_text_diagram())
# Great for debugging and documentation!
```

**Example: Direct gates for performance**
```python
import qconduit as qc

# More efficient for tight loops
state = qc.zero_state(n_qubits=3, batch_shape=(100,))  # Batched
for i in range(100):
    state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=3)
```

### Debug Mode Best Practices

**Enable debug mode during development:**
```python
import qconduit as qc

# Global enable for development
qc.set_debug_enabled(True)

# Or use environment variable
# QCONDUIT_DEBUG=1 python your_script.py
```

**Use context managers for specific sections:**
```python
# Only enable for critical sections
with qc.debug_context(True):
    # Critical quantum operations
    state = complex_quantum_operation(state)
# Automatically disabled after context
```

**Debug mode automatically:**
- Validates state normalization after gate applications
- Helps catch bugs early in development
- Has minimal overhead when disabled (production-ready)

### Diagnostics for Validation and Debugging

**State validation:**
```python
from qconduit.diagnostics import assert_normalized, state_norm

# Validate states in test suites
def test_my_quantum_function():
    state = my_quantum_function()
    assert_normalized(state)  # Raises if not normalized
    assert state_norm(state).item() == pytest.approx(1.0)
```

**Fidelity for algorithm verification:**
```python
from qconduit.diagnostics import fidelity

# Compare expected vs actual states
expected = create_expected_state()
actual = run_algorithm()
f = fidelity(expected, actual)
assert f > 0.99  # High fidelity means correct implementation
```

**Bloch vector for single-qubit visualization:**
```python
from qconduit.diagnostics import bloch_vector

# Visualize single-qubit states
state = create_single_qubit_state()
bloch = bloch_vector(state)  # (x, y, z) coordinates
# Use for plotting or analysis
```

### QAOA for Optimization Problems

**MaxCut optimization:**
```python
from qconduit.algorithms import QAOAAnsatz, ising_maxcut_hamiltonian, Edge, VQE

# Define your graph
edges = [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 0)]  # 4-cycle
hamiltonian = ising_maxcut_hamiltonian(num_nodes=4, edges=edges)

# Use QAOA to find maximum cut
qaoa = QAOAAnsatz(n_qubits=4, problem_hamiltonian=hamiltonian, p=3)
vqe = VQE(ansatz=qaoa, hamiltonian=hamiltonian)
# Optimize to find maximum cut value
```

**Weighted graphs:**
```python
# Use weighted edges for optimization problems
weighted_edges = [
    Edge(0, 1, weight=2.0),
    Edge(1, 2, weight=1.5),
    Edge(2, 0, weight=1.0),
]
hamiltonian = ising_maxcut_hamiltonian(num_nodes=3, edges=weighted_edges)
```

### Training Workflows

**Complete training pipeline:**
```python
from qconduit.training import VQETrainer, TrainingCallback, EarlyStoppingConfig

# Set up training with callbacks
class CheckpointCallback(TrainingCallback):
    def __call__(self, info):
        if info.step % 50 == 0:
            # Save checkpoint
            torch.save(params, f"checkpoint_step_{info.step}.pt")

trainer = VQETrainer(vqe, optimizer=optimizer)
history = trainer.train(
    params,
    max_steps=500,
    callbacks=[CheckpointCallback()],
    early_stopping=EarlyStoppingConfig(patience=20),
)

# Analyze training
print(f"Converged in {history.num_steps()} steps")
print(f"Best energy: {history.best_energy()}")
```

### Sampling for Measurement Simulation

**Simulate quantum measurements:**
```python
from qconduit.sampling import sample_bitstrings_state, bitstring_counts

# Simulate 1000 measurements
samples = sample_bitstrings_state(state, n_qubits=4, n_shots=1000)

# Analyze measurement statistics
counts = bitstring_counts(samples)
most_common = max(counts.items(), key=lambda x: x[1])
print(f"Most frequent outcome: {most_common[0]} ({most_common[1]} times)")

# Compare with theoretical probabilities
probs = qc.measure_probs(state, n_qubits=4)
# Use KL divergence to measure agreement
```

**Partial measurements:**
```python
# Sample only specific qubits
samples = sample_bitstrings_state(
    state, n_qubits=4, n_shots=1000, qubits=[0, 1]  # Only measure qubits 0,1
)
```

### Time Evolution for Quantum Dynamics

**Simulate quantum dynamics:**
```python
from qconduit.time_evolution import time_evolve_state

# Evolve state under a Hamiltonian
times = torch.linspace(0, 1.0, 100)
states = []
for t_val in times:
    evolved = time_evolve_state(state, hamiltonian, t=t_val.item(), n_steps=20, n_qubits=2)
    states.append(evolved)

# Analyze time-dependent properties
expectations = [qc.measure_expectation_z(s, qubit=0, n_qubits=2) for s in states]
```

**Trotter circuit for hardware:**
```python
# Build circuit representation for hardware execution
circuit = build_trotter_circuit(
    hamiltonian, t=1.0, n_steps=50, n_qubits=2, order=2  # Second-order Trotter
)
print(circuit.to_text_diagram())  # Visualize the circuit
```

### Quantum State Tomography

**Reconstruct density matrices from measurements:**
```python
from qconduit.measurement import (
    single_qubit_pauli_expectations_from_statevector,
    reconstruct_single_qubit_density_from_pauli,
    two_qubit_pauli_expectations_from_statevector,
    reconstruct_two_qubit_density_from_pauli,
)

# Single-qubit tomography
state = qc.zero_state(n_qubits=1)
state = qc.apply_gate(state, qc.H(), qubit=0, n_qubits=1)

# Measure Pauli expectations
ex_x, ex_y, ex_z = single_qubit_pauli_expectations_from_statevector(state)

# Reconstruct density matrix
rho_reconstructed = reconstruct_single_qubit_density_from_pauli(ex_x, ex_y, ex_z)

# Verify reconstruction fidelity
rho_actual = qc.dm_from_statevector(state)
fidelity = qc.fidelity(rho_actual, rho_reconstructed)
print(f"Reconstruction fidelity: {fidelity.item():.6f}")

# Two-qubit tomography
state_2q = qc.zero_state(n_qubits=2)
state_2q = qc.apply_gate(state_2q, qc.H(), qubit=0, n_qubits=2)
state_2q = qc.apply_two_qubit_gate(state_2q, qc.CNOT(), qubit1=0, qubit2=1, n_qubits=2)

# Get all two-qubit Pauli expectations
pauli_expectations = two_qubit_pauli_expectations_from_statevector(state_2q)
rho_2q = reconstruct_two_qubit_density_from_pauli(pauli_expectations)
```

**Estimate expectations from samples:**
```python
from qconduit.measurement import estimate_pauli_z_expectation_from_samples

# Sample bitstrings from state
samples = sample_bitstrings_from_statevector(state, n_shots=10000)

# Estimate Pauli-Z expectation from samples
z_expectation, std_error = estimate_pauli_z_expectation_from_samples(samples, qubit_index=0)
```

### High-Level Variational Algorithms

**Run VQE with minimal code:**
```python
from qconduit.variational import run_vqe

# Simple VQE execution
from qconduit.variational import HardwareEfficientAnsatz
import torch

ansatz = HardwareEfficientAnsatz(num_qubits=4, num_layers=3)
initial_params = torch.randn(ansatz.num_parameters)

result = run_vqe(
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    initial_params=initial_params,
    max_iterations=200,
)

print(f"Ground state energy: {result.optimal_value:.6f}")
print(f"Converged: {result.converged}")
```

**Run QAOA for optimization:**
```python
from qconduit.variational import run_qaoa

# High-level QAOA API
qaoa_result = run_qaoa(
    cost_hamiltonian=maxcut_hamiltonian,
    num_qubits=5,
    depth=2,
    max_iterations=150,
)

print(f"Optimal cost: {qaoa_result.optimal_value:.6f}")
```

**Custom ansätze with result objects:**
```python
from qconduit.variational import HardwareEfficientAnsatz, evaluate_expectation_value

ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=2)
params = torch.randn(ansatz.num_parameters)
energy = evaluate_expectation_value(ansatz, params, hamiltonian)
```

### Circuit Transpilation for Hardware

**Transpile to hardware-native gates:**
```python
from qconduit.transpile import transpile_to_rx_rz_cx_basis, transpile_to_clifford_t

# Original circuit with various gates
circuit = QuantumCircuit(n_qubits=3)
circuit.add_gate("H", [0])
circuit.add_gate("T", [1])
circuit.add_gate("S", [2])
circuit.add_gate("CNOT", [0, 1])

# Transpile to RX, RZ, CNOT (common hardware basis)
hardware_circuit = transpile_to_rx_rz_cx_basis(circuit)

# Transpile to Clifford+T (for fault-tolerant quantum computing)
clifford_t_circuit = transpile_to_clifford_t(circuit)

# Analyze gate counts
from qconduit.transpile import summarize_gate_counts
summary = summarize_gate_counts(clifford_t_circuit)
print(f"T-count: {summary.t_count}")  # Important for fault-tolerant computing
print(f"Clifford count: {summary.clifford_count}")
```

**Gate decomposition:**
```python
from qconduit.transpile import decompose_h_to_rz_rx_rz

# Decompose Hadamard gate
circuit = QuantumCircuit(n_qubits=1)
circuit.add_gate("H", [0])
decompose_h_to_rz_rx_rz(circuit, qubit=0)  # Replaces H with RZ-RX-RZ
```

### Advanced Noise Modeling

**Enhanced Kraus channels:**
```python
from qconduit.noise import (
    bit_flip_channel,
    phase_flip_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
    compose_kraus_channels,
)

# Create various noise channels
bit_flip = bit_flip_channel(p=0.01)
phase_flip = phase_flip_channel(p=0.02)

# Generalized amplitude damping (with temperature)
amp_damp = generalized_amplitude_damping_channel(gamma=0.1, n_th=0.1)

# Two-qubit correlated noise
two_qubit_depol = two_qubit_depolarizing_channel(p=0.01)

# Compose multiple noise channels (compose two at a time)
combined_noise = compose_kraus_channels(bit_flip, phase_flip)

# Apply to state
from qconduit.noise import apply_kraus_channel_to_statevector
noisy_state = apply_kraus_channel_to_statevector(
    state, combined_noise, qubit=0, n_qubits=2
)
```

### Exact vs Approximate Evolution

**Compare exact and Trotter evolution:**
```python
from qconduit.evolution import (
    exact_time_evolution_statevector,
    TrotterSchedule,
    evolve_state_trotter,
)

# Exact evolution (for small systems)
evolved_exact = exact_time_evolution_statevector(
    state, hamiltonian, time=0.5
)

# Trotter evolution (scales to larger systems)
schedule = TrotterSchedule(
    num_steps=10,
    total_time=0.5,
    order=1,  # First-order Trotter
)
evolved_trotter = evolve_state_trotter(
    state, hamiltonian, schedule
)

# Compare fidelity
fidelity = qc.fidelity(
    qc.dm_from_statevector(evolved_exact),
    qc.dm_from_statevector(evolved_trotter)
)
print(f"Fidelity: {fidelity.item():.6f}")

# Use higher-order Trotter for better accuracy
schedule_2nd = TrotterSchedule(
    num_steps=10,
    total_time=0.5,
    order=2,  # Second-order symmetric Trotter
)
```

### Parameter Sweeps for Algorithm Exploration

**Explore parameter landscapes:**
```python
from qconduit.experiments import sweep_vqe_1d, sweep_vqe_2d

# 1D sweep: explore single parameter
result = sweep_vqe_1d(
    vqe,
    points=torch.linspace(0, 2 * torch.pi, 100),
    base_params=params_template,
    index=0,
)

# Find optimal parameter value
optimal_idx = result.values.argmin()
optimal_param = result.points[optimal_idx]
print(f"Optimal parameter: {optimal_param}")

# 2D sweep: explore parameter interactions
result_2d = sweep_vqe_2d(
    vqe,
    x_points=torch.linspace(0, 2 * torch.pi, 50),
    y_points=torch.linspace(0, 2 * torch.pi, 50),
    base_params=params_template,
    x_index=0,
    y_index=1,
)
# Visualize with matplotlib: plt.contourf(result_2d.values)
```

### Exact Solvers for Benchmarking

**Validate VQE results:**
```python
from qconduit.exact import exact_ground_state
from qconduit.algorithms import VQE

# Get exact ground state energy
exact_energy, exact_state = exact_ground_state(hamiltonian, num_qubits=4)

# Compare with VQE result
vqe = VQE(ansatz=ansatz, hamiltonian=hamiltonian)
vqe_energy = vqe.energy(optimized_params)

error = abs(vqe_energy - exact_energy)
print(f"VQE error: {error.item():.6f}")
print(f"Relative error: {(error / abs(exact_energy)).item():.2%}")
```

**Analyze full spectrum:**
```python
from qconduit.exact import exact_eigensystem

# Get all eigenvalues and eigenvectors
eigenvalues, eigenvectors = exact_eigensystem(hamiltonian, num_qubits=3)

# Analyze energy gap
gap = eigenvalues[1] - eigenvalues[0]
print(f"Ground state energy: {eigenvalues[0].item():.6f}")
print(f"First excited state: {eigenvalues[1].item():.6f}")
print(f"Energy gap: {gap.item():.6f}")
```

### Pre-built Models for Research

**Study phase transitions:**
```python
from qconduit.models import transverse_field_ising_chain
from qconduit.exact import exact_ground_state

# Study critical point in TFIM
h_values = torch.linspace(0.1, 2.0, 20)
energies = []
for h in h_values:
    hamiltonian = transverse_field_ising_chain(
        num_sites=8, j_coupling=1.0, h_field=h.item(), periodic=True
    )
    energy, _ = exact_ground_state(hamiltonian, num_qubits=8)
    energies.append(energy.item())

# Plot energy vs field strength to identify phase transition
```

**Compare different models:**
```python
from qconduit.models import (
    transverse_field_ising_chain,
    heisenberg_xxz_chain,
    ising_zz_chain,
)

# Compare ground state energies
tfim = transverse_field_ising_chain(4, j_coupling=1.0, h_field=0.5)
heisenberg = heisenberg_xxz_chain(4, j_coupling=1.0, delta=0.5)
ising = ising_zz_chain(4, j_coupling=1.0)

# Use exact diagonalization or VQE to compare
```

### Adiabatic Quantum Computing

**Adiabatic optimization:**
```python
from qconduit.adiabatic import (
    AdiabaticConfig, linear_schedule, adiabatic_evolve_state,
    build_x_mixer_hamiltonian
)

# Set up adiabatic evolution for optimization
h_mixer = build_x_mixer_hamiltonian(num_qubits=4)
h_problem = your_problem_hamiltonian

config = AdiabaticConfig(
    total_time=2.0,
    num_steps=50,
    schedule=linear_schedule(50),
    trotter_steps_per_interval=10
)

# Prepare initial state (ground state of mixer = |+⟩^⊗n)
initial_state = qc.zero_state(n_qubits=4)
for i in range(4):
    initial_state = qc.apply_gate(initial_state, qc.H(), qubit=i, n_qubits=4)

# Evolve adiabatically
final_state = adiabatic_evolve_state(
    initial_state, h_mixer, h_problem, config
)

# Measure to get solution
probs = qc.measure_probs(final_state, n_qubits=4)
solution = torch.argmax(probs)
```

**Custom schedules:**
```python
# Use polynomial schedule for slower initial evolution
schedule = polynomial_schedule(num_steps=50, power=3.0)

# Or create custom schedule
def custom_schedule(num_steps):
    # Spend more time near s=1 (problem Hamiltonian)
    t = torch.linspace(0, 1, num_steps)
    return t ** 0.3  # Slow start, fast finish
```

### Quantum Chemistry Applications

**Map fermionic Hamiltonians:**
```python
from qconduit.fermion import FermionOperator, FermionTerm, jordan_wigner

# Create molecular Hamiltonian (simplified example)
# H = sum_{p,q} h_{pq} a^†_p a_q + sum_{p,q,r,s} g_{pqrs} a^†_p a^†_q a_r a_s
terms = []
# One-body terms
for p in range(n_orbitals):
    for q in range(n_orbitals):
        if h_matrix[p, q] != 0:
            terms.append(FermionTerm(
                coeff=h_matrix[p, q],
                operators=((p, "+"), (q, "-"))
            ))
# Two-body terms (simplified)
# ... add interaction terms ...

fermion_ham = FermionOperator(terms)

# Map to qubits
qubit_hamiltonian = jordan_wigner(fermion_ham, n_spin_orbitals=n_orbitals)

# Use with VQE or exact diagonalization
from qconduit.algorithms import VQE
vqe = VQE(ansatz=chemistry_ansatz, hamiltonian=qubit_hamiltonian)
```

**Compare mapping methods:**
```python
from qconduit.fermion import jordan_wigner, bravyi_kitaev

# Jordan-Wigner typically has more Pauli terms but simpler structure
jw_ham = jordan_wigner(fermion_op, n_spin_orbitals=4)
print(f"JW: {len(jw_ham.terms)} terms")

# Bravyi-Kitaev often has fewer terms but more complex structure
bk_ham = bravyi_kitaev(fermion_op, n_spin_orbitals=4)
print(f"BK: {len(bk_ham.terms)} terms")

# Choose based on your hardware constraints
```

### Noisy Circuit Simulation

**Model realistic hardware:**
```python
from qconduit.noise import NoiseConfig, simulate_noisy_circuit_dm
from qconduit.noise import DepolarizingChannel, AmplitudeDampingChannel

# Model realistic noise from quantum hardware
noise_config = NoiseConfig(
    per_qubit_channels={
        0: DepolarizingChannel(p=0.005),  # 0.5% gate error
        1: DepolarizingChannel(p=0.008),  # 0.8% gate error
        2: AmplitudeDampingChannel(gamma=0.01),  # T1 decay
    }
)

# Simulate circuit with noise
rho = simulate_noisy_circuit_dm(circuit, noise=noise_config)

# Compare with ideal simulation
ideal_state = circuit.simulate_state()
ideal_rho = qc.dm_from_statevector(ideal_state)

# Compute fidelity
from qconduit.diagnostics import fidelity
f = fidelity(ideal_rho, rho)
print(f"Fidelity: {f.item():.6f}")
```

**Error mitigation studies:**
```python
# Study how noise affects algorithm performance
noise_levels = [0.001, 0.005, 0.01, 0.02]
fidelities = []

for prob in noise_levels:
    noise = NoiseConfig(per_qubit_channels={
        i: DepolarizingChannel(p=prob) for i in range(n_qubits)
    })
    rho = simulate_noisy_circuit_dm(circuit, noise=noise)
    f = fidelity(ideal_rho, rho)
    fidelities.append(f.item())

# Analyze noise threshold
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
   git clone https://github.com/seansimms/Quantum_Conduit.git
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

## Citation

If you use Quantum Conduit in your research, please cite it as:

**APA:**
```
Simms, S. (2025). Quantum Conduit: A PyTorch-native quantum statevector plumbing library for quantum machine learning (Version 0.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17599984
```

**BibTeX:**
```bibtex
@software{simms2025quantum,
  author       = {Simms, Sean},
  title        = {Quantum Conduit: A PyTorch-native quantum statevector 
                  plumbing library for quantum machine learning},
  version      = {0.0.1},
  month        = {11},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17599984},
  url          = {https://doi.org/10.5281/zenodo.17599984}
}
```

**Citation File Format:**
The repository includes a `CITATION.cff` file that can be used by citation management tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the quantum machine learning community**

[Report Bug](https://github.com/seansimms/Quantum_Conduit/issues) • [Request Feature](https://github.com/seansimms/Quantum_Conduit/issues) • [Documentation](https://github.com/seansimms/Quantum_Conduit)

</div>
