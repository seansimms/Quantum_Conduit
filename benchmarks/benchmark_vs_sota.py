#!/usr/bin/env python3
"""
Benchmark Quantum Conduit against State-of-the-Art Quantum Simulation Libraries.

This benchmark compares performance across:
1. State initialization
2. Single-qubit gate application
3. Two-qubit gate application (CNOT)
4. Circuit execution (random circuit)
5. Expectation value computation

Libraries compared:
- Quantum Conduit (this library)
- Qiskit Aer (IBM)
- Cirq (Google)
- PennyLane (Xanadu)

Run: python benchmarks/benchmark_vs_sota.py
"""

import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

warnings.filterwarnings("ignore")

# Results storage
@dataclass
class BenchmarkResult:
    library: str
    operation: str
    n_qubits: int
    time_ms: float
    ops_per_sec: float
    notes: str = ""


def time_operation(func: Callable, n_iterations: int = 100, warmup: int = 10) -> float:
    """Time an operation with warmup."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        func()
    end = time.perf_counter()
    
    return (end - start) / n_iterations * 1000  # ms


# =============================================================================
# Quantum Conduit Benchmarks
# =============================================================================

def benchmark_qconduit(n_qubits_list: List[int]) -> List[BenchmarkResult]:
    """Benchmark Quantum Conduit."""
    results = []
    
    try:
        import torch
        from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate, zero_state
        from qconduit.gates import standard as gates
        from qconduit.operators.expectation import expectation_pauli_term
        from qconduit.operators.pauli import PauliTerm
        
        for n_qubits in n_qubits_list:
            print(f"  Quantum Conduit: {n_qubits} qubits...")
            
            # State initialization
            def init_state():
                return zero_state(n_qubits)
            
            time_ms = time_operation(init_state, n_iterations=1000)
            results.append(BenchmarkResult(
                library="Quantum Conduit",
                operation="State Init",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Single-qubit gate (H)
            state = zero_state(n_qubits)
            h_gate = gates.H(dtype=state.dtype, device=state.device)
            
            def apply_h():
                nonlocal state
                state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
            
            time_ms = time_operation(apply_h, n_iterations=1000)
            results.append(BenchmarkResult(
                library="Quantum Conduit",
                operation="H Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Two-qubit gate (CNOT)
            state = zero_state(n_qubits)
            cnot = gates.CNOT(dtype=state.dtype, device=state.device)
            
            def apply_cnot():
                nonlocal state
                state = apply_two_qubit_gate(state, cnot, qubit1=0, qubit2=1, n_qubits=n_qubits)
            
            time_ms = time_operation(apply_cnot, n_iterations=500)
            results.append(BenchmarkResult(
                library="Quantum Conduit",
                operation="CNOT Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Random circuit (H on all + CNOTs)
            def random_circuit():
                s = zero_state(n_qubits)
                for q in range(n_qubits):
                    s = apply_gate(s, h_gate, qubit=q, n_qubits=n_qubits)
                for q in range(n_qubits - 1):
                    s = apply_two_qubit_gate(s, cnot, qubit1=q, qubit2=q+1, n_qubits=n_qubits)
                return s
            
            time_ms = time_operation(random_circuit, n_iterations=100)
            results.append(BenchmarkResult(
                library="Quantum Conduit",
                operation="Circuit (H+CNOT)",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Expectation value
            state = random_circuit()
            term = PauliTerm(coeff=1.0, paulis=tuple(["Z"] * n_qubits))
            
            def compute_expectation():
                return expectation_pauli_term(state, term)
            
            time_ms = time_operation(compute_expectation, n_iterations=500)
            results.append(BenchmarkResult(
                library="Quantum Conduit",
                operation="Expectation",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
    
    except ImportError as e:
        print(f"  Quantum Conduit: Not available ({e})")
    
    return results


# =============================================================================
# Qiskit Benchmarks
# =============================================================================

def benchmark_qiskit(n_qubits_list: List[int]) -> List[BenchmarkResult]:
    """Benchmark Qiskit Aer."""
    results = []
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, Operator
        from qiskit.circuit.library import HGate, CXGate
        import numpy as np
        
        for n_qubits in n_qubits_list:
            print(f"  Qiskit: {n_qubits} qubits...")
            
            # State initialization
            def init_state():
                return Statevector.from_int(0, 2**n_qubits)
            
            time_ms = time_operation(init_state, n_iterations=1000)
            results.append(BenchmarkResult(
                library="Qiskit",
                operation="State Init",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Single-qubit gate (H)
            state = init_state()
            h_op = Operator(HGate())
            
            def apply_h():
                nonlocal state
                state = state.evolve(h_op, [0])
            
            time_ms = time_operation(apply_h, n_iterations=500)
            results.append(BenchmarkResult(
                library="Qiskit",
                operation="H Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Two-qubit gate (CNOT)
            state = init_state()
            cx_op = Operator(CXGate())
            
            def apply_cnot():
                nonlocal state
                state = state.evolve(cx_op, [0, 1])
            
            time_ms = time_operation(apply_cnot, n_iterations=500)
            results.append(BenchmarkResult(
                library="Qiskit",
                operation="CNOT Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Random circuit
            def random_circuit():
                qc = QuantumCircuit(n_qubits)
                for q in range(n_qubits):
                    qc.h(q)
                for q in range(n_qubits - 1):
                    qc.cx(q, q+1)
                return Statevector(qc)
            
            time_ms = time_operation(random_circuit, n_iterations=100)
            results.append(BenchmarkResult(
                library="Qiskit",
                operation="Circuit (H+CNOT)",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Expectation value
            from qiskit.quantum_info import SparsePauliOp
            state = random_circuit()
            z_string = "Z" * n_qubits
            op = SparsePauliOp(z_string)
            
            def compute_expectation():
                return state.expectation_value(op)
            
            time_ms = time_operation(compute_expectation, n_iterations=500)
            results.append(BenchmarkResult(
                library="Qiskit",
                operation="Expectation",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
    
    except ImportError as e:
        print(f"  Qiskit: Not available ({e})")
    
    return results


# =============================================================================
# Cirq Benchmarks
# =============================================================================

def benchmark_cirq(n_qubits_list: List[int]) -> List[BenchmarkResult]:
    """Benchmark Google Cirq."""
    results = []
    
    try:
        import cirq
        import numpy as np
        
        for n_qubits in n_qubits_list:
            print(f"  Cirq: {n_qubits} qubits...")
            
            qubits = cirq.LineQubit.range(n_qubits)
            
            # State initialization
            def init_state():
                return cirq.StateVectorSimulationState(qubits=qubits)
            
            time_ms = time_operation(init_state, n_iterations=1000)
            results.append(BenchmarkResult(
                library="Cirq",
                operation="State Init",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Single-qubit gate (H) - via circuit simulation
            simulator = cirq.Simulator()
            h_circuit = cirq.Circuit(cirq.H(qubits[0]))
            
            def apply_h():
                return simulator.simulate(h_circuit)
            
            time_ms = time_operation(apply_h, n_iterations=500)
            results.append(BenchmarkResult(
                library="Cirq",
                operation="H Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Two-qubit gate (CNOT)
            cnot_circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[1]))
            
            def apply_cnot():
                return simulator.simulate(cnot_circuit)
            
            time_ms = time_operation(apply_cnot, n_iterations=500)
            results.append(BenchmarkResult(
                library="Cirq",
                operation="CNOT Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Random circuit
            random_circuit = cirq.Circuit()
            for q in qubits:
                random_circuit.append(cirq.H(q))
            for i in range(n_qubits - 1):
                random_circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            
            def run_circuit():
                return simulator.simulate(random_circuit)
            
            time_ms = time_operation(run_circuit, n_iterations=100)
            results.append(BenchmarkResult(
                library="Cirq",
                operation="Circuit (H+CNOT)",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Expectation value
            z_obs = cirq.Z(qubits[0])
            for q in qubits[1:]:
                z_obs = z_obs * cirq.Z(q)
            
            result = run_circuit()
            
            def compute_expectation():
                # Cirq computes expectations differently
                state_vector = result.final_state_vector
                return z_obs.expectation_from_state_vector(
                    state_vector, {q: i for i, q in enumerate(qubits)}
                )
            
            time_ms = time_operation(compute_expectation, n_iterations=500)
            results.append(BenchmarkResult(
                library="Cirq",
                operation="Expectation",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
    
    except ImportError as e:
        print(f"  Cirq: Not available ({e})")
    
    return results


# =============================================================================
# PennyLane Benchmarks
# =============================================================================

def benchmark_pennylane(n_qubits_list: List[int]) -> List[BenchmarkResult]:
    """Benchmark PennyLane."""
    results = []
    
    try:
        import pennylane as qml
        import numpy as np
        
        for n_qubits in n_qubits_list:
            print(f"  PennyLane: {n_qubits} qubits...")
            
            dev = qml.device("default.qubit", wires=n_qubits)
            
            # State initialization (via circuit)
            @qml.qnode(dev)
            def init_circuit():
                return qml.state()
            
            time_ms = time_operation(init_circuit, n_iterations=500)
            results.append(BenchmarkResult(
                library="PennyLane",
                operation="State Init",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Single-qubit gate (H)
            @qml.qnode(dev)
            def h_circuit():
                qml.Hadamard(wires=0)
                return qml.state()
            
            time_ms = time_operation(h_circuit, n_iterations=500)
            results.append(BenchmarkResult(
                library="PennyLane",
                operation="H Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Two-qubit gate (CNOT)
            @qml.qnode(dev)
            def cnot_circuit():
                qml.CNOT(wires=[0, 1])
                return qml.state()
            
            time_ms = time_operation(cnot_circuit, n_iterations=500)
            results.append(BenchmarkResult(
                library="PennyLane",
                operation="CNOT Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Random circuit
            @qml.qnode(dev)
            def random_circuit():
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                return qml.state()
            
            time_ms = time_operation(random_circuit, n_iterations=100)
            results.append(BenchmarkResult(
                library="PennyLane",
                operation="Circuit (H+CNOT)",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # Expectation value
            @qml.qnode(dev)
            def expectation_circuit():
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) if n_qubits > 1 else qml.PauliZ(0))
            
            time_ms = time_operation(expectation_circuit, n_iterations=500)
            results.append(BenchmarkResult(
                library="PennyLane",
                operation="Expectation",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
    
    except ImportError as e:
        print(f"  PennyLane: Not available ({e})")
    
    return results


# =============================================================================
# Pure NumPy Reference (baseline)
# =============================================================================

def benchmark_numpy(n_qubits_list: List[int]) -> List[BenchmarkResult]:
    """Benchmark pure NumPy implementation (baseline)."""
    results = []
    
    try:
        import numpy as np
        
        for n_qubits in n_qubits_list:
            print(f"  NumPy: {n_qubits} qubits...")
            
            dim = 2 ** n_qubits
            
            # State initialization
            def init_state():
                state = np.zeros(dim, dtype=np.complex128)
                state[0] = 1.0
                return state
            
            time_ms = time_operation(init_state, n_iterations=1000)
            results.append(BenchmarkResult(
                library="NumPy",
                operation="State Init",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
            ))
            
            # H gate matrix
            H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
            
            # Single-qubit gate (H) - via full matrix
            def apply_h_matrix():
                # Build full H ⊗ I ⊗ ... ⊗ I
                full = H
                for _ in range(n_qubits - 1):
                    full = np.kron(full, np.eye(2))
                state = init_state()
                return full @ state
            
            time_ms = time_operation(apply_h_matrix, n_iterations=100)
            results.append(BenchmarkResult(
                library="NumPy",
                operation="H Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
                notes="Full matrix method",
            ))
            
            # CNOT gate
            CNOT = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ], dtype=np.complex128)
            
            def apply_cnot_matrix():
                if n_qubits < 2:
                    return init_state()
                # Build CNOT ⊗ I ⊗ ... ⊗ I
                full = CNOT
                for _ in range(n_qubits - 2):
                    full = np.kron(full, np.eye(2))
                state = init_state()
                return full @ state
            
            time_ms = time_operation(apply_cnot_matrix, n_iterations=50)
            results.append(BenchmarkResult(
                library="NumPy",
                operation="CNOT Gate",
                n_qubits=n_qubits,
                time_ms=time_ms,
                ops_per_sec=1000 / time_ms,
                notes="Full matrix method",
            ))
    
    except ImportError as e:
        print(f"  NumPy: Not available ({e})")
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    # Group by operation
    operations = ["State Init", "H Gate", "CNOT Gate", "Circuit (H+CNOT)", "Expectation"]
    
    for op in operations:
        op_results = [r for r in results if r.operation == op]
        if not op_results:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"OPERATION: {op}")
        print(f"{'=' * 80}")
        
        # Group by n_qubits
        n_qubits_set = sorted(set(r.n_qubits for r in op_results))
        
        for n_qubits in n_qubits_set:
            print(f"\n{n_qubits} Qubits:")
            print(f"{'Library':<20} {'Time (ms)':<15} {'Ops/sec':<15} {'Relative':<10}")
            print("-" * 60)
            
            qubit_results = [r for r in op_results if r.n_qubits == n_qubits]
            qubit_results.sort(key=lambda r: r.time_ms)
            
            fastest_time = qubit_results[0].time_ms if qubit_results else 1.0
            
            for r in qubit_results:
                relative = r.time_ms / fastest_time
                marker = "🏆" if relative == 1.0 else ""
                print(f"{r.library:<20} {r.time_ms:<15.4f} {r.ops_per_sec:<15.0f} {relative:<10.2f}x {marker}")


def print_summary(results: List[BenchmarkResult]):
    """Print overall summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Wins by Library")
    print("=" * 80)
    
    # Count wins per library
    operations = ["State Init", "H Gate", "CNOT Gate", "Circuit (H+CNOT)", "Expectation"]
    wins = {}
    
    for op in operations:
        op_results = [r for r in results if r.operation == op]
        n_qubits_set = sorted(set(r.n_qubits for r in op_results))
        
        for n_qubits in n_qubits_set:
            qubit_results = [r for r in op_results if r.n_qubits == n_qubits]
            if qubit_results:
                fastest = min(qubit_results, key=lambda r: r.time_ms)
                wins[fastest.library] = wins.get(fastest.library, 0) + 1
    
    total = sum(wins.values())
    print(f"\n{'Library':<20} {'Wins':<10} {'Percentage':<15}")
    print("-" * 45)
    for lib, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"{lib:<20} {count:<10} {pct:<15.1f}%")


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("QUANTUM SIMULATION BENCHMARK: Quantum Conduit vs State-of-the-Art")
    print("=" * 80)
    print()
    
    # Test configurations
    n_qubits_list = [4, 6, 8, 10]
    
    all_results = []
    
    # Run benchmarks
    print("Running benchmarks...")
    print()
    
    # Quantum Conduit (our library)
    print("Benchmarking Quantum Conduit...")
    all_results.extend(benchmark_qconduit(n_qubits_list))
    
    # Qiskit
    print("\nBenchmarking Qiskit...")
    all_results.extend(benchmark_qiskit(n_qubits_list))
    
    # Cirq
    print("\nBenchmarking Cirq...")
    all_results.extend(benchmark_cirq(n_qubits_list))
    
    # PennyLane
    print("\nBenchmarking PennyLane...")
    all_results.extend(benchmark_pennylane(n_qubits_list))
    
    # NumPy reference
    print("\nBenchmarking NumPy (baseline)...")
    all_results.extend(benchmark_numpy(n_qubits_list[:2]))  # Only small for NumPy
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print_results_table(all_results)
    print_summary(all_results)
    
    print("\n" + "=" * 80)
    print("NOTES:")
    print("- Lower time is better")
    print("- 🏆 indicates fastest for that configuration")
    print("- All tests use statevector simulation (no sampling)")
    print("- Results may vary based on hardware and library versions")
    print("=" * 80)


if __name__ == "__main__":
    main()

