"""Benchmark expectation value computation."""

import time
from typing import Dict

import torch

import qconduit as qc
from qconduit.backend.statevector import zero_state
from qconduit.operators import PauliSum, PauliTerm
from qconduit.operators.expectation import expectation_pauli_sum


def benchmark_expectation_computation(
    n_qubits: int,
    n_terms: int = 100,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> Dict[str, float]:
    """Benchmark expectation value computation.
    
    Args:
        n_qubits: Number of qubits.
        n_terms: Number of Pauli terms in Hamiltonian.
        device: Device ('cpu' or 'cuda').
        dtype: Data type.
    
    Returns:
        Dictionary with timing results.
    """
    # Map device string to qconduit device
    device_map = {"cpu": "sv_cpu", "cuda": "sv_cuda"}
    qc_device = qc.device(device_map.get(device, "sv_cpu"))
    
    # Create random state
    state = zero_state(n_qubits=n_qubits, device=qc_device, dtype=dtype)
    # Apply Hadamard to all qubits for non-trivial state
    for i in range(n_qubits):
        H = qc.H(dtype=dtype, device=torch.device(device))
        state = qc.apply_gate(state, H, qubit=i, n_qubits=n_qubits)
    
    # Create random PauliSum
    terms = []
    for _ in range(n_terms):
        paulis = tuple(
            torch.randint(0, 4, (n_qubits,)).tolist()
        )  # 0=I, 1=X, 2=Y, 3=Z
        pauli_str = ["I", "X", "Y", "Z"][paulis[0]]
        for p in paulis[1:]:
            pauli_str += ["I", "X", "Y", "Z"][p]
        coeff = torch.randn(1).item()
        terms.append(PauliTerm(coeff=coeff, paulis=pauli_str))
    
    hamiltonian = PauliSum.from_terms(terms)
    
    # Warmup
    for _ in range(5):
        expectation_pauli_sum(state, hamiltonian, n_qubits=n_qubits)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        expectation_pauli_sum(state, hamiltonian, n_qubits=n_qubits)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_eval = total_time / 100
    
    return {
        "n_qubits": n_qubits,
        "n_terms": n_terms,
        "total_time_sec": total_time,
        "time_per_eval_sec": time_per_eval,
        "evals_per_sec": 100 / total_time,
    }


if __name__ == "__main__":
    print("Benchmarking expectation computation...")
    
    results = benchmark_expectation_computation(n_qubits=5, n_terms=50)
    print(f"Expectation (5 qubits, 50 terms):")
    print(f"  Time per evaluation: {results['time_per_eval_sec']*1e3:.2f} ms")
    print(f"  Evaluations per second: {results['evals_per_sec']:.0f}")

