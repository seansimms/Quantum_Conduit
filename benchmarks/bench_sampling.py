"""Benchmark measurement sampling operations."""

import time
from typing import Dict

import torch

import qconduit as qc
from qconduit.backend.statevector import zero_state
from qconduit.sampling import sample_bitstrings_state


def benchmark_sampling(
    n_qubits: int,
    n_shots: int = 10000,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> Dict[str, float]:
    """Benchmark bitstring sampling from statevector.
    
    Args:
        n_qubits: Number of qubits.
        n_shots: Number of shots (samples).
        device: Device ('cpu' or 'cuda').
        dtype: Data type.
    
    Returns:
        Dictionary with timing results.
    """
    # Map device string to qconduit device
    device_map = {"cpu": "sv_cpu", "cuda": "sv_cuda"}
    qc_device = qc.device(device_map.get(device, "sv_cpu"))
    
    # Create random state (apply Hadamard to all qubits)
    state = zero_state(n_qubits=n_qubits, device=qc_device, dtype=dtype)
    for i in range(n_qubits):
        H = qc.H(dtype=dtype, device=torch.device(device))
        state = qc.apply_gate(state, H, qubit=i, n_qubits=n_qubits)
    
    # Warmup
    sample_bitstrings_state(state, n_qubits=n_qubits, n_shots=100)
    
    # Benchmark
    start = time.perf_counter()
    samples = sample_bitstrings_state(state, n_qubits=n_qubits, n_shots=n_shots)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_shot = total_time / n_shots
    
    return {
        "n_qubits": n_qubits,
        "n_shots": n_shots,
        "total_time_sec": total_time,
        "time_per_shot_sec": time_per_shot,
        "shots_per_sec": n_shots / total_time,
    }


if __name__ == "__main__":
    print("Benchmarking sampling...")
    
    results = benchmark_sampling(n_qubits=10, n_shots=100000)
    print(f"Sampling (10 qubits, 100k shots):")
    print(f"  Time per shot: {results['time_per_shot_sec']*1e6:.2f} μs")
    print(f"  Shots per second: {results['shots_per_sec']:.0f}")

