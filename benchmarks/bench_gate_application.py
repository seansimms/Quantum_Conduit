"""Benchmark gate application operations."""

import time
from typing import Dict

import torch

import qconduit as qc
from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.gates import standard as stdgates


def benchmark_gate_application(
    n_qubits: int,
    n_gates: int = 1000,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> Dict[str, float]:
    """Benchmark gate application on statevector.
    
    Args:
        n_qubits: Number of qubits.
        n_gates: Number of gates to apply.
        device: Device ('cpu' or 'cuda').
        dtype: Data type.
    
    Returns:
        Dictionary with timing results.
    """
    # Map device string to qconduit device
    device_map = {"cpu": "sv_cpu", "cuda": "sv_cuda"}
    qc_device = qc.device(device_map.get(device, "sv_cpu"))
    torch_device = torch.device(device)
    
    # Create initial state
    state = zero_state(n_qubits=n_qubits, device=qc_device, dtype=dtype)
    
    # Create gates
    gates = [
        stdgates.H(dtype=dtype, device=torch_device),
        stdgates.X(dtype=dtype, device=torch_device),
        stdgates.Y(dtype=dtype, device=torch_device),
        stdgates.Z(dtype=dtype, device=torch_device),
    ]
    
    # Warmup
    for _ in range(10):
        apply_gate(state, gates[0], qubit=0, n_qubits=n_qubits)
    
    # Benchmark
    start = time.perf_counter()
    for i in range(n_gates):
        gate = gates[i % len(gates)]
        qubit = i % n_qubits
        state = apply_gate(state, gate, qubit=qubit, n_qubits=n_qubits)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_gate = total_time / n_gates
    
    return {
        "n_qubits": n_qubits,
        "n_gates": n_gates,
        "total_time_sec": total_time,
        "time_per_gate_sec": time_per_gate,
        "gates_per_sec": n_gates / total_time,
    }


def benchmark_batched_gate_application(
    n_qubits: int,
    batch_size: int = 100,
    n_gates: int = 100,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> Dict[str, float]:
    """Benchmark batched gate application.
    
    Args:
        n_qubits: Number of qubits.
        batch_size: Batch size.
        n_gates: Number of gates to apply.
        device: Device ('cpu' or 'cuda').
        dtype: Data type.
    
    Returns:
        Dictionary with timing results.
    """
    # Map device string to qconduit device
    device_map = {"cpu": "sv_cpu", "cuda": "sv_cuda"}
    qc_device = qc.device(device_map.get(device, "sv_cpu"))
    torch_device = torch.device(device)
    
    # Create batched initial state
    state = zero_state(
        n_qubits=n_qubits,
        batch_shape=(batch_size,),
        device=qc_device,
        dtype=dtype,
    )
    
    # Create gate
    gate = stdgates.H(dtype=dtype, device=torch_device)
    
    # Warmup
    for _ in range(5):
        apply_gate(state, gate, qubit=0, n_qubits=n_qubits)
    
    # Benchmark
    start = time.perf_counter()
    for i in range(n_gates):
        qubit = i % n_qubits
        state = apply_gate(state, gate, qubit=qubit, n_qubits=n_qubits)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_gate = total_time / n_gates
    
    return {
        "n_qubits": n_qubits,
        "batch_size": batch_size,
        "n_gates": n_gates,
        "total_time_sec": total_time,
        "time_per_gate_sec": time_per_gate,
        "gates_per_sec": n_gates / total_time,
    }


if __name__ == "__main__":
    print("Benchmarking gate application...")
    
    # Single statevector
    results = benchmark_gate_application(n_qubits=5, n_gates=1000)
    print(f"Single statevector (5 qubits, 1000 gates):")
    print(f"  Time per gate: {results['time_per_gate_sec']*1e6:.2f} μs")
    print(f"  Gates per second: {results['gates_per_sec']:.0f}")
    
    # Batched
    results_batched = benchmark_batched_gate_application(
        n_qubits=5, batch_size=100, n_gates=100
    )
    print(f"\nBatched (5 qubits, batch_size=100, 100 gates):")
    print(f"  Time per gate: {results_batched['time_per_gate_sec']*1e3:.2f} ms")
    print(f"  Gates per second: {results_batched['gates_per_sec']:.0f}")

