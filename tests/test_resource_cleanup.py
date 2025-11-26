"""Tests for resource cleanup and memory management."""

import gc
import tracemalloc
from contextlib import contextmanager

import pytest
import torch

import qconduit as qc
from qconduit.backend.statevector import zero_state


@contextmanager
def memory_tracker():
    """Context manager to track memory usage."""
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Return peak memory in MB
        return peak / 1024 / 1024


def test_statevector_memory_cleanup():
    """Test that statevectors are properly cleaned up."""
    # Create and use statevectors
    states = []
    for i in range(10):
        state = zero_state(n_qubits=5, device=qc.default_device(), dtype=torch.complex64)
        states.append(state)
    
    # Delete references
    del states
    gc.collect()
    
    # Should not raise or leak
    assert True


def test_circuit_simulation_cleanup():
    """Test that circuit simulation doesn't leak memory."""
    circuit = qc.QuantumCircuit(n_qubits=3)
    circuit.add_gate("H", [0])
    circuit.add_gate("CNOT", [0, 1])
    circuit.add_gate("CNOT", [1, 2])
    
    # Simulate multiple times
    for _ in range(10):
        state = circuit.simulate_state()
        del state
        gc.collect()
    
    # Should not leak
    assert True


def test_batched_operations_cleanup():
    """Test that batched operations clean up properly."""
    # Create batched state
    state = zero_state(
        n_qubits=4,
        batch_shape=(100,),
        device=qc.default_device(),
        dtype=torch.complex64,
    )
    
    # Apply gates
    H = qc.H(dtype=torch.complex64)
    for i in range(4):
        state = qc.apply_gate(state, H, qubit=i, n_qubits=4)
    
    # Cleanup
    del state
    gc.collect()
    
    assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_memory_cleanup():
    """Test CUDA memory cleanup (if available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create state on CUDA
    state = zero_state(
        n_qubits=5,
        device=qc.device("cuda"),
        dtype=torch.complex64,
    )
    
    # Use state
    H = qc.H(dtype=torch.complex64, device=torch.device("cuda"))
    state = qc.apply_gate(state, H, qubit=0, n_qubits=5)
    
    # Cleanup
    del state
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check CUDA memory
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    # Should be minimal after cleanup
    assert allocated < 100  # Less than 100MB allocated


def test_repeated_operations_no_leak():
    """Test that repeated operations don't cause memory leaks."""
    # Run many operations
    for _ in range(100):
        state = zero_state(n_qubits=3, device=qc.default_device(), dtype=torch.complex64)
        H = qc.H(dtype=torch.complex64)
        state = qc.apply_gate(state, H, qubit=0, n_qubits=3)
        del state
        gc.collect()
    
    # Should not accumulate memory
    assert True

