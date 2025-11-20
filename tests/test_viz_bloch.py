"""Tests for Bloch sphere coordinate computation and plotting."""

import pytest
import torch
import math

from qconduit.viz.bloch import (
    bloch_coords_from_statevector,
    bloch_coords_from_density,
    partial_trace_statevector,
)


def test_bloch_coords_from_density_zero():
    """Test Bloch coordinates for |0⟩ state."""
    # |0⟩ density matrix: [[1, 0], [0, 0]]
    rho = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
    x, y, z = bloch_coords_from_density(rho)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z - 1.0) < 1e-6


def test_bloch_coords_from_density_one():
    """Test Bloch coordinates for |1⟩ state."""
    # |1⟩ density matrix: [[0, 0], [0, 1]]
    rho = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
    x, y, z = bloch_coords_from_density(rho)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z + 1.0) < 1e-6


def test_bloch_coords_from_density_plus():
    """Test Bloch coordinates for |+⟩ = (|0⟩ + |1⟩)/√2."""
    # |+⟩ density matrix: [[0.5, 0.5], [0.5, 0.5]]
    rho = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.complex64)
    x, y, z = bloch_coords_from_density(rho)
    assert abs(x - 1.0) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6


def test_bloch_coords_from_density_iplus():
    """Test Bloch coordinates for |i+⟩ = (|0⟩ + i|1⟩)/√2."""
    # |i+⟩ density matrix: [[0.5, -0.5j], [0.5j, 0.5]]
    # For |i+⟩ = (|0⟩ + i|1⟩)/√2, ⟨i+| = (⟨0| - i⟨1|)/√2
    # So ρ[0,1] = -i/2, giving y = 2 * Im(-i/2) = 2 * (-1/2) = -1
    rho = torch.tensor([[0.5, -0.5j], [0.5j, 0.5]], dtype=torch.complex64)
    x, y, z = bloch_coords_from_density(rho)
    assert abs(x) < 1e-6
    assert abs(y + 1.0) < 1e-6  # y = -1 for |i+⟩
    assert abs(z) < 1e-6


def test_bloch_coords_from_statevector_zero():
    """Test Bloch coordinates from |0⟩ statevector."""
    psi = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    x, y, z = bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=1)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z - 1.0) < 1e-6


def test_bloch_coords_from_statevector_one():
    """Test Bloch coordinates from |1⟩ statevector."""
    psi = torch.tensor([0.0, 1.0], dtype=torch.complex64)
    x, y, z = bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=1)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z + 1.0) < 1e-6


def test_bloch_coords_from_statevector_plus():
    """Test Bloch coordinates from |+⟩ statevector."""
    sqrt2 = 1.0 / math.sqrt(2.0)
    psi = torch.tensor([sqrt2, sqrt2], dtype=torch.complex64)
    x, y, z = bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=1)
    assert abs(x - 1.0) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6


def test_bloch_coords_from_statevector_iplus():
    """Test Bloch coordinates from |i+⟩ statevector."""
    sqrt2 = 1.0 / math.sqrt(2.0)
    psi = torch.tensor([sqrt2, 1j * sqrt2], dtype=torch.complex64)
    x, y, z = bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=1)
    assert abs(x) < 1e-6
    assert abs(y + 1.0) < 1e-6  # y = -1 for |i+⟩
    assert abs(z) < 1e-6


def test_bloch_coords_bell_state():
    """Test Bloch coordinates for Bell state reduced density (maximally mixed)."""
    # Bell state: (|00⟩ + |11⟩)/√2
    sqrt2 = 1.0 / math.sqrt(2.0)
    psi = torch.tensor([sqrt2, 0.0, 0.0, sqrt2], dtype=torch.complex64)
    x, y, z = bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=2)
    # Reduced state should be maximally mixed: (|0><0| + |1><1|)/2
    # Bloch coordinates: (0, 0, 0)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6


def test_partial_trace_statevector():
    """Test partial trace computation."""
    # Simple 2-qubit state: |00⟩
    psi = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    rho = partial_trace_statevector(psi, qubit_index=0, n_qubits=2)
    assert rho.shape == (2, 2)
    # Should be |0><0| for qubit 0
    assert abs(rho[0, 0] - 1.0) < 1e-6
    assert abs(rho[0, 1]) < 1e-6
    assert abs(rho[1, 0]) < 1e-6
    assert abs(rho[1, 1]) < 1e-6


def test_plot_bloch_vector():
    """Test plot_bloch_vector function (requires matplotlib)."""
    pytest.importorskip("matplotlib")
    from qconduit.viz.bloch import plot_bloch_vector

    coords = (0.0, 0.0, 1.0)
    ax = plot_bloch_vector(coords)
    assert ax is not None


def test_plot_bloch_projections():
    """Test plot_bloch_projections function (requires matplotlib)."""
    pytest.importorskip("matplotlib")
    from qconduit.viz.bloch import plot_bloch_projections

    coords = (1.0, 0.0, 0.0)
    ax = plot_bloch_projections(coords)
    assert ax is not None


def test_plot_bloch_vector_no_matplotlib():
    """Test that plotting raises error when matplotlib is missing."""
    # This test assumes matplotlib might not be installed
    # We'll skip if it is installed, or test the error if not
    try:
        import matplotlib
        pytest.skip("matplotlib is installed")
    except ImportError:
        from qconduit.viz.bloch import plot_bloch_vector
        with pytest.raises(RuntimeError, match="matplotlib"):
            plot_bloch_vector((0, 0, 1))


def test_bloch_coords_invalid_density():
    """Test error handling for invalid density matrix shape."""
    rho = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="shape"):
        bloch_coords_from_density(rho)


def test_bloch_coords_invalid_statevector():
    """Test error handling for invalid statevector."""
    psi = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)  # Not power of 2
    with pytest.raises(ValueError):
        bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=None)


def test_bloch_coords_invalid_qubit_index():
    """Test error handling for invalid qubit index."""
    psi = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    with pytest.raises(ValueError, match="qubit_index"):
        bloch_coords_from_statevector(psi, qubit_index=5, n_qubits=1)


def test_bloch_coords_statevector_wrong_shape():
    """Test error handling for wrong statevector shape."""
    psi = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)  # 2D instead of 1D
    with pytest.raises(ValueError, match="shape"):
        bloch_coords_from_statevector(psi, qubit_index=0, n_qubits=1)

