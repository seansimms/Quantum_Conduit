"""Tests for toy qubit model builders."""

from __future__ import annotations

import torch

from qconduit.exact import exact_eigensystem, paulisum_to_dense
from qconduit.models import (
    diagonal_z_field,
    two_qubit_generic_chemistry_like,
)


def test_two_qubit_diagonal_z0():
    """Test two-qubit model with only Z⊗I term."""
    H = two_qubit_generic_chemistry_like(
        c_i=0.0, c_z0=1.0, c_z1=0.0, c_z0z1=0.0, c_xx=0.0, c_yy=0.0
    )

    # This is just Z⊗I
    # Eigenvalues should be [+1, +1, -1, -1] in some order
    evals, _ = exact_eigensystem(H, num_qubits=2)

    evals_sorted = torch.sort(evals)[0]
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=evals.dtype)
    expected_sorted = torch.sort(expected)[0]

    assert torch.allclose(evals_sorted, expected_sorted, atol=1e-10)


def test_two_qubit_xx_yy_coupling():
    """Test two-qubit model with symmetric XX + YY coupling."""
    H = two_qubit_generic_chemistry_like(
        c_i=0.0, c_z0=0.0, c_z1=0.0, c_z0z1=0.0, c_xx=0.5, c_yy=0.5
    )

    # Convert to dense matrix
    dense = paulisum_to_dense(H, num_qubits=2)

    # Matrix should be Hermitian
    assert torch.allclose(dense, dense.conj().T, atol=1e-10)

    # Trace should be zero (no identity term)
    trace = torch.trace(dense)
    assert abs(trace) < 1e-10


def test_two_qubit_zero_coefficients():
    """Test that zero coefficients are handled correctly."""
    H = two_qubit_generic_chemistry_like(
        c_i=0.0, c_z0=0.0, c_z1=0.0, c_z0z1=0.0, c_xx=0.0, c_yy=0.0
    )

    # Should have no terms (all zero)
    assert len(H.terms) == 0


def test_diagonal_z_field_basic():
    """Test diagonal Z field with specific values."""
    H = diagonal_z_field(num_qubits=3, local_fields=[1.0, -2.0, 0.5])

    # Convert to dense matrix
    dense = paulisum_to_dense(H, num_qubits=3)

    # Matrix should be diagonal
    off_diag = dense - torch.diag(torch.diag(dense))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-10)

    # Check specific diagonal entries
    # For |000> (index 0): E = 1.0 + (-2.0) + 0.5 = -0.5
    # But careful: in little-endian, |000> is index 0, and Z|0> = |0>, so
    # the sign depends on the bit value. For |000>, all bits are 0, so
    # Z_i|0> = +1|0>, so E = 1.0*1 + (-2.0)*1 + 0.5*1 = -0.5
    diag = torch.diag(dense)
    assert abs(diag[0].real - (-0.5)) < 1e-10

    # For |111> (index 7): all bits are 1, so Z_i|1> = -1|1>
    # E = 1.0*(-1) + (-2.0)*(-1) + 0.5*(-1) = -1.0 + 2.0 - 0.5 = 0.5
    assert abs(diag[7].real - 0.5) < 1e-10


def test_diagonal_z_field_validation():
    """Test that diagonal_z_field validates inputs correctly."""
    # Wrong length
    try:
        diagonal_z_field(num_qubits=2, local_fields=[1.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Negative num_qubits
    try:
        diagonal_z_field(num_qubits=0, local_fields=[])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        diagonal_z_field(num_qubits=-1, local_fields=[1.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_diagonal_z_field_single_qubit():
    """Test diagonal Z field with single qubit."""
    H = diagonal_z_field(num_qubits=1, local_fields=[2.5])

    # Should be 2.5 * Z
    dense = paulisum_to_dense(H, num_qubits=1)
    expected = torch.tensor([[2.5, 0.0], [0.0, -2.5]], dtype=torch.complex128)

    assert torch.allclose(dense, expected, atol=1e-10)


def test_diagonal_z_field_zero_fields():
    """Test diagonal Z field with all zero fields."""
    H = diagonal_z_field(num_qubits=3, local_fields=[0.0, 0.0, 0.0])

    # Should have no terms
    assert len(H.terms) == 0


def test_two_qubit_full_hamiltonian():
    """Test two-qubit model with all terms present."""
    H = two_qubit_generic_chemistry_like(
        c_i=1.0, c_z0=0.5, c_z1=-0.3, c_z0z1=0.2, c_xx=0.1, c_yy=0.1
    )

    # Should have 6 terms
    assert len(H.terms) == 6

    # Convert to dense and check Hermiticity
    dense = paulisum_to_dense(H, num_qubits=2)
    assert torch.allclose(dense, dense.conj().T, atol=1e-10)

    # Eigenvalues should be real
    evals, _ = exact_eigensystem(H, num_qubits=2)
    assert torch.allclose(evals, evals.real, atol=1e-10)

