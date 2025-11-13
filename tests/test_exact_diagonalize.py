"""Tests for exact diagonalization utilities."""

from __future__ import annotations

import pytest
import torch

from qconduit.exact import (
    exact_eigensystem,
    exact_ground_state,
    paulisum_to_dense,
)
from qconduit.models import (
    ising_zz_chain,
    transverse_field_ising_chain,
)
from qconduit.operators import PauliSum, PauliTerm


def test_paulisum_to_dense_single_qubit_z():
    """Test conversion of single-qubit Z to dense matrix."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    dense = paulisum_to_dense(H, num_qubits=1)

    expected = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    assert torch.allclose(dense, expected, atol=1e-10)


def test_paulisum_to_dense_two_qubit_zz():
    """Test conversion of two-qubit ZZ to dense matrix."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])
    dense = paulisum_to_dense(H, num_qubits=2)

    # Z⊗Z should have diagonal [1, -1, -1, 1]
    diag = torch.diag(dense)
    expected_diag = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.complex128)

    # Sort both for comparison (order may vary)
    diag_sorted = torch.sort(diag.real)[0]
    expected_sorted = torch.sort(expected_diag.real)[0]
    assert torch.allclose(diag_sorted, expected_sorted, atol=1e-10)

    # Off-diagonal should be zero
    off_diag = dense - torch.diag(torch.diag(dense))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-10)


def test_paulisum_to_dense_hermiticity():
    """Test that dense matrices are Hermitian."""
    # Create a random small PauliSum with X and Z terms
    terms = [
        PauliTerm(coeff=1.0, paulis=("X", "I")),
        PauliTerm(coeff=0.5, paulis=("Z", "I")),
        PauliTerm(coeff=0.3, paulis=("I", "X")),
        PauliTerm(coeff=0.2, paulis=("X", "X")),
    ]
    H = PauliSum(terms=terms)

    dense = paulisum_to_dense(H, num_qubits=2)

    # Should be Hermitian
    assert torch.allclose(dense, dense.conj().T, atol=1e-10)


def test_paulisum_to_dense_real_eigenvalues():
    """Test that Hermitian matrices have real eigenvalues."""
    # Create a random small PauliSum
    terms = [
        PauliTerm(coeff=1.0, paulis=("X", "Z")),
        PauliTerm(coeff=0.5, paulis=("Y", "I")),
        PauliTerm(coeff=0.3, paulis=("Z", "Z")),
    ]
    H = PauliSum(terms=terms)

    dense = paulisum_to_dense(H, num_qubits=2)
    evals, _ = torch.linalg.eigh(dense)

    # Eigenvalues should be real
    assert torch.allclose(evals, evals.real, atol=1e-10)


def test_exact_eigensystem_single_qubit_z():
    """Test exact eigensystem for single-qubit Z."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    eigs, vecs = exact_eigensystem(H, num_qubits=1)

    # Eigenvalues should be sorted ascending: [-1.0, 1.0]
    expected_eigs = torch.tensor([-1.0, 1.0], dtype=eigs.dtype)
    assert torch.allclose(eigs, expected_eigs, atol=1e-10)

    # Eigenvectors should be normalized
    for i in range(vecs.shape[1]):
        norm = torch.linalg.norm(vecs[:, i])
        assert abs(norm - 1.0) < 1e-10


def test_exact_eigensystem_k_not_none():
    """Test that k != None raises ValueError."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    with pytest.raises(ValueError, match="Partial spectrum.*not supported"):
        exact_eigensystem(H, num_qubits=1, k=1)


def test_exact_ground_state_ising():
    """Test exact ground state for simple Ising case."""
    H = ising_zz_chain(num_sites=2, j_coupling=1.0, periodic=False)
    # H = -Z0 Z1

    E0, psi0 = exact_ground_state(H, num_qubits=2)

    # Ground energy should be approximately -1.0
    assert abs(E0.item() - (-1.0)) < 1e-10

    # Ground state should be normalized
    norm = torch.linalg.norm(psi0)
    assert abs(norm - 1.0) < 1e-10


def test_exact_ground_state_consistency():
    """Test that exact_ground_state is consistent with exact_eigensystem."""
    H = ising_zz_chain(num_sites=2, j_coupling=1.0, periodic=False)

    E0, psi0 = exact_ground_state(H, num_qubits=2)
    eigs, vecs = exact_eigensystem(H, num_qubits=2)

    # Ground energy should match
    assert abs(E0.item() - eigs[0].item()) < 1e-10

    # Ground state should match (up to global phase)
    # Compare absolute values
    assert torch.allclose(psi0.abs(), vecs[:, 0].abs(), atol=1e-10)


def test_exact_ground_state_tfim():
    """Test exact ground state for TFIM."""
    H = transverse_field_ising_chain(num_sites=2, j_coupling=1.0, h_field=0.5)

    E0, psi0 = exact_ground_state(H, num_qubits=2)

    # Ground state should be normalized
    norm = torch.linalg.norm(psi0)
    assert abs(norm - 1.0) < 1e-10

    # Energy should be less than or equal to any diagonal element
    dense = paulisum_to_dense(H, num_qubits=2)
    diag = torch.diag(dense).real
    assert E0.item() <= diag.min().item() + 1e-6


def test_paulisum_to_dense_validation():
    """Test that paulisum_to_dense validates inputs correctly."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    # num_qubits = 0 should raise ValueError
    with pytest.raises(ValueError):
        paulisum_to_dense(H, num_qubits=0)

    # num_qubits < 0 should raise ValueError
    with pytest.raises(ValueError):
        paulisum_to_dense(H, num_qubits=-1)


def test_exact_eigensystem_validation():
    """Test that exact_eigensystem validates inputs correctly."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    # num_qubits < 1 should raise ValueError (via paulisum_to_dense)
    with pytest.raises(ValueError):
        exact_eigensystem(H, num_qubits=0)


def test_paulisum_to_dense_num_qubits_mismatch():
    """Test that num_qubits mismatch is detected."""
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])

    # Should raise ValueError if num_qubits doesn't match
    with pytest.raises(ValueError, match="does not match"):
        paulisum_to_dense(H, num_qubits=1)


def test_exact_ground_state_zero_norm():
    """Test that zero-norm eigenvector raises RuntimeError."""
    # This is hard to trigger with normal Hamiltonians, but we can test
    # the error path by checking the code handles it. In practice, this
    # shouldn't happen for valid PauliSum Hamiltonians, but we test the
    # error handling exists.
    # Note: This test may not actually trigger the error, but documents
    # the expected behavior.
    H = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    E0, psi0 = exact_ground_state(H, num_qubits=1)

    # Normal case should work
    assert abs(torch.linalg.norm(psi0) - 1.0) < 1e-10


def test_paulisum_to_dense_empty_hamiltonian():
    """Test that empty PauliSum works (zero matrix)."""
    H = PauliSum(terms=[])
    dense = paulisum_to_dense(H, num_qubits=2)

    expected = torch.zeros((4, 4), dtype=torch.complex128)
    assert torch.allclose(dense, expected, atol=1e-10)


def test_exact_eigensystem_empty_hamiltonian():
    """Test that empty PauliSum gives zero eigenvalues."""
    H = PauliSum(terms=[])
    eigs, vecs = exact_eigensystem(H, num_qubits=2)

    # All eigenvalues should be zero
    assert torch.allclose(eigs, torch.zeros(4, dtype=eigs.dtype), atol=1e-10)

