"""Tests for state tomography utilities."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.measurement import (
    pauli_matrix_from_label,
    pauli_expectation_from_statevector,
    single_qubit_pauli_expectations_from_statevector,
    reconstruct_single_qubit_density_from_pauli,
    two_qubit_pauli_expectations_from_statevector,
    reconstruct_two_qubit_density_from_pauli,
)


def test_pauli_matrix_from_label():
    """Test Pauli matrix construction from labels."""
    # Single qubit
    P_x = pauli_matrix_from_label("X")
    assert P_x.shape == (2, 2)
    expected_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    assert torch.allclose(P_x, expected_X)
    
    # Two qubits: ZZ
    P_zz = pauli_matrix_from_label("ZZ")
    assert P_zz.shape == (4, 4)
    # Check that P_zz @ P_zz = I_4 (Pauli matrices square to identity)
    identity_4 = torch.eye(4, dtype=torch.complex128)
    assert torch.allclose(P_zz @ P_zz, identity_4, atol=1e-10)
    
    # Invalid label: empty
    with pytest.raises(ValueError, match="non-empty"):
        pauli_matrix_from_label("")
    
    # Invalid label: invalid character
    with pytest.raises(ValueError, match="Invalid Pauli label"):
        pauli_matrix_from_label("A")


def test_single_qubit_pauli_expectations():
    """Test single-qubit Pauli expectation computations."""
    # |0⟩: ⟨Z⟩ = +1, ⟨X⟩ = 0, ⟨Y⟩ = 0
    state_0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    ex_x, ex_y, ex_z = single_qubit_pauli_expectations_from_statevector(state_0)
    assert abs(ex_z - 1.0) < 1e-10
    assert abs(ex_x) < 1e-10
    assert abs(ex_y) < 1e-10
    
    # |1⟩: ⟨Z⟩ = -1
    state_1 = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex128)
    ex_x1, ex_y1, ex_z1 = single_qubit_pauli_expectations_from_statevector(state_1)
    assert abs(ex_z1 - (-1.0)) < 1e-10
    assert abs(ex_x1) < 1e-10
    assert abs(ex_y1) < 1e-10
    
    # |+⟩ = (1/√2)(|0⟩ + |1⟩): ⟨X⟩ = +1, ⟨Z⟩ = 0, ⟨Y⟩ = 0
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state_plus = torch.tensor([sqrt2_inv + 0j, sqrt2_inv + 0j], dtype=torch.complex128)
    ex_x_plus, ex_y_plus, ex_z_plus = single_qubit_pauli_expectations_from_statevector(
        state_plus
    )
    assert abs(ex_x_plus - 1.0) < 1e-10
    assert abs(ex_z_plus) < 1e-10
    assert abs(ex_y_plus) < 1e-10
    
    # |+i⟩ = (1/√2)(|0⟩ + i|1⟩): ⟨Y⟩ = +1, others 0
    state_plus_i = torch.tensor(
        [sqrt2_inv + 0j, 1.0j * sqrt2_inv], dtype=torch.complex128
    )
    ex_x_pi, ex_y_pi, ex_z_pi = single_qubit_pauli_expectations_from_statevector(
        state_plus_i
    )
    assert abs(ex_y_pi - 1.0) < 1e-10
    assert abs(ex_x_pi) < 1e-10
    assert abs(ex_z_pi) < 1e-10
    
    # Test direct pauli_expectation_from_statevector
    ex_z_direct = pauli_expectation_from_statevector(state_0, "Z")
    assert abs(ex_z_direct.real - 1.0) < 1e-10
    assert abs(ex_z_direct.imag) < 1e-10


def test_single_qubit_reconstruction():
    """Test single-qubit density matrix reconstruction."""
    # Test for |0⟩
    state_0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    ex_x, ex_y, ex_z = single_qubit_pauli_expectations_from_statevector(state_0)
    rho = reconstruct_single_qubit_density_from_pauli(ex_x, ex_y, ex_z)
    
    # Check Hermiticity
    assert torch.allclose(rho, rho.conj().T, atol=1e-7)
    
    # Check trace
    assert torch.allclose(rho.trace(), torch.tensor(1.0, dtype=torch.complex128), atol=1e-7)
    
    # Compare with true density |0⟩⟨0|
    s = state_0 / state_0.norm()
    rho_true = s.unsqueeze(1) @ s.conj().unsqueeze(0)
    assert torch.allclose(rho, rho_true, atol=1e-7)
    
    # Test for |+⟩
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state_plus = torch.tensor([sqrt2_inv + 0j, sqrt2_inv + 0j], dtype=torch.complex128)
    ex_x_p, ex_y_p, ex_z_p = single_qubit_pauli_expectations_from_statevector(
        state_plus
    )
    rho_plus = reconstruct_single_qubit_density_from_pauli(ex_x_p, ex_y_p, ex_z_p)
    
    assert torch.allclose(rho_plus, rho_plus.conj().T, atol=1e-7)
    assert torch.allclose(
        rho_plus.trace(), torch.tensor(1.0, dtype=torch.complex128), atol=1e-7
    )
    
    s_plus = state_plus / state_plus.norm()
    rho_plus_true = s_plus.unsqueeze(1) @ s_plus.conj().unsqueeze(0)
    assert torch.allclose(rho_plus, rho_plus_true, atol=1e-7)
    
    # Test for |+i⟩
    state_plus_i = torch.tensor(
        [sqrt2_inv + 0j, 1.0j * sqrt2_inv], dtype=torch.complex128
    )
    ex_x_pi, ex_y_pi, ex_z_pi = single_qubit_pauli_expectations_from_statevector(
        state_plus_i
    )
    rho_plus_i = reconstruct_single_qubit_density_from_pauli(ex_x_pi, ex_y_pi, ex_z_pi)
    
    assert torch.allclose(rho_plus_i, rho_plus_i.conj().T, atol=1e-7)
    assert torch.allclose(
        rho_plus_i.trace(), torch.tensor(1.0, dtype=torch.complex128), atol=1e-7
    )
    
    s_plus_i = state_plus_i / state_plus_i.norm()
    rho_plus_i_true = s_plus_i.unsqueeze(1) @ s_plus_i.conj().unsqueeze(0)
    assert torch.allclose(rho_plus_i, rho_plus_i_true, atol=1e-7)


def test_two_qubit_bell_state_expectations():
    """Test two-qubit Pauli expectations for Bell state."""
    # Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state = torch.zeros(4, dtype=torch.complex128)
    state[0] = sqrt2_inv  # |00⟩
    state[3] = sqrt2_inv  # |11⟩
    
    exps = two_qubit_pauli_expectations_from_statevector(state)
    
    # Check known values
    assert abs(exps["II"] - 1.0) < 1e-10
    assert abs(exps["ZZ"] - 1.0) < 1e-10
    assert abs(exps["XX"] - 1.0) < 1e-10
    assert abs(exps["YY"] - (-1.0)) < 1e-10  # For |Φ+⟩, ⟨YY⟩ = -1
    assert abs(exps["ZI"]) < 1e-10
    assert abs(exps["IZ"]) < 1e-10


def test_two_qubit_reconstruction():
    """Test two-qubit density matrix reconstruction."""
    # Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    state = torch.zeros(4, dtype=torch.complex128)
    state[0] = sqrt2_inv
    state[3] = sqrt2_inv
    
    exps = two_qubit_pauli_expectations_from_statevector(state)
    rho = reconstruct_two_qubit_density_from_pauli(exps)
    
    # Check Hermiticity
    assert torch.allclose(rho, rho.conj().T, atol=1e-7)
    
    # Check trace
    assert torch.allclose(rho.trace(), torch.tensor(1.0, dtype=torch.complex128), atol=1e-7)
    
    # Compare with true density
    s = state / state.norm()
    rho_true = s.unsqueeze(1) @ s.conj().unsqueeze(0)
    assert torch.allclose(rho, rho_true, atol=1e-7)
    
    # Test missing label error
    exps_incomplete = exps.copy()
    del exps_incomplete["XY"]
    with pytest.raises(ValueError, match="Missing required"):
        reconstruct_two_qubit_density_from_pauli(exps_incomplete)


def test_tomography_error_paths():
    """Test error handling in tomography functions."""
    # pauli_expectation_from_statevector with mismatched dimensions
    state_1q = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="dimension.*does not match"):
        pauli_expectation_from_statevector(state_1q, "XX")
    
    # single_qubit_pauli_expectations_from_statevector with wrong length
    state_wrong = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="length 2"):
        single_qubit_pauli_expectations_from_statevector(state_wrong)
    
    # two_qubit_pauli_expectations_from_statevector with wrong length
    state_wrong2 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="length 4"):
        two_qubit_pauli_expectations_from_statevector(state_wrong2)
    
    # Zero norm state
    state_zero = torch.tensor([0.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError, match="zero norm"):
        single_qubit_pauli_expectations_from_statevector(state_zero)
    
    # Non-1D state
    state_2d = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="1D"):
        pauli_expectation_from_statevector(state_2d, "X")

