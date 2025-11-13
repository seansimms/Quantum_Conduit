"""Tests for exact time evolution."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.evolution import exact_time_evolution_statevector
from qconduit.operators import PauliSum, PauliTerm


def test_exact_evolution_1qubit_z_ground_state():
    """Test exact evolution for 1-qubit Z Hamiltonian with |0⟩ state."""
    # H = Z (Pauli Z on qubit 0, coefficient 1.0)
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    # Initial state |0⟩ = [1, 0]
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    # For any time t, |0⟩ is an eigenstate with eigenvalue +1
    # So evolution is just a global phase: |0⟩ -> e^(-i t) |0⟩
    t = 1.0
    psi_t = exact_time_evolution_statevector(psi0, H, time=t)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10

    # Check probability of |0⟩ is still 1 (up to global phase)
    prob_0 = abs(psi_t[0]) ** 2
    prob_1 = abs(psi_t[1]) ** 2
    assert abs(prob_0 - 1.0) < 1e-10
    assert abs(prob_1 - 0.0) < 1e-10


def test_exact_evolution_1qubit_z_plus_state():
    """Test exact evolution for 1-qubit Z Hamiltonian with |+⟩ state."""
    # H = Z
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    # Initial state |+⟩ = (1/√2)(|0⟩ + |1⟩)
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    psi0 = torch.tensor([sqrt2_inv + 0j, sqrt2_inv + 0j], dtype=torch.complex128)

    # At t = π/2, the state should be |+⟩ with a phase
    # |+⟩ is not an eigenstate, so it will evolve
    t = math.pi / 2.0
    psi_t = exact_time_evolution_statevector(psi0, H, time=t)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10

    # The probabilities of |0⟩ and |1⟩ should remain 0.5
    prob_0 = abs(psi_t[0]) ** 2
    prob_1 = abs(psi_t[1]) ** 2
    assert abs(prob_0 - 0.5) < 1e-10
    assert abs(prob_1 - 0.5) < 1e-10

    # Under Z evolution: |0⟩ -> e^(-i t) |0⟩, |1⟩ -> e^(i t) |1⟩
    # So |+⟩ -> (1/√2)(e^(-i t) |0⟩ + e^(i t) |1⟩)
    # At t = π/2: |+⟩ -> (1/√2)(-i |0⟩ + i |1⟩) = i (1/√2)(-|0⟩ + |1⟩)
    # The ratio psi_t[1]/psi_t[0] = e^(i t) / e^(-i t) = e^(2i t)
    # At t = π/2, this is e^(i π) = -1
    import cmath

    expected_phase_ratio = cmath.exp(2.0j * t)  # e^(2i t)
    phase_ratio = (psi_t[1] / psi_t[0]).item()
    # Check that the phase ratio matches (up to numerical precision)
    assert abs(phase_ratio - expected_phase_ratio) < 1e-10


def test_exact_evolution_invalid_state_length():
    """Test that non-power-of-two state length raises ValueError."""
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    # Invalid state length (3, not a power of 2)
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    with pytest.raises(ValueError, match="must be a power of 2"):
        exact_time_evolution_statevector(psi0, H, time=1.0)


def test_exact_evolution_invalid_state_shape():
    """Test that non-1D state raises ValueError."""
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    # 2D state (invalid)
    psi0 = torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.complex128)

    with pytest.raises(ValueError, match="must be 1D"):
        exact_time_evolution_statevector(psi0, H, time=1.0)


def test_exact_evolution_empty_state():
    """Test that empty state raises ValueError."""
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    # Empty state
    psi0 = torch.tensor([], dtype=torch.complex128)

    with pytest.raises(ValueError, match="nonzero length"):
        exact_time_evolution_statevector(psi0, H, time=1.0)


def test_exact_evolution_2qubit_commuting():
    """Test exact evolution for 2-qubit commuting Hamiltonian."""
    # H = Z ⊗ I + I ⊗ Z (commuting terms)
    H = PauliSum.from_terms(
        [
            PauliTerm(coeff=1.0, paulis=("Z", "I")),
            PauliTerm(coeff=1.0, paulis=("I", "Z")),
        ]
    )

    # Initial state |00⟩
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    t = 1.0
    psi_t = exact_time_evolution_statevector(psi0, H, time=t)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10

    # |00⟩ is an eigenstate, so it should remain |00⟩ (up to global phase)
    prob_00 = abs(psi_t[0]) ** 2
    assert abs(prob_00 - 1.0) < 1e-10


def test_exact_evolution_preserves_norm():
    """Test that exact evolution preserves state norm."""
    # H = X + Z (non-commuting terms)
    H = PauliSum.from_terms(
        [
            PauliTerm(coeff=1.0, paulis=("X",)),
            PauliTerm(coeff=1.0, paulis=("Z",)),
        ]
    )

    # Random normalized state
    psi0 = torch.tensor([0.6 + 0.3j, 0.7 - 0.1j], dtype=torch.complex128)
    psi0 = psi0 / torch.linalg.norm(psi0)

    t = 0.5
    psi_t = exact_time_evolution_statevector(psi0, H, time=t)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10


def test_exact_evolution_with_device():
    """Test exact evolution with explicit device parameter."""
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    device = torch.device("cpu")
    psi_t = exact_time_evolution_statevector(psi0, H, time=1.0, device=device)

    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10

