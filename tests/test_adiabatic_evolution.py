"""Tests for adiabatic evolution functions."""

from __future__ import annotations

import pytest
import torch

from qconduit.adiabatic import (
    AdiabaticConfig,
    adiabatic_evolve_state,
    adiabatic_x_mixer_to_problem_state,
    build_adiabatic_circuit,
    build_x_mixer_hamiltonian,
    linear_schedule,
)
from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.circuit import QuantumCircuit
from qconduit.gates.standard import H
from qconduit.operators import PauliSum, PauliTerm
from qconduit.operators.expectation import expectation_pauli_sum
from qconduit.time_evolution import time_evolve_state


def _create_plus_state(n_qubits: int, device=None, dtype=torch.complex64):
    """Create |+⟩ state by applying H to |0⟩."""
    state = zero_state(n_qubits=n_qubits, device=device, dtype=dtype)
    h_gate = H(dtype=dtype, device=state.device)
    state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
    return state


def test_adiabatic_config_validation():
    """Test AdiabaticConfig validation."""
    schedule = linear_schedule(num_steps=5)

    # Valid config
    config = AdiabaticConfig(
        total_time=1.0, num_steps=5, schedule=schedule, trotter_steps_per_interval=1
    )
    assert config.total_time == 1.0
    assert config.num_steps == 5

    # Invalid total_time
    with pytest.raises(ValueError, match="total_time must be positive"):
        AdiabaticConfig(total_time=0.0, num_steps=5, schedule=schedule)

    with pytest.raises(ValueError, match="total_time must be positive"):
        AdiabaticConfig(total_time=-1.0, num_steps=5, schedule=schedule)

    # Invalid num_steps
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        AdiabaticConfig(total_time=1.0, num_steps=0, schedule=schedule)

    # Invalid schedule shape
    bad_schedule = torch.tensor([[0.0, 1.0], [0.5, 0.5]])  # 2D
    with pytest.raises(ValueError, match="schedule must be 1D tensor"):
        AdiabaticConfig(total_time=1.0, num_steps=5, schedule=bad_schedule)

    # Schedule length mismatch
    short_schedule = linear_schedule(num_steps=3)
    with pytest.raises(ValueError, match="schedule length"):
        AdiabaticConfig(total_time=1.0, num_steps=5, schedule=short_schedule)

    # Invalid schedule values
    bad_values = torch.tensor([0.0, 0.5, 1.5, 1.0])  # Value > 1
    with pytest.raises(ValueError, match="schedule values must be in \\[0, 1\\]"):
        AdiabaticConfig(total_time=1.0, num_steps=4, schedule=bad_values)

    # Invalid trotter_steps_per_interval
    with pytest.raises(ValueError, match="trotter_steps_per_interval must be >= 1"):
        AdiabaticConfig(
            total_time=1.0, num_steps=5, schedule=schedule, trotter_steps_per_interval=0
        )


def test_constant_hamiltonian_reduces_to_time_evolution():
    """Test that adiabatic evolution with H_initial == H_final matches time evolution."""
    n_qubits = 1
    psi0 = _create_plus_state(n_qubits)

    # H0 = H1 = Z
    H_z = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    total_time = 1.234
    num_steps = 10
    schedule = linear_schedule(num_steps=num_steps)
    config = AdiabaticConfig(
        total_time=total_time,
        num_steps=num_steps,
        schedule=schedule,
        trotter_steps_per_interval=1,
    )

    # Adiabatic evolution
    psi_ad = adiabatic_evolve_state(psi0, H_z, H_z, config)

    # Reference: direct time evolution
    psi_ref = time_evolve_state(
        state=psi0,
        hamiltonian=H_z,
        t=total_time,
        n_steps=num_steps,
        n_qubits=n_qubits,
        order=1,
    )

    # Compare up to global phase
    # Normalize both
    psi_ad_norm = psi_ad / torch.linalg.norm(psi_ad)
    psi_ref_norm = psi_ref / torch.linalg.norm(psi_ref)

    # Compute overlap
    overlap = (psi_ad_norm.conj() * psi_ref_norm).sum()
    overlap_abs = abs(overlap)

    # Should be very close to 1 (up to global phase)
    assert overlap_abs > 0.999, f"Overlap {overlap_abs} too low"


def test_circuit_vs_state_evolution_equivalence():
    """Test that circuit simulation matches state evolution for constant H."""
    n_qubits = 1
    psi0 = _create_plus_state(n_qubits)

    H_z = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    total_time = 0.5
    num_steps = 5
    schedule = linear_schedule(num_steps=num_steps)
    config = AdiabaticConfig(
        total_time=total_time,
        num_steps=num_steps,
        schedule=schedule,
        trotter_steps_per_interval=1,
    )

    # State evolution
    psi_state = adiabatic_evolve_state(psi0, H_z, H_z, config)

    # Circuit evolution
    circuit = build_adiabatic_circuit(n_qubits, H_z, H_z, config)
    # Simulate circuit from |0⟩, then apply initial state preparation
    # Actually, we need to simulate with the initial state
    # Let's build a circuit that prepares |+⟩ then applies adiabatic evolution
    prep_circuit = QuantumCircuit(n_qubits=n_qubits)
    prep_circuit.add_gate("H", [0])
    for op in circuit.ops:
        prep_circuit.add_gate(
            op.name,
            list(op.qubits),
            params=list(op.params) if op.params is not None else None,
        )

    psi_circ = prep_circuit.simulate_state(dtype=psi0.dtype)

    # Compare up to global phase
    psi_state_norm = psi_state / torch.linalg.norm(psi_state)
    psi_circ_norm = psi_circ / torch.linalg.norm(psi_circ)

    overlap = (psi_state_norm.conj() * psi_circ_norm).sum()
    overlap_abs = abs(overlap)

    assert overlap_abs > 0.999, f"Overlap {overlap_abs} too low"


def test_x_mixer_to_z_problem_basic():
    """Test basic X mixer to Z problem path."""
    n_qubits = 1
    psi0 = _create_plus_state(n_qubits)

    # Problem Hamiltonian: Z
    H_problem = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    schedule = linear_schedule(num_steps=8)
    config = AdiabaticConfig(
        total_time=2.0, num_steps=8, schedule=schedule, trotter_steps_per_interval=1
    )

    psi_final = adiabatic_x_mixer_to_problem_state(psi0, H_problem, config)

    # Basic checks
    norm = torch.linalg.norm(psi_final)
    assert norm == pytest.approx(1.0, abs=1e-4)

    # Energy should be real and between min/max eigenvalues of H_problem
    energy = expectation_pauli_sum(psi_final, H_problem).real.item()
    # For H = Z, eigenvalues are -1 and 1
    assert -1.0 <= energy <= 1.0


def test_x_mixer_hamiltonian():
    """Test build_x_mixer_hamiltonian."""
    n_qubits = 3
    H_mixer = build_x_mixer_hamiltonian(n_qubits, strength=1.0)

    assert H_mixer.n_qubits() == n_qubits
    assert len(H_mixer.terms) == n_qubits

    # Check each term is -X_i
    for i in range(n_qubits):
        term = H_mixer.terms[i]
        assert term.coeff == pytest.approx(-1.0, abs=1e-10)
        # Check that X is on qubit i
        assert term.paulis[i] == "X"
        # Check all other qubits are I
        for j in range(n_qubits):
            if j != i:
                assert term.paulis[j] == "I"


def test_x_mixer_hamiltonian_strength():
    """Test build_x_mixer_hamiltonian with custom strength."""
    n_qubits = 2
    strength = 2.5
    H_mixer = build_x_mixer_hamiltonian(n_qubits, strength=strength)

    for term in H_mixer.terms:
        assert term.coeff == pytest.approx(-strength, abs=1e-10)


def test_x_mixer_hamiltonian_invalid():
    """Test build_x_mixer_hamiltonian with invalid num_qubits."""
    with pytest.raises(ValueError, match="num_qubits must be >= 1"):
        build_x_mixer_hamiltonian(0)


def test_adiabatic_evolve_state_invalid_state_dimension():
    """Test that adiabatic_evolve_state raises ValueError for invalid state dimensions."""
    # State with length not a power of 2
    bad_state = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)

    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z",))])
    schedule = linear_schedule(num_steps=5)
    config = AdiabaticConfig(total_time=1.0, num_steps=5, schedule=schedule)

    with pytest.raises(ValueError, match="initial_state length must be a power of 2"):
        adiabatic_evolve_state(bad_state, H0, H1, config)


def test_adiabatic_evolve_state_wrong_n_qubits():
    """Test that adiabatic_evolve_state raises ValueError for mismatched n_qubits."""
    n_qubits = 1
    psi0 = _create_plus_state(n_qubits)

    # Hamiltonians for 2 qubits
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "I"))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z", "I"))])

    schedule = linear_schedule(num_steps=5)
    config = AdiabaticConfig(total_time=1.0, num_steps=5, schedule=schedule)

    with pytest.raises(ValueError, match="acts on.*qubits"):
        adiabatic_evolve_state(psi0, H0, H1, config)


def test_build_adiabatic_circuit_invalid_n_qubits():
    """Test that build_adiabatic_circuit raises ValueError for invalid n_qubits."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z",))])
    schedule = linear_schedule(num_steps=5)
    config = AdiabaticConfig(total_time=1.0, num_steps=5, schedule=schedule)

    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        build_adiabatic_circuit(0, H0, H1, config)


def test_adiabatic_energy_monotonicity():
    """Test that energy (roughly) decreases as total_time increases."""
    n_qubits = 1
    psi0 = _create_plus_state(n_qubits)
    H_problem = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    schedule1 = linear_schedule(num_steps=5)
    config1 = AdiabaticConfig(
        total_time=1.0, num_steps=5, schedule=schedule1, trotter_steps_per_interval=1
    )

    schedule2 = linear_schedule(num_steps=5)
    config2 = AdiabaticConfig(
        total_time=2.0, num_steps=5, schedule=schedule2, trotter_steps_per_interval=1
    )

    psi1 = adiabatic_x_mixer_to_problem_state(psi0, H_problem, config1)
    psi2 = adiabatic_x_mixer_to_problem_state(psi0, H_problem, config2)

    E1 = expectation_pauli_sum(psi1, H_problem).real.item()
    E2 = expectation_pauli_sum(psi2, H_problem).real.item()

    # Energy should not increase (allowing small tolerance for numerical errors)
    assert E2 <= E1 + 1e-3, f"Energy increased: E1={E1}, E2={E2}"


def test_adiabatic_evolve_state_2d_state():
    """Test that adiabatic_evolve_state raises ValueError for 2D state."""
    bad_state = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)

    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z",))])
    schedule = linear_schedule(num_steps=5)
    config = AdiabaticConfig(total_time=1.0, num_steps=5, schedule=schedule)

    with pytest.raises(ValueError, match="initial_state must be 1D"):
        adiabatic_evolve_state(bad_state, H0, H1, config)


