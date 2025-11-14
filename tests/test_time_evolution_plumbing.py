"""Comprehensive tests for time evolution and Trotterization (G3).

This test module validates:
1. Single-qubit Z Hamiltonian evolution
2. Two-qubit ZZ Hamiltonian evolution
3. Multiple commuting terms
4. Circuit vs state evolution
5. Term decomposition correctness
"""

import pytest
import torch
import math
import cmath
import qconduit as qc
from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate, zero_state
from qconduit.operators import PauliTerm, PauliSum
from qconduit.time_evolution import (
    time_evolve_state,
    trotter_step_pauli_sum,
    build_trotter_circuit,
    build_trotter_step_circuit,
)
from qconduit.circuit import QuantumCircuit
from qconduit.operators.expectation import expectation_pauli_sum


class TestSingleQubitZHamiltonian:
    """Test single-qubit Z Hamiltonian evolution."""

    def test_single_qubit_z_hamiltonian_analytic(self):
        """Test H = (ω/2) Z; start from |+⟩ matches analytic solution."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 1
        omega = 1.0
        coeff = omega / 2.0
        term = PauliTerm(coeff=coeff, paulis=("Z",))
        H = PauliSum.from_terms([term])

        # Start from |+⟩
        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        t = 0.7
        # Single term, so 1-step Trotter is exact
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=1,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Analytic: |ψ(t)⟩ = (e^{-iωt/2}|0⟩ + e^{iωt/2}|1⟩)/√2
        phase0 = cmath.exp(-1j * omega * t / 2.0)
        phase1 = cmath.exp(1j * omega * t / 2.0)
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        analytic = torch.tensor(
            [sqrt2_inv * complex(phase0), sqrt2_inv * complex(phase1)],
            dtype=dtype,
            device=dev.as_torch_device(),
        )

        # Compare up to global phase
        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-3)

    def test_single_qubit_z_hamiltonian_many_steps(self):
        """Test that many-step Trotter also matches analytic."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 1
        omega = 1.0
        coeff = omega / 2.0
        term = PauliTerm(coeff=coeff, paulis=("Z",))
        H = PauliSum.from_terms([term])

        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)

        t = 0.5
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=50,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Analytic solution
        phase0 = cmath.exp(-1j * omega * t / 2.0)
        phase1 = cmath.exp(1j * omega * t / 2.0)
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        analytic = torch.tensor(
            [sqrt2_inv * complex(phase0), sqrt2_inv * complex(phase1)],
            dtype=dtype,
            device=dev.as_torch_device(),
        )

        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-3)


class TestTwoQubitZZHamiltonian:
    """Test two-qubit ZZ Hamiltonian evolution."""

    def test_two_qubit_zz_hamiltonian_exact(self):
        """Test H = J Z₀Z₁; start from |++⟩. Single term so first-order Trotter is exact."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.3
        term = PauliTerm(coeff=J, paulis=("Z", "Z"))
        H = PauliSum.from_terms([term])

        # Start from |++⟩
        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 1.2
        # Single term, so all Trotter orders should give same result
        evolved_1 = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=1,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        evolved_10 = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=10,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Should be the same up to global phase
        inner = (evolved_1.conj() * evolved_10).sum()
        global_phase = inner / inner.abs()
        phased = evolved_10 * global_phase.conj()
        assert torch.allclose(phased, evolved_1, atol=1e-4)


class TestMultipleCommutingTerms:
    """Test multiple commuting terms."""

    def test_commuting_terms_exact(self):
        """Test H = a Z₀ + b Z₁ with 2 qubits. Commuting terms so first-order Trotter is exact."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        a = 0.5
        b = 0.3
        term1 = PauliTerm(coeff=a, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=b, paulis=("I", "Z"))
        H = PauliSum.from_terms([term1, term2])

        # Start from |++⟩
        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 0.8
        # Commuting terms, so first-order Trotter is exact
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=1,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Compare with direct matrix exponentiation
        H_matrix = H.to_matrix(dtype=dtype, device=dev.as_torch_device())
        U = torch.linalg.matrix_exp(-1j * t * H_matrix)
        analytic = U @ state

        # Compare up to global phase
        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-4)

    def test_commuting_terms_many_steps(self):
        """Test that many-step Trotter also works for commuting terms."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        a = 0.5
        b = 0.3
        term1 = PauliTerm(coeff=a, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=b, paulis=("I", "Z"))
        H = PauliSum.from_terms([term1, term2])

        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 0.5
        evolved_1 = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=1,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        evolved_20 = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=20,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Should be the same up to global phase
        inner = (evolved_1.conj() * evolved_20).sum()
        global_phase = inner / inner.abs()
        phased = evolved_20 * global_phase.conj()
        assert torch.allclose(phased, evolved_1, atol=1e-4)


class TestCircuitVsStateEvolution:
    """Test circuit vs state evolution consistency."""

    def test_trotter_circuit_matches_state_evolution(self):
        """Test that Trotter circuit simulation matches state evolution."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.3
        term = PauliTerm(coeff=J, paulis=("Z", "Z"))
        H = PauliSum.from_terms([term])

        # Start from |++⟩
        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 0.5
        n_steps = 5

        # State evolution
        evolved_state = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Circuit evolution
        circuit = build_trotter_circuit(
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
        )
        evolved_circuit = circuit.simulate_state(device=dev, dtype=dtype)

        # Compare up to global phase
        inner = (evolved_state.conj() * evolved_circuit).sum()
        global_phase = inner / inner.abs()
        phased = evolved_circuit * global_phase.conj()
        assert torch.allclose(phased, evolved_state, atol=1e-4)

    def test_trotter_circuit_expectation_matches(self):
        """Test that expectation values match between circuit and state evolution."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.3
        term = PauliTerm(coeff=J, paulis=("Z", "Z"))
        H = PauliSum.from_terms([term])

        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 0.3
        n_steps = 3

        # State evolution
        evolved_state = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Circuit evolution
        circuit = build_trotter_circuit(
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
        )
        evolved_circuit = circuit.simulate_state(device=dev, dtype=dtype)

        # Compute ⟨Z₀Z₁⟩
        zz_term = PauliTerm(coeff=1.0, paulis=("Z", "Z"))
        zz_H = PauliSum.from_terms([zz_term])

        exp_state = expectation_pauli_sum(evolved_state, zz_H, n_qubits=n_qubits)
        exp_circuit = expectation_pauli_sum(evolved_circuit, zz_H, n_qubits=n_qubits)

        assert torch.allclose(exp_state, exp_circuit, atol=1e-4)


class TestTermDecompositionCorrectness:
    """Test term decomposition correctness for various Pauli terms."""

    def test_xy_term_decomposition(self):
        """Test H = J X₀Y₁ decomposition matches dense matrix evolution."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.2
        term = PauliTerm(coeff=J, paulis=("X", "Y"))
        H = PauliSum.from_terms([term])

        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        dt = 0.1
        # Small dt for better accuracy
        evolved = trotter_step_pauli_sum(
            state=state,
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Compare with dense matrix evolution
        H_matrix = H.to_matrix(dtype=dtype, device=dev.as_torch_device())
        U = torch.linalg.matrix_exp(-1j * dt * H_matrix)
        analytic = U @ state

        # Compare up to global phase
        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-3)

    def test_yx_term_decomposition(self):
        """Test H = J Y₀X₁ decomposition."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.2
        term = PauliTerm(coeff=J, paulis=("Y", "X"))
        H = PauliSum.from_terms([term])

        state = zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        dt = 0.1
        evolved = trotter_step_pauli_sum(
            state=state,
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Compare with dense matrix
        H_matrix = H.to_matrix(dtype=dtype, device=dev.as_torch_device())
        U = torch.linalg.matrix_exp(-1j * dt * H_matrix)
        analytic = U @ state

        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-3)


