"""Tests for core time-evolution functions."""

from __future__ import annotations

import cmath
import math

import pytest
import torch

from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.core.device import default_device
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum, PauliTerm
from qconduit.time_evolution import (
    time_evolve_state,
    trotter_step_pauli_sum,
)


class TestTimeEvolveState:
    """Tests for time_evolve_state function."""

    def test_time_evolve_state_single_qubit_z_matches_analytic(self):
        """Test that Trotterized evolution matches analytic result for 1-qubit Z Hamiltonian."""
        dev = default_device()
        dtype = torch.complex64

        n_qubits = 1
        omega = 1.0
        coeff = omega / 2.0
        term = PauliTerm(coeff=coeff, paulis=("Z",))
        H = PauliSum([term])

        # Build |+> state
        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        state = apply_gate(state, H_gate, qubit=0, n_qubits=n_qubits)

        t = 0.7
        n_steps = 50  # reasonably fine Trotterization

        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Analytic solution (up to a global phase)
        phase0 = cmath.exp(-1j * omega * t / 2.0)
        phase1 = cmath.exp(1j * omega * t / 2.0)
        analytic = torch.empty_like(evolved)
        analytic[0] = (1.0 / math.sqrt(2.0)) * complex(phase0)
        analytic[1] = (1.0 / math.sqrt(2.0)) * complex(phase1)

        # Compare up to global phase: normalize the overlap
        inner = (analytic.conj() * evolved).sum()
        global_phase = inner / inner.abs()
        phased = evolved * global_phase.conj()
        assert torch.allclose(phased, analytic, atol=1e-3, rtol=1e-3)

    def test_trotter_single_term_is_exact(self):
        """Test that single-term Trotterization is exact (up to global phase)."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 2

        J = 0.3
        term = PauliTerm(coeff=J, paulis=("Z", "Z"))
        H = PauliSum([term])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        state = apply_gate(state, H_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, H_gate, qubit=1, n_qubits=n_qubits)

        t = 1.2
        # Since there's only one term, first-order Trotter is exact up to global phase.
        evolved_1_step = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=1,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        evolved_10_steps = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=10,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # For a single term, n_steps=1 and n_steps=10 should give the same result
        # (up to global phase) since there's no Trotter error
        inner = (evolved_1_step.conj() * evolved_10_steps).sum()
        global_phase = inner / inner.abs()
        phased = evolved_10_steps * global_phase.conj()
        assert torch.allclose(phased, evolved_1_step, atol=1e-4, rtol=1e-4)

    def test_trotter_step_pauli_sum_first_order(self):
        """Test that trotter_step_pauli_sum works for first-order Trotter."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 2

        term1 = PauliTerm(coeff=0.5, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("I", "Z"))
        H = PauliSum([term1, term2])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        state = apply_gate(state, H_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, H_gate, qubit=1, n_qubits=n_qubits)

        dt = 0.1
        evolved = trotter_step_pauli_sum(
            state=state,
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Check that state is normalized
        norm = torch.abs(evolved).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_trotter_step_pauli_sum_second_order(self):
        """Test that trotter_step_pauli_sum works for second-order Trotter."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 2

        term1 = PauliTerm(coeff=0.5, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("I", "Z"))
        H = PauliSum([term1, term2])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        state = apply_gate(state, H_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, H_gate, qubit=1, n_qubits=n_qubits)

        dt = 0.1
        evolved = trotter_step_pauli_sum(
            state=state,
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Check that state is normalized
        norm = torch.abs(evolved).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_time_evolve_state_invalid_n_steps(self):
        """Test that time_evolve_state raises error for invalid n_steps."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 1

        term = PauliTerm(coeff=1.0, paulis=("Z",))
        H = PauliSum([term])
        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)

        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            time_evolve_state(
                state=state,
                hamiltonian=H,
                t=1.0,
                n_steps=0,
                n_qubits=n_qubits,
                order=1,
                device=dev,
            )

    def test_trotter_step_pauli_sum_invalid_order(self):
        """Test that trotter_step_pauli_sum raises error for invalid order."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 1

        term = PauliTerm(coeff=1.0, paulis=("Z",))
        H = PauliSum([term])
        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)

        with pytest.raises(ValueError, match="Unsupported Trotter order"):
            trotter_step_pauli_sum(
                state=state,
                hamiltonian=H,
                dt=0.1,
                n_qubits=n_qubits,
                order=3,  # type: ignore
                device=dev,
            )

    def test_time_evolve_state_pauli_y_term(self):
        """Test time evolution with a Y Pauli term."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 1

        term = PauliTerm(coeff=0.5, paulis=("Y",))
        H = PauliSum([term])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)

        t = 0.5
        n_steps = 20
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Check that state is normalized
        norm = torch.abs(evolved).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_time_evolve_state_pauli_x_term(self):
        """Test time evolution with an X Pauli term."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 1

        term = PauliTerm(coeff=0.5, paulis=("X",))
        H = PauliSum([term])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)

        t = 0.5
        n_steps = 20
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Check that state is normalized
        norm = torch.abs(evolved).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_time_evolve_state_multi_qubit_pauli(self):
        """Test time evolution with a multi-qubit Pauli term."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 3

        term = PauliTerm(coeff=0.3, paulis=("X", "Y", "Z"))
        H = PauliSum([term])

        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)

        t = 0.3
        n_steps = 15
        evolved = time_evolve_state(
            state=state,
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Check that state is normalized
        norm = torch.abs(evolved).pow(2).sum()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

