"""Tests for circuit-based time-evolution functions."""

from __future__ import annotations

import pytest
import torch

from qconduit.backend.statevector import apply_gate, zero_state
from qconduit.core.device import default_device
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum, PauliTerm
from qconduit.time_evolution import (
    build_trotter_circuit,
    build_trotter_step_circuit,
    time_evolve_state,
)


class TestTrotterCircuits:
    """Tests for Trotter circuit builders."""

    def test_trotter_circuit_matches_state_evolution_single_qubit(self):
        """Test that circuit-based evolution matches direct state evolution."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 1

        omega = 0.9
        coeff = omega / 2.0
        term = PauliTerm(coeff=coeff, paulis=("Z",))
        H = PauliSum([term])

        state_init = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        state_init = apply_gate(state_init, H_gate, qubit=0, n_qubits=n_qubits)

        t = 0.5
        n_steps = 5

        # Direct state evolution
        evolved_direct = time_evolve_state(
            state=state_init.clone(),
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Circuit-based evolution: build Trotter circuit and simulate
        circuit = build_trotter_circuit(
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
        )

        # Simulate circuit starting from |+> state
        # We need to apply H first to get |+>, then simulate the circuit
        state_circ_init = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        state_circ_init = apply_gate(state_circ_init, H_gate, qubit=0, n_qubits=n_qubits)

        # Apply circuit gates manually to the initial state
        from qconduit.backend.statevector import apply_two_qubit_gate
        from qconduit.circuit.core import _resolve_single_qubit_gate, _resolve_two_qubit_gate

        state_circ = state_circ_init.clone()
        for op in circuit.ops:
            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate = _resolve_single_qubit_gate(name, op.params, dtype, dev.as_torch_device())
                state_circ = apply_gate(state_circ, gate, qubit=q, n_qubits=n_qubits)
            elif len(op.qubits) == 2:
                q0, q1 = op.qubits
                gate = _resolve_two_qubit_gate(name, q0, q1, dtype, dev.as_torch_device())
                state_circ = apply_two_qubit_gate(
                    state_circ, gate, qubit1=q0, qubit2=q1, n_qubits=n_qubits
                )

        # Compare up to global phase
        inner = (evolved_direct.conj() * state_circ).sum()
        global_phase = inner / inner.abs()
        phased = state_circ * global_phase.conj()
        assert torch.allclose(phased, evolved_direct, atol=1e-3, rtol=1e-3)

    def test_trotter_step_circuit_matches_trotter_step_function(self):
        """Test that step circuit matches direct step function."""
        dev = default_device()
        dtype = torch.complex64
        n_qubits = 2

        J = 0.7
        term1 = PauliTerm(coeff=J, paulis=("Z", "Z"))
        term2 = PauliTerm(coeff=0.2, paulis=("Z", "I"))
        H = PauliSum([term1, term2])

        from qconduit.backend.statevector import apply_two_qubit_gate
        state = zero_state(n_qubits=n_qubits, batch_shape=None, device=dev, dtype=dtype)
        H_gate = stdgates.H(dtype=dtype, device=dev.as_torch_device())
        # Prepare |++>
        state = apply_gate(state, H_gate, qubit=0, n_qubits=n_qubits)
        state = apply_gate(state, H_gate, qubit=1, n_qubits=n_qubits)

        dt = 0.1

        # Direct step
        from qconduit.time_evolution import trotter_step_pauli_sum
        step_state = trotter_step_pauli_sum(
            state=state.clone(),
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=2,
            device=dev,
        )

        # Circuit step
        circuit = build_trotter_step_circuit(
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=2,
        )

        # Apply circuit gates to the initial state
        from qconduit.circuit.core import _resolve_single_qubit_gate, _resolve_two_qubit_gate

        circ_state = state.clone()
        for op in circuit.ops:
            name = op.name.upper()
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate = _resolve_single_qubit_gate(name, op.params, dtype, dev.as_torch_device())
                circ_state = apply_gate(circ_state, gate, qubit=q, n_qubits=n_qubits)
            elif len(op.qubits) == 2:
                q0, q1 = op.qubits
                gate = _resolve_two_qubit_gate(name, q0, q1, dtype, dev.as_torch_device())
                circ_state = apply_two_qubit_gate(
                    circ_state, gate, qubit1=q0, qubit2=q1, n_qubits=n_qubits
                )

        inner = (step_state.conj() * circ_state).sum()
        global_phase = inner / inner.abs()
        phased = circ_state * global_phase.conj()
        assert torch.allclose(phased, step_state, atol=1e-4, rtol=1e-4)

    def test_build_trotter_step_circuit_first_order(self):
        """Test building a first-order Trotter step circuit."""
        n_qubits = 2
        term1 = PauliTerm(coeff=0.5, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("I", "Z"))
        H = PauliSum([term1, term2])

        dt = 0.1
        circuit = build_trotter_step_circuit(
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=1,
        )

        assert circuit.n_qubits == n_qubits
        assert len(circuit.ops) > 0

    def test_build_trotter_step_circuit_second_order(self):
        """Test building a second-order Trotter step circuit."""
        n_qubits = 2
        term1 = PauliTerm(coeff=0.5, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("I", "Z"))
        H = PauliSum([term1, term2])

        dt = 0.1
        circuit = build_trotter_step_circuit(
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=2,
        )

        assert circuit.n_qubits == n_qubits
        assert len(circuit.ops) > 0

    def test_build_trotter_circuit(self):
        """Test building a full Trotter circuit."""
        n_qubits = 2
        term1 = PauliTerm(coeff=0.5, paulis=("Z", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("I", "Z"))
        H = PauliSum([term1, term2])

        t = 1.0
        n_steps = 5
        circuit = build_trotter_circuit(
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=2,
        )

        assert circuit.n_qubits == n_qubits
        assert len(circuit.ops) > 0

    def test_build_trotter_circuit_invalid_n_steps(self):
        """Test that build_trotter_circuit raises error for invalid n_steps."""
        n_qubits = 1
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        H = PauliSum([term])

        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            build_trotter_circuit(
                hamiltonian=H,
                t=1.0,
                n_steps=0,
                n_qubits=n_qubits,
                order=1,
            )

    def test_build_trotter_step_circuit_invalid_order(self):
        """Test that build_trotter_step_circuit raises error for invalid order."""
        n_qubits = 1
        term = PauliTerm(coeff=1.0, paulis=("Z",))
        H = PauliSum([term])

        with pytest.raises(ValueError, match="Unsupported Trotter order"):
            build_trotter_step_circuit(
                hamiltonian=H,
                dt=0.1,
                n_qubits=n_qubits,
                order=3,  # type: ignore
            )

    def test_build_trotter_circuit_with_identity_term(self):
        """Test building a Trotter circuit with an identity term (should be skipped)."""
        n_qubits = 2
        term1 = PauliTerm(coeff=0.5, paulis=("I", "I"))
        term2 = PauliTerm(coeff=0.3, paulis=("Z", "I"))
        H = PauliSum([term1, term2])

        dt = 0.1
        circuit = build_trotter_step_circuit(
            hamiltonian=H,
            dt=dt,
            n_qubits=n_qubits,
            order=1,
        )

        # Circuit should still be valid (identity term adds no gates)
        assert circuit.n_qubits == n_qubits


