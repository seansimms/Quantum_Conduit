"""Comprehensive tests for QAOA Ising/MaxCut layer (G4).

This test module validates:
1. Ising/MaxCut Hamiltonian construction
2. QAOAAnsatz zero-parameter behavior
3. QAOAAnsatz rejects non-diagonal H
4. Simple QAOA energy checks
"""

import pytest
import torch
import math
import qconduit as qc
from qconduit.algorithms import ising_maxcut_hamiltonian, QAOAAnsatz
from qconduit.operators import PauliTerm, PauliSum
from qconduit.algorithms.vqe import VQE
from qconduit.backend.statevector import measure_probs


class TestIsingMaxCutHamiltonianConstruction:
    """Test Ising/MaxCut Hamiltonian construction."""

    def test_ising_maxcut_hamiltonian_two_nodes(self):
        """Test num_nodes=2, edges=[(0,1)] with include_constant=True."""
        num_nodes = 2
        edges = [(0, 1)]

        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        # Should have one ZZ term with coeff -0.5 and one II term with coeff +0.5
        zz_coeff = None
        ii_coeff = None
        for term in H.terms:
            if term.paulis == ("Z", "Z"):
                zz_coeff = term.coeff
            elif term.paulis == ("I", "I"):
                ii_coeff = term.coeff

        assert zz_coeff is not None
        assert abs(float(zz_coeff) + 0.5) < 1e-8
        assert ii_coeff is not None
        assert abs(float(ii_coeff) - 0.5) < 1e-8

    def test_ising_maxcut_hamiltonian_triangle_weights(self):
        """Test 3-node triangle with weights [1,2,3]."""
        num_nodes = 3
        edges = [(0, 1), (1, 2), (0, 2)]
        weights = [1.0, 2.0, 3.0]

        H = ising_maxcut_hamiltonian(
            num_nodes=num_nodes, edges=edges, weights=weights, include_constant=True
        )

        # Constant term sum should be (1+2+3)/2 = 3
        const_coeff = 0.0
        zz_coeff_sum = 0.0
        for term in H.terms:
            if all(p == "I" for p in term.paulis):
                const_coeff += float(term.coeff)
            else:
                zz_coeff_sum += float(term.coeff)

        assert abs(const_coeff - 3.0) < 1e-8
        # Sum of ZZ coefficients should be -3
        assert abs(zz_coeff_sum + 3.0) < 1e-8


class TestQAOAAnsatzZeroParameterBehavior:
    """Test QAOAAnsatz zero-parameter behavior."""

    def test_qaoa_ansatz_zero_params_uniform_superposition(self):
        """Test that QAOAAnsatz at params=0 prepares uniform superposition."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)

        # At params = 0, should prepare |+⟩^n
        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)

        # Simulate and check probabilities
        state = circuit.simulate_state()
        probs = measure_probs(state, n_qubits=num_nodes)

        # Uniform superposition: all probabilities ≈ 1/2^n = 1/4
        expected_prob = 1.0 / (2**num_nodes)
        for i in range(2**num_nodes):
            assert abs(probs[i].item() - expected_prob) < 1e-6

    def test_qaoa_ansatz_zero_params_p_greater_than_one(self):
        """Test zero-parameter behavior for p > 1."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=2)

        params = torch.zeros(ansatz.num_parameters)
        circuit = ansatz.build_circuit(params)

        state = circuit.simulate_state()
        probs = measure_probs(state, n_qubits=num_nodes)

        # Should still be uniform
        expected_prob = 1.0 / (2**num_nodes)
        for i in range(2**num_nodes):
            assert abs(probs[i].item() - expected_prob) < 1e-6


class TestQAOAAnsatzRejectsNonDiagonalH:
    """Test that QAOAAnsatz rejects non-diagonal Hamiltonians."""

    def test_qaoa_ansatz_rejects_x_term(self):
        """Test that QAOAAnsatz raises ValueError for X term."""
        # Create Hamiltonian with X term
        x_term = PauliTerm(coeff=1.0, paulis=("X",))
        H = PauliSum.from_terms([x_term])

        with pytest.raises(ValueError, match="only supports diagonal"):
            QAOAAnsatz(n_qubits=1, problem_hamiltonian=H, p=1)

    def test_qaoa_ansatz_rejects_y_term(self):
        """Test that QAOAAnsatz raises ValueError for Y term."""
        y_term = PauliTerm(coeff=1.0, paulis=("Y",))
        H = PauliSum.from_terms([y_term])

        with pytest.raises(ValueError, match="only supports diagonal"):
            QAOAAnsatz(n_qubits=1, problem_hamiltonian=H, p=1)

    def test_qaoa_ansatz_accepts_z_only(self):
        """Test that QAOAAnsatz accepts Z-only Hamiltonian."""
        z_term = PauliTerm(coeff=1.0, paulis=("Z",))
        H = PauliSum.from_terms([z_term])

        # Should not raise
        ansatz = QAOAAnsatz(n_qubits=1, problem_hamiltonian=H, p=1)
        assert ansatz.n_qubits == 1
        assert ansatz.p == 1


class TestSimpleQAOAEnergyCheck:
    """Test simple QAOA energy evaluation."""

    def test_qaoa_energy_at_zero_params(self):
        """Test QAOA energy at (γ,β) = (0,0) is 0.5 for 2-node MaxCut."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        # At (γ,β) = (0,0)
        params = torch.zeros(ansatz.num_parameters)
        energy = vqe.energy(params)

        # For uniform superposition, energy should be 0.5
        assert abs(energy.item() - 0.5) < 1e-6

    def test_qaoa_energy_can_find_better_cut(self):
        """Test that QAOA can find better cuts than random baseline."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        # Energy at (0,0) is 0.5
        params_zero = torch.zeros(ansatz.num_parameters)
        energy_zero = vqe.energy(params_zero)

        # Try a few parameter values
        found_better = False
        for gamma in [0.1, 0.5, 1.0, math.pi / 4]:
            for beta in [0.1, 0.5, 1.0, math.pi / 4]:
                params = torch.tensor([gamma, beta])
                energy = vqe.energy(params)
                # Lower energy is better for MaxCut (we're minimizing)
                if energy.item() < energy_zero.item() - 1e-6:
                    found_better = True
                    break
            if found_better:
                break

        # Should find at least one parameter set with better energy
        assert found_better, "QAOA should be able to find better cuts than random baseline"

    def test_qaoa_energy_landscape(self):
        """Test QAOA energy landscape for small parameter grid."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        # Sample a small grid
        gammas = torch.linspace(0.0, math.pi / 2, 5)
        betas = torch.linspace(0.0, math.pi / 4, 3)

        energies = []
        for gamma in gammas:
            for beta in betas:
                params = torch.tensor([gamma.item(), beta.item()])
                energy = vqe.energy(params)
                energies.append(energy.item())

        # All energies should be finite
        assert all(math.isfinite(e) for e in energies)

        # Minimum energy should be <= 0.5 (random baseline)
        min_energy = min(energies)
        assert min_energy <= 0.5 + 1e-6


