"""End-to-end integration tests.

This test module validates:
1. MaxCut + QAOA + VQE + sampling integration
2. Time evolution + measurement + sampling integration
3. Sweeps + VQE + QAOA integration
"""

import pytest
import torch
import math
import qconduit as qc
from qconduit.algorithms import ising_maxcut_hamiltonian, QAOAAnsatz
from qconduit.algorithms.vqe import VQE
from qconduit.operators import PauliTerm, PauliSum
from qconduit.operators.expectation import expectation_pauli_sum
from qconduit.sampling import sample_bitstrings_circuit, bitstring_counts
from qconduit.time_evolution import time_evolve_state, build_trotter_circuit
from qconduit.experiments import sweep_vqe_1d, sweep_vqe_2d


class TestMaxCutQAOAVQESampling:
    """Test MaxCut + QAOA + VQE + sampling integration."""

    def test_maxcut_qaoa_vqe_sampling_integration(self):
        """Test that VQE energy, direct statevector expectation, and sampled average agree."""
        # Build a small 3-node graph (triangle)
        num_nodes = 3
        edges = [(0, 1), (1, 2), (0, 2)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        # p=1 QAOAAnsatz
        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        # Choose a fixed parameter vector (not all zeros)
        params = torch.tensor([0.5, 0.3])

        # 1. Use VQE's energy(params) to compute expectation value
        energy_vqe = vqe.energy(params)

        # 2. Build the QAOA circuit, simulate the state, and compute energy directly
        circuit = ansatz.build_circuit(params)
        state = circuit.simulate_state()

        # Compute energy via statevector and Hamiltonian matrix
        H_matrix = H.to_matrix(dtype=state.dtype, device=state.device)
        energy_direct = torch.real(torch.vdot(state, H_matrix @ state))

        # 3. Sample bitstrings and compute empirical cut value
        generator = torch.Generator()
        generator.manual_seed(42)
        n_shots = 10000
        samples = sample_bitstrings_circuit(circuit, n_shots=n_shots, generator=generator)

        # Compute empirical energy from samples
        # For each bitstring, compute its cut value
        # H = sum_{(i,j) in E} w_ij * (1 - Z_i Z_j) / 2
        # For unweighted graph, w_ij = 1
        # Cut value for bitstring b: sum_{(i,j) in E} (1 - (-1)^{b_i + b_j}) / 2
        counts = bitstring_counts(samples)
        empirical_energy = 0.0
        total_count = sum(counts.values())

        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring]
            # Compute cut value for this bitstring
            cut_value = 0.0
            for i, j in edges:
                # If bits[i] != bits[j], edge is cut
                if bits[i] != bits[j]:
                    cut_value += 1.0
            # Add constant term (sum w_ij / 2 = 3/2 = 1.5)
            energy_for_bitstring = cut_value + 1.5
            empirical_energy += (count / total_count) * energy_for_bitstring

        # All three estimates should agree within tolerances
        # VQE and direct should be very close
        assert abs(energy_vqe.item() - energy_direct.item()) < 1e-5

        # Sampled average should be within statistical tolerance
        # For 10000 shots, standard error is ~0.01
        assert abs(energy_vqe.item() - empirical_energy) < 0.1


class TestTimeEvolutionMeasurementSampling:
    """Test time evolution + measurement + sampling integration."""

    def test_time_evolution_measurement_sampling_integration(self):
        """Test that statevector expectation, circuit expectation, and sampled estimate agree."""
        dev = qc.default_device()
        dtype = torch.complex64

        n_qubits = 2
        J = 0.3
        term = PauliTerm(coeff=J, paulis=("Z", "Z"))
        H = PauliSum.from_terms([term])

        # Start from |++⟩
        state = qc.zero_state(n_qubits=n_qubits, device=dev, dtype=dtype)
        h_gate = qc.H()
        state = qc.apply_gate(state, h_gate, qubit=0, n_qubits=n_qubits)
        state = qc.apply_gate(state, h_gate, qubit=1, n_qubits=n_qubits)

        t = 0.5
        n_steps = 5

        # 1. Use time_evolve_state to evolve
        evolved_state = time_evolve_state(
            state=state.clone(),
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

        # Measure ⟨Z₀Z₁⟩ via statevector expectation
        zz_term = PauliTerm(coeff=1.0, paulis=("Z", "Z"))
        zz_H = PauliSum.from_terms([zz_term])
        zz_exp_state = expectation_pauli_sum(evolved_state, zz_H, n_qubits=n_qubits)

        # 2. Build Trotter circuit, simulate, and compute ⟨Z₀Z₁⟩
        circuit = build_trotter_circuit(
            hamiltonian=H,
            t=t,
            n_steps=n_steps,
            n_qubits=n_qubits,
            order=1,
        )
        # Need to prepare |++⟩ first
        prep_circuit = qc.QuantumCircuit(n_qubits=n_qubits)
        prep_circuit.add_gate("H", [0])
        prep_circuit.add_gate("H", [1])
        # Combine circuits
        full_circuit = qc.QuantumCircuit(n_qubits=n_qubits)
        for op in prep_circuit.ops:
            full_circuit.add_gate(op.name, list(op.qubits), params=list(op.params) if op.params else None)
        for op in circuit.ops:
            full_circuit.add_gate(op.name, list(op.qubits), params=list(op.params) if op.params else None)

        evolved_circuit_state = full_circuit.simulate_state(device=dev, dtype=dtype)
        zz_exp_circuit = expectation_pauli_sum(evolved_circuit_state, zz_H, n_qubits=n_qubits)

        # 3. Sample bitstrings and estimate ⟨Z₀Z₁⟩ empirically
        generator = torch.Generator()
        generator.manual_seed(42)
        n_shots = 10000
        samples = sample_bitstrings_circuit(full_circuit, n_shots=n_shots, generator=generator)

        # Estimate ⟨Z₀Z₁⟩ from samples
        # Z₀Z₁ = +1 if bits are same, -1 if different
        counts = bitstring_counts(samples)
        total_count = sum(counts.values())
        zz_exp_sampled = 0.0

        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring]
            # Z₀Z₁ = +1 if bits[0] == bits[1], -1 otherwise
            zz_value = 1.0 if bits[0] == bits[1] else -1.0
            zz_exp_sampled += (count / total_count) * zz_value

        # All three values should match within tolerance
        assert abs(zz_exp_state.item() - zz_exp_circuit.item()) < 1e-4
        # Sampled estimate should be within statistical tolerance
        assert abs(zz_exp_state.item() - zz_exp_sampled) < 0.1


class TestSweepsVQEOAOA:
    """Test sweeps + VQE + QAOA integration."""

    def test_sweep_vqe_1d_reproduces_direct_calls(self):
        """Test that sweep_vqe_1d reproduces direct calls to VQE.energy."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        base_params = torch.tensor([0.5, 0.3])
        points = torch.linspace(0.0, math.pi, 5)

        # Run sweep
        result = sweep_vqe_1d(vqe, points, base_params=base_params, index=0)

        # Compare with direct calls
        for i, point in enumerate(points):
            params = base_params.clone()
            params[0] = point
            direct_energy = vqe.energy(params)
            sweep_energy = result.values[i]

            assert abs(direct_energy.item() - sweep_energy.item()) < 1e-6

    def test_sweep_vqe_2d_reproduces_direct_calls(self):
        """Test that sweep_vqe_2d reproduces direct calls to VQE.energy."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        base_params = torch.tensor([0.5, 0.3])
        x_points = torch.linspace(0.0, math.pi / 2, 3)
        y_points = torch.linspace(0.0, math.pi / 4, 2)

        # Run sweep
        result = sweep_vqe_2d(
            vqe, x_points, y_points, base_params, x_index=0, y_index=1
        )

        # Compare with direct calls
        for i, x_val in enumerate(x_points):
            for j, y_val in enumerate(y_points):
                params = base_params.clone()
                params[0] = x_val
                params[1] = y_val
                direct_energy = vqe.energy(params)
                sweep_energy = result.values[i, j]

                assert abs(direct_energy.item() - sweep_energy.item()) < 1e-6

    def test_sweep_vqe_no_shape_inconsistencies(self):
        """Test that sweeps don't produce shape or type inconsistencies."""
        num_nodes = 2
        edges = [(0, 1)]
        H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

        ansatz = QAOAAnsatz(n_qubits=num_nodes, problem_hamiltonian=H, p=1)
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        base_params = torch.tensor([0.5, 0.3])

        # 1D sweep
        points = torch.linspace(0.0, math.pi, 10)
        result_1d = sweep_vqe_1d(vqe, points, base_params=base_params, index=0)

        assert result_1d.points.shape == (10,)
        assert result_1d.values.shape == (10,)
        assert result_1d.points.dtype in [torch.float32, torch.float64]
        assert result_1d.values.dtype in [torch.float32, torch.float64]

        # 2D sweep
        x_points = torch.linspace(0.0, math.pi / 2, 5)
        y_points = torch.linspace(0.0, math.pi / 4, 3)
        result_2d = sweep_vqe_2d(
            vqe, x_points, y_points, base_params, x_index=0, y_index=1
        )

        assert result_2d.x_points.shape == (5,)
        assert result_2d.y_points.shape == (3,)
        assert result_2d.values.shape == (5, 3)
        assert result_2d.x_points.dtype in [torch.float32, torch.float64]
        assert result_2d.y_points.dtype in [torch.float32, torch.float64]
        assert result_2d.values.dtype in [torch.float32, torch.float64]


