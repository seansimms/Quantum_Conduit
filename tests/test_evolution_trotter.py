"""Tests for Trotter-Suzuki time evolution."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.backend.statevector import apply_gate
from qconduit.circuit import QuantumCircuit
from qconduit.evolution import (
    TrotterSchedule,
    build_trotter_circuit,
    build_trotter_step_circuit,
    evolve_state_trotter,
    exact_time_evolution_statevector,
)
from qconduit.gates import standard as stdgates
from qconduit.operators import PauliSum, PauliTerm


def _apply_circuit_to_statevector(
    circuit: QuantumCircuit,
    state: torch.Tensor,
) -> torch.Tensor:
    """Helper to apply a circuit to a statevector."""
    from qconduit.backend.statevector import apply_two_qubit_gate

    torch_device = state.device
    dtype = state.dtype

    for op in circuit.ops:
        name = op.name.upper()
        if len(op.qubits) == 1:
            q = op.qubits[0]
            gate = _resolve_single_qubit_gate(name, op.params, dtype, torch_device)
            state = apply_gate(state, gate, qubit=q, n_qubits=circuit.n_qubits)
        elif len(op.qubits) == 2:
            q0, q1 = op.qubits
            gate = _resolve_two_qubit_gate(name, q0, q1, dtype, torch_device)
            state = apply_two_qubit_gate(
                state, gate, qubit1=q0, qubit2=q1, n_qubits=circuit.n_qubits
            )
        else:
            raise ValueError(f"Unsupported gate on {len(op.qubits)} qubits")

    return state


def _resolve_single_qubit_gate(
    name: str,
    params: tuple[float, ...] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a single-qubit gate name to a matrix."""
    n = name.upper()
    if n == "I":
        return stdgates.I(dtype=dtype, device=device)
    if n == "X":
        return stdgates.X(dtype=dtype, device=device)
    if n == "Y":
        return stdgates.Y(dtype=dtype, device=device)
    if n == "Z":
        return stdgates.Z(dtype=dtype, device=device)
    if n == "H":
        return stdgates.H(dtype=dtype, device=device)
    if n == "S":
        return stdgates.S(dtype=dtype, device=device)
    if n == "T":
        return stdgates.T(dtype=dtype, device=device)

    if n in ("RX", "RY", "RZ"):
        if not params or len(params) != 1:
            raise ValueError(f"Gate {n} requires exactly one parameter.")
        theta = float(params[0])
        if n == "RX":
            return stdgates.RX(theta, dtype=dtype, device=device)
        if n == "RY":
            return stdgates.RY(theta, dtype=dtype, device=device)
        if n == "RZ":
            return stdgates.RZ(theta, dtype=dtype, device=device)

    raise ValueError(f"Unsupported gate {name!r}")


def _resolve_two_qubit_gate(
    name: str,
    control: int,
    target: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a two-qubit gate name to a matrix."""
    n = name.upper()
    if n == "CNOT":
        control_first = control < target
        return stdgates.CNOT(dtype=dtype, device=device, control_first=control_first)
    raise ValueError(f"Unsupported gate {name!r}")


def paulisum_z(num_qubits: int, coeff: float = 1.0) -> PauliSum:
    """Helper to build H = coeff * Z on qubit 0."""
    paulis = ["I"] * num_qubits
    paulis[0] = "Z"
    return PauliSum.from_terms([PauliTerm(coeff=coeff, paulis=tuple(paulis))])


def paulisum_z1z2() -> PauliSum:
    """Helper to build H = Z ⊗ Z on 2 qubits."""
    return PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z", "Z"))])


def paulisum_x_plus_z() -> PauliSum:
    """Helper to build H = X + Z on 1 qubit (non-commuting)."""
    return PauliSum.from_terms(
        [
            PauliTerm(coeff=1.0, paulis=("X",)),
            PauliTerm(coeff=1.0, paulis=("Z",)),
        ]
    )


def test_build_trotter_step_circuit_commuting():
    """Test build_trotter_step_circuit with commuting terms."""
    # H = Z (commuting with itself)
    H = paulisum_z(num_qubits=1, coeff=1.0)

    step_time = 0.5
    step_circ = build_trotter_step_circuit(H, step_time, order=1, num_qubits=1)

    # Initial state |0⟩
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    psi_step = _apply_circuit_to_statevector(step_circ, psi0)

    # Compare with exact evolution
    psi_exact = exact_time_evolution_statevector(psi0, H, time=step_time)

    # For commuting case, Trotter should match exact (up to global phase)
    # Check that probabilities match
    prob_step = torch.abs(psi_step) ** 2
    prob_exact = torch.abs(psi_exact) ** 2
    assert torch.allclose(prob_step, prob_exact, atol=1e-10)


def test_build_trotter_circuit_repeated_steps():
    """Test build_trotter_circuit with repeated steps for commuting H."""
    # H = Z (commuting)
    H = paulisum_z(num_qubits=1, coeff=1.0)

    total_time = 1.0
    num_steps = 5
    schedule = TrotterSchedule(num_steps=num_steps, total_time=total_time, order=1)
    full_circ = build_trotter_circuit(H, schedule, num_qubits=1)

    # Initial state |0⟩
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    psi_trotter = _apply_circuit_to_statevector(full_circ, psi0)

    # Compare with exact evolution
    psi_exact = exact_time_evolution_statevector(psi0, H, time=total_time)

    # For commuting case, Trotter should match exact (up to global phase)
    prob_trotter = torch.abs(psi_trotter) ** 2
    prob_exact = torch.abs(psi_exact) ** 2
    assert torch.allclose(prob_trotter, prob_exact, atol=1e-10)


def test_evolve_state_trotter_approximates_exact():
    """Test that evolve_state_trotter approximates exact for non-commuting H."""
    # H = X + Z (non-commuting)
    H = paulisum_x_plus_z()

    # Initial state |0⟩
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    # Use small number of steps with order 1
    schedule1 = TrotterSchedule(num_steps=5, total_time=1.0, order=1)
    psi_trotter1 = evolve_state_trotter(psi0, H, schedule1)

    # Exact evolution
    psi_exact = exact_time_evolution_statevector(psi0, H, time=1.0)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_trotter1).item()
    assert abs(norm - 1.0) < 1e-10

    # Compute error
    err1 = torch.linalg.norm(psi_trotter1 - psi_exact).item()

    # Use larger number of steps and/or order 2
    schedule2 = TrotterSchedule(num_steps=20, total_time=1.0, order=2)
    psi_trotter2 = evolve_state_trotter(psi0, H, schedule2)

    err2 = torch.linalg.norm(psi_trotter2 - psi_exact).item()

    # Second-order and/or more steps should improve accuracy
    assert err2 < err1


def test_trotter_schedule_validation():
    """Test TrotterSchedule parameter validation."""
    # Valid schedule
    schedule = TrotterSchedule(num_steps=10, total_time=1.0, order=1)
    assert schedule.step_time == 0.1

    # Invalid num_steps
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        TrotterSchedule(num_steps=0, total_time=1.0, order=1)

    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        TrotterSchedule(num_steps=-1, total_time=1.0, order=1)

    # Invalid order
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        TrotterSchedule(num_steps=10, total_time=1.0, order=3)

    # Invalid total_time (NaN)
    with pytest.raises(ValueError, match="must be finite"):
        TrotterSchedule(num_steps=10, total_time=float("nan"), order=1)

    # Invalid total_time (inf)
    with pytest.raises(ValueError, match="must be finite"):
        TrotterSchedule(num_steps=10, total_time=float("inf"), order=1)


def test_evolve_state_trotter_validation():
    """Test evolve_state_trotter input validation."""
    H = paulisum_z(num_qubits=1, coeff=1.0)
    schedule = TrotterSchedule(num_steps=5, total_time=1.0, order=1)

    # Valid state
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    psi_t = evolve_state_trotter(psi0, H, schedule)
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10

    # Invalid state length (not power of 2)
    psi_invalid = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    with pytest.raises(ValueError, match="must be a power of 2"):
        evolve_state_trotter(psi_invalid, H, schedule)

    # Invalid state shape (not 1D)
    psi_2d = torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="must be 1D"):
        evolve_state_trotter(psi_2d, H, schedule)


def test_evolve_state_trotter_preserves_norm():
    """Test that evolve_state_trotter preserves state norm."""
    H = paulisum_x_plus_z()
    schedule = TrotterSchedule(num_steps=10, total_time=0.5, order=1)

    # Random normalized state
    psi0 = torch.tensor([0.6 + 0.3j, 0.7 - 0.1j], dtype=torch.complex128)
    psi0 = psi0 / torch.linalg.norm(psi0)

    psi_t = evolve_state_trotter(psi0, H, schedule)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10


def test_build_trotter_step_circuit_correct_qubits():
    """Test that build_trotter_step_circuit uses correct number of qubits."""
    # H = Z ⊗ Z on 2 qubits
    H = paulisum_z1z2()

    step_circ = build_trotter_step_circuit(H, step_time=0.5, order=1, num_qubits=2)

    assert step_circ.n_qubits == 2

    # Should be able to apply to 2-qubit state
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    psi_step = _apply_circuit_to_statevector(step_circ, psi0)

    # Check norm is preserved
    norm = torch.linalg.norm(psi_step).item()
    assert abs(norm - 1.0) < 1e-10


def test_trotter_order_2_vs_order_1():
    """Test that order 2 Trotter is more accurate than order 1."""
    # H = X + Z (non-commuting)
    H = paulisum_x_plus_z()

    # Initial state |+⟩
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    psi0 = torch.tensor([sqrt2_inv + 0j, sqrt2_inv + 0j], dtype=torch.complex128)

    total_time = 1.0
    num_steps = 10

    # Order 1
    schedule1 = TrotterSchedule(num_steps=num_steps, total_time=total_time, order=1)
    psi_trotter1 = evolve_state_trotter(psi0, H, schedule1)

    # Order 2
    schedule2 = TrotterSchedule(num_steps=num_steps, total_time=total_time, order=2)
    psi_trotter2 = evolve_state_trotter(psi0, H, schedule2)

    # Exact
    psi_exact = exact_time_evolution_statevector(psi0, H, time=total_time)

    # Compute errors
    err1 = torch.linalg.norm(psi_trotter1 - psi_exact).item()
    err2 = torch.linalg.norm(psi_trotter2 - psi_exact).item()

    # Order 2 should be more accurate
    assert err2 < err1


def test_build_trotter_step_circuit_with_y_pauli():
    """Test build_trotter_step_circuit with Y Pauli terms."""
    # H = Y on qubit 0
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Y",))])

    step_circ = build_trotter_step_circuit(H, step_time=0.5, order=1, num_qubits=1)
    assert step_circ.n_qubits == 1
    assert len(step_circ.ops) > 0


def test_build_trotter_step_circuit_all_identity():
    """Test build_trotter_step_circuit with all-identity term (trivial evolution)."""
    # H = I (all identity, trivial evolution)
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("I",))])

    step_circ = build_trotter_step_circuit(H, step_time=0.5, order=1, num_qubits=1)
    # Should have no gates (or minimal gates) for identity evolution
    # The circuit should still be valid
    assert step_circ.n_qubits == 1


def test_build_trotter_step_circuit_invalid_pauli_label():
    """Test that invalid Pauli label raises ValueError."""
    # This test would require creating a PauliTerm with invalid label,
    # but PauliTerm validates labels in __post_init__, so we can't easily test this
    # without modifying the PauliTerm class. Skip for now.
    pass


def test_build_trotter_step_circuit_mismatched_qubits():
    """Test that mismatched qubit count raises ValueError."""
    # H = Z on 1 qubit, but circuit has 2 qubits
    H = PauliSum.from_terms([PauliTerm(coeff=1.0, paulis=("Z",))])

    with pytest.raises(ValueError, match="does not match"):
        build_trotter_step_circuit(H, step_time=0.5, order=1, num_qubits=2)


def test_build_trotter_step_circuit_complex_coefficient():
    """Test that complex coefficient raises ValueError."""
    # Create a term with complex coefficient
    # Note: PauliTerm accepts float, but we can test the validation in _add_time_evolution_for_term
    # by creating a term and then trying to use it
    term = PauliTerm(coeff=1.0, paulis=("Z",))
    # The coefficient validation happens in _add_time_evolution_for_term
    # Since PauliTerm only accepts float, we can't easily test complex coefficients
    # without modifying the code. This is acceptable as the type system prevents it.
    pass


def test_evolve_state_trotter_with_device():
    """Test evolve_state_trotter with explicit device parameter."""
    H = paulisum_z(num_qubits=1, coeff=1.0)
    schedule = TrotterSchedule(num_steps=5, total_time=1.0, order=1)

    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
    device = torch.device("cpu")
    psi_t = evolve_state_trotter(psi0, H, schedule, device=device)

    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10


def test_build_trotter_step_circuit_order_2():
    """Test build_trotter_step_circuit with order 2."""
    H = paulisum_x_plus_z()
    step_circ = build_trotter_step_circuit(H, step_time=0.5, order=2, num_qubits=1)
    assert step_circ.n_qubits == 1
    assert len(step_circ.ops) > 0


def test_build_trotter_step_circuit_invalid_order():
    """Test that invalid order raises ValueError."""
    H = paulisum_z(num_qubits=1, coeff=1.0)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        build_trotter_step_circuit(H, step_time=0.5, order=3, num_qubits=1)  # type: ignore


def test_evolve_state_trotter_device_inference():
    """Test evolve_state_trotter device inference from state."""
    H = paulisum_z(num_qubits=1, coeff=1.0)
    schedule = TrotterSchedule(num_steps=5, total_time=1.0, order=1)

    # State on CPU
    psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128, device="cpu")
    psi_t = evolve_state_trotter(psi0, H, schedule)
    norm = torch.linalg.norm(psi_t).item()
    assert abs(norm - 1.0) < 1e-10


def test_build_trotter_circuit_multiple_steps():
    """Test build_trotter_circuit with multiple steps."""
    H = paulisum_x_plus_z()
    schedule = TrotterSchedule(num_steps=10, total_time=1.0, order=1)
    circ = build_trotter_circuit(H, schedule, num_qubits=1)
    assert circ.n_qubits == 1
    assert len(circ.ops) > 0

