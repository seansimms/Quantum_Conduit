"""Quantum Conduit - a PyTorch-native quantum statevector plumbing library."""

__version__ = "0.0.4"

# Core abstractions
# Algorithms
from .algorithms import Edge, ising_maxcut_hamiltonian, QAOAAnsatz, VQE

# Backend operations
from .backend import (
    apply_gate,
    apply_kraus_single_qubit,
    apply_two_qubit_gate,
    dm_from_statevector,
    measure_expectation_z,
    measure_expectation_z_dm,
    measure_probs,
    measure_probs_dm,
    zero_dm_state,
    zero_state,
)

# Circuit IR
from .circuit import GateOp, QuantumCircuit
from .core import Device, QuantumModule, default_device, device

# Diagnostics
from .diagnostics import (
    assert_hermitian,
    assert_normalized,
    bloch_vector,
    debug_context,
    fidelity,
    is_debug_enabled,
    is_hermitian,
    set_debug_enabled,
    state_norm,
)

# Gates
from .gates import (
    CNOT,
    RX,
    RY,
    RZ,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    is_unitary,
)

# Gradients
from .grad import (
    ParameterShiftRule,
    autograd_gradient,
    param_shift_energy,
    parameter_shift_gradient,
    parameter_shift_single,
    vqe_parameter_shift_gradient,
)

# Layers
from .layers import HardwareEfficientAnsatz, ParametricAnsatz, QuantumBlock

# Noise models
from .noise import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    NoiseModel,
    PhaseDampingChannel,
    SingleQubitChannel,
    depolarizing_channel,
    phase_damping_channel,
    amplitude_damping_channel,
    identity_channel,
    NoiseConfig,
    simulate_noisy_circuit_dm,
    sample_noisy_circuit_dm,
    # New G11 APIs
    KrausChannel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    generalized_amplitude_damping_channel,
    two_qubit_depolarizing_channel,
    to_density_matrix,
    apply_kraus_channel_to_density_matrix,
    apply_kraus_channel_to_statevector,
    compose_kraus_channels,
)

# Operators
from .operators import (
    PauliSum,
    PauliTerm,
    expectation_pauli_sum,
    expectation_pauli_sum_dm,
    expectation_pauli_term,
)

# Optimizers
from .optim import OptimConfig, create_optimizer

# Training
from .training import (
    EarlyStoppingConfig,
    TrainingCallback,
    TrainingHistory,
    TrainingStepInfo,
    VQETrainer,
)

# Sampling
from .sampling import (
    bitstring_counts,
    counts_to_probs,
    kl_divergence,
    marginalize_probs,
    sample_bitstrings_circuit,
    sample_bitstrings_dm,
    sample_bitstrings_state,
    sample_from_probs,
)

# Time evolution
from .time_evolution import (
    OrderLiteral,
    trotter_step_pauli_sum,
    time_evolve_state,
    build_trotter_step_circuit,
    build_trotter_circuit,
)

# Evolution (G15: Hamiltonian Time Evolution & Trotterization)
from .evolution import (
    exact_time_evolution_statevector,
    TrotterOrder,
    TrotterSchedule,
    evolve_state_trotter,
)

# Experiments
from .experiments import (
    SweepResult1D,
    SweepResult2D,
    run_1d_sweep,
    run_2d_sweep,
    sweep_vqe_1d,
    sweep_vqe_2d,
)

# Exact solvers
from .exact import (
    paulisum_to_dense,
    exact_eigensystem,
    exact_ground_state,
)

# Models
from .models import (
    transverse_field_ising_chain,
    heisenberg_xxz_chain,
    ising_zz_chain,
    two_qubit_generic_chemistry_like,
    diagonal_z_field,
)

# Adiabatic evolution
from .adiabatic import (
    ScheduleFn,
    linear_schedule,
    polynomial_schedule,
    sample_schedule,
    AdiabaticConfig,
    interpolate_paulisum,
    adiabatic_evolve_state,
    build_adiabatic_circuit,
    build_x_mixer_hamiltonian,
    adiabatic_x_mixer_to_problem_state,
)

# Fermion-to-qubit mappings
from .fermion import (
    FermionOpSymbol,
    FermionTerm,
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
)

# Measurement, sampling, and tomography
from .measurement import (
    basis_probabilities_from_statevector,
    sample_bitstrings_from_probabilities,
    sample_bitstrings_from_statevector,
    bitstring_counts,
    empirical_probabilities_from_bitstrings,
    estimate_pauli_z_expectation_from_samples,
    pauli_matrix_from_label,
    pauli_expectation_from_statevector,
    single_qubit_pauli_expectations_from_statevector,
    reconstruct_single_qubit_density_from_pauli,
    two_qubit_pauli_expectations_from_statevector,
    reconstruct_two_qubit_density_from_pauli,
)

# Variational algorithms
from .variational import (
    VariationalAnsatz,
    HardwareEfficientAnsatz,
    LayeredEntanglerAnsatz,
    QAOAAnsatz,
    VQEResult,
    evaluate_expectation_value,
    run_vqe,
    QAOAResult,
    run_qaoa,
)

# Transpilation and decomposition
from .transpile import (
    decompose_h_to_rz_rx_rz,
    decompose_x_to_rx,
    decompose_y_to_ry,
    decompose_z_to_rz,
    decompose_rz_to_clifford_t,
    decompose_gate_to_basis,
    transpile_to_basis,
    transpile_to_rx_rz_cx_basis,
    transpile_to_clifford_t,
    GateCountSummary,
    summarize_gate_counts,
    estimate_circuit_depth,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Device",
    "device",
    "default_device",
    "QuantumModule",
    # Gates
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "CNOT",
    "RX",
    "RY",
    "RZ",
    "is_unitary",
    # Backend
    "zero_state",
    "apply_gate",
    "apply_two_qubit_gate",
    "measure_expectation_z",
    "measure_probs",
    "zero_dm_state",
    "dm_from_statevector",
    "apply_kraus_single_qubit",
    "measure_probs_dm",
    "measure_expectation_z_dm",
    # Operators
    "PauliTerm",
    "PauliSum",
    "expectation_pauli_term",
    "expectation_pauli_sum",
    "expectation_pauli_sum_dm",
    # Gradients
    "param_shift_energy",
    "ParameterShiftRule",
    "parameter_shift_single",
    "parameter_shift_gradient",
    "autograd_gradient",
    "vqe_parameter_shift_gradient",
    # Noise
    "NoiseModel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    "SingleQubitChannel",
    "depolarizing_channel",
    "phase_damping_channel",
    "amplitude_damping_channel",
    "identity_channel",
    "NoiseConfig",
    "simulate_noisy_circuit_dm",
    "sample_noisy_circuit_dm",
    # New G11 APIs
    "KrausChannel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "generalized_amplitude_damping_channel",
    "two_qubit_depolarizing_channel",
    "to_density_matrix",
    "apply_kraus_channel_to_density_matrix",
    "apply_kraus_channel_to_statevector",
    "compose_kraus_channels",
    # Circuit IR
    "GateOp",
    "QuantumCircuit",
    # Layers
    "ParametricAnsatz",
    "HardwareEfficientAnsatz",
    "QuantumBlock",
    # Algorithms
    "Edge",
    "ising_maxcut_hamiltonian",
    "QAOAAnsatz",
    "VQE",
    # Diagnostics
    "state_norm",
    "assert_normalized",
    "is_hermitian",
    "assert_hermitian",
    "fidelity",
    "bloch_vector",
    "is_debug_enabled",
    "set_debug_enabled",
    "debug_context",
    # Optimizers
    "OptimConfig",
    "create_optimizer",
    # Training
    "TrainingStepInfo",
    "TrainingHistory",
    "TrainingCallback",
    "EarlyStoppingConfig",
    "VQETrainer",
    # Sampling
    "sample_from_probs",
    "sample_bitstrings_state",
    "sample_bitstrings_dm",
    "sample_bitstrings_circuit",
    "bitstring_counts",
    "counts_to_probs",
    "kl_divergence",
    "marginalize_probs",
    # Time evolution
    "OrderLiteral",
    "trotter_step_pauli_sum",
    "time_evolve_state",
    "build_trotter_step_circuit",
    "build_trotter_circuit",
    # Evolution (G15: Hamiltonian Time Evolution & Trotterization)
    "exact_time_evolution_statevector",
    "TrotterOrder",
    "TrotterSchedule",
    "evolve_state_trotter",
    # Experiments
    "SweepResult1D",
    "SweepResult2D",
    "run_1d_sweep",
    "run_2d_sweep",
    "sweep_vqe_1d",
    "sweep_vqe_2d",
    # Exact solvers
    "paulisum_to_dense",
    "exact_eigensystem",
    "exact_ground_state",
    # Models
    "transverse_field_ising_chain",
    "heisenberg_xxz_chain",
    "ising_zz_chain",
    "two_qubit_generic_chemistry_like",
    "diagonal_z_field",
    # Adiabatic evolution
    "ScheduleFn",
    "linear_schedule",
    "polynomial_schedule",
    "sample_schedule",
    "AdiabaticConfig",
    "interpolate_paulisum",
    "adiabatic_evolve_state",
    "build_adiabatic_circuit",
    "build_x_mixer_hamiltonian",
    "adiabatic_x_mixer_to_problem_state",
    # Fermion-to-qubit mappings
    "FermionOpSymbol",
    "FermionTerm",
    "FermionOperator",
    "jordan_wigner",
    "bravyi_kitaev",
    # Measurement, sampling, and tomography
    "basis_probabilities_from_statevector",
    "sample_bitstrings_from_probabilities",
    "sample_bitstrings_from_statevector",
    "bitstring_counts",
    "empirical_probabilities_from_bitstrings",
    "estimate_pauli_z_expectation_from_samples",
    "pauli_matrix_from_label",
    "pauli_expectation_from_statevector",
    "single_qubit_pauli_expectations_from_statevector",
    "reconstruct_single_qubit_density_from_pauli",
    "two_qubit_pauli_expectations_from_statevector",
    "reconstruct_two_qubit_density_from_pauli",
    # Variational algorithms
    "VariationalAnsatz",
    "HardwareEfficientAnsatz",
    "LayeredEntanglerAnsatz",
    "QAOAAnsatz",
    "VQEResult",
    "evaluate_expectation_value",
    "run_vqe",
    "QAOAResult",
    "run_qaoa",
    # Transpilation and decomposition
    "decompose_h_to_rz_rx_rz",
    "decompose_x_to_rx",
    "decompose_y_to_ry",
    "decompose_y_to_rz_rx_rz",
    "decompose_z_to_rz",
    "decompose_rz_to_clifford_t",
    "decompose_gate_to_basis",
    "transpile_to_basis",
    "transpile_to_rx_rz_cx_basis",
    "transpile_to_clifford_t",
    "GateCountSummary",
    "summarize_gate_counts",
    "estimate_circuit_depth",
]
