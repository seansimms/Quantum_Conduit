"""Quantum Conduit - a PyTorch-native quantum statevector plumbing library."""

__version__ = "0.0.4"

# Core abstractions
# Algorithms
# Adiabatic evolution
from .adiabatic import (
    AdiabaticConfig,
    ScheduleFn,
    adiabatic_evolve_state,
    adiabatic_x_mixer_to_problem_state,
    build_adiabatic_circuit,
    build_x_mixer_hamiltonian,
    interpolate_paulisum,
    linear_schedule,
    polynomial_schedule,
    sample_schedule,
)
from .algorithms import VQE, Edge, QAOAAnsatz, ising_maxcut_hamiltonian

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

# Convex optimization (G26: Convex Optimization Textbook Library)
from .convex import (
    LPProblem,
    OptimizeResult,
    QPProblem,
    Status,
    active_set_qp,
    is_kkt_optimal,
    kkt_residuals,
    log_barrier_method,
    projected_gradient,
    projected_newton,
    simplex,
)
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

# Deterministic Signal Processing (G30: Deterministic Signal Processing Suite)
from .dsp import (
    STFT,
    apply_fir,
    blackman,
    butterworth,
    cheby1,
    check_1d_array,
    convolve,
    correlate,
    design_frequency_grid,
    fft_convolve,
    filtfilt,
    fir_window_design,
    freqz,
    hamming,
    hann,
    istft,
    lfilter,
    next_pow2,
    overlap_add_filter,
    overlap_save_filter,
    rectangular,
    sosfilt,
    sosfiltfilt,
    spectrogram,
    stft,
    tf2sos,
)

# Evolution (G15: Hamiltonian Time Evolution & Trotterization)
from .evolution import (
    TrotterOrder,
    TrotterSchedule,
    evolve_state_trotter,
    exact_time_evolution_statevector,
)

# Exact solvers
from .exact import (
    exact_eigensystem,
    exact_ground_state,
    paulisum_to_dense,
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

# Feature engineering transforms (G25)
from .features import (
    PCA,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
    Transformer,
)

# Fermion-to-qubit mappings
from .fermion import (
    FermionOperator,
    FermionOpSymbol,
    FermionTerm,
    bravyi_kitaev,
    jordan_wigner,
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

# Graph algorithms (G27: Graph Algorithms Textbook Classics)
from .graphs import (
    Graph,
    WeightedGraph,
    bellman_ford,
    bfs,
    dfs_iterative,
    dfs_recursive,
    dijkstra,
    edges_from_weighted_graph,
    floyd_warshall,
    graph_laplacian_matrix,
    kruskal_mst,
    node_index_map,
    pagerank,
    prim_mst,
    reconstruct_path,
    spectral_clustering,
)

# Layers
from .layers import HardwareEfficientAnsatz, ParametricAnsatz, QuantumBlock

# Measurement, sampling, and tomography
from .measurement import (
    basis_probabilities_from_statevector,
    bitstring_counts,
    empirical_probabilities_from_bitstrings,
    estimate_pauli_z_expectation_from_samples,
    pauli_expectation_from_statevector,
    pauli_matrix_from_label,
    reconstruct_single_qubit_density_from_pauli,
    reconstruct_two_qubit_density_from_pauli,
    sample_bitstrings_from_probabilities,
    sample_bitstrings_from_statevector,
    single_qubit_pauli_expectations_from_statevector,
    two_qubit_pauli_expectations_from_statevector,
)

# Models
from .models import (
    diagonal_z_field,
    heisenberg_xxz_chain,
    ising_zz_chain,
    transverse_field_ising_chain,
    two_qubit_generic_chemistry_like,
)

# Noise models
from .noise import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    # New G11 APIs
    KrausChannel,
    NoiseConfig,
    NoiseModel,
    PhaseDampingChannel,
    SingleQubitChannel,
    amplitude_damping_channel,
    apply_kraus_channel_to_density_matrix,
    apply_kraus_channel_to_statevector,
    bit_flip_channel,
    bit_phase_flip_channel,
    compose_kraus_channels,
    depolarizing_channel,
    generalized_amplitude_damping_channel,
    identity_channel,
    phase_damping_channel,
    phase_flip_channel,
    sample_noisy_circuit_dm,
    simulate_noisy_circuit_dm,
    to_density_matrix,
    two_qubit_depolarizing_channel,
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

# Probabilistic inference (G28: Textbook Probabilistic Inference Tools)
from .probabilistic import (
    GaussianMixture,
    HiddenMarkovModel,
    StateSpaceModel,
    bootstrap_particle_filter,
    effective_sample_size,
    log_normal_pdf,
    logsumexp,
    normal_pdf,
    normalize_log_weights,
    systematic_resample,
)

# Reinforcement Learning (G29: Classic Reinforcement Learning Algorithms)
from .rl import (
    Bandit,
    ChainMDP,
    Env,
    GridWorldTiny,
    Policy,
    TabularAgent,
    TabularPolicy,
    compute_advantage,
    discounted_returns,
    epsilon_greedy_action,
    evaluate_policy,
    greedy_policy_from_Q,
    mc_control_on_policy,
    q_learning,
    reinforce,
    reinforce_with_baseline,
    sarsa,
    seed_rng,
    softmax,
    td_lambda,
)

# Sampling
from .sampling import (
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
    build_trotter_circuit,
    build_trotter_step_circuit,
    time_evolve_state,
    trotter_step_pauli_sum,
)

# Time-series forecasting (G24: Textbook Time-Series Forecasting Models)
from .timeseries import (
    AR,
    ARIMA,
    ARMA,
    MA,
    SARIMA,
    FitResult,
    StateSpace,
    kalman_filter,
    kalman_predict,
    kalman_smoother,
)

# Training
from .training import (
    EarlyStoppingConfig,
    TrainingCallback,
    TrainingHistory,
    TrainingStepInfo,
    VQETrainer,
)

# Transpilation and decomposition
from .transpile import (
    GateCountSummary,
    decompose_gate_to_basis,
    decompose_h_to_rz_rx_rz,
    decompose_rz_to_clifford_t,
    decompose_x_to_rx,
    decompose_y_to_ry,
    decompose_z_to_rz,
    estimate_circuit_depth,
    summarize_gate_counts,
    transpile_to_basis,
    transpile_to_clifford_t,
    transpile_to_rx_rz_cx_basis,
)

# Variational algorithms
from .variational import (
    LayeredEntanglerAnsatz,
    QAOAResult,
    VariationalAnsatz,
    VQEResult,
    evaluate_expectation_value,
    run_qaoa,
    run_vqe,
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
    # Feature engineering
    "Transformer",
    "StandardScaler",
    "RobustScaler",
    "MinMaxScaler",
    "PCA",
    "PolynomialFeatures",
    "OneHotEncoder",
    "TargetEncoder",
    "KBinsDiscretizer",
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
    # Time-series models (G24)
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    "SARIMA",
    "FitResult",
    "StateSpace",
    "kalman_filter",
    "kalman_smoother",
    "kalman_predict",
    # Convex optimization (G26)
    "Status",
    "OptimizeResult",
    "LPProblem",
    "QPProblem",
    "simplex",
    "active_set_qp",
    "log_barrier_method",
    "projected_gradient",
    "projected_newton",
    "kkt_residuals",
    "is_kkt_optimal",
    # Graph algorithms (G27)
    "Graph",
    "WeightedGraph",
    "bfs",
    "dfs_recursive",
    "dfs_iterative",
    "dijkstra",
    "bellman_ford",
    "floyd_warshall",
    "kruskal_mst",
    "prim_mst",
    "pagerank",
    "graph_laplacian_matrix",
    "spectral_clustering",
    "node_index_map",
    "edges_from_weighted_graph",
    "reconstruct_path",
    # Probabilistic inference (G28)
    "HiddenMarkovModel",
    "GaussianMixture",
    "StateSpaceModel",
    "bootstrap_particle_filter",
    "logsumexp",
    "normal_pdf",
    "log_normal_pdf",
    "systematic_resample",
    "effective_sample_size",
    "normalize_log_weights",
    # Reinforcement Learning (G29)
    "Env",
    "ChainMDP",
    "GridWorldTiny",
    "Bandit",
    "TabularAgent",
    "Policy",
    "TabularPolicy",
    "mc_control_on_policy",
    "q_learning",
    "sarsa",
    "td_lambda",
    "reinforce",
    "reinforce_with_baseline",
    "seed_rng",
    "epsilon_greedy_action",
    "softmax",
    "discounted_returns",
    "compute_advantage",
    "evaluate_policy",
    "greedy_policy_from_Q",
    # Deterministic Signal Processing (G30)
    "STFT",
    "check_1d_array",
    "next_pow2",
    "freqz",
    "design_frequency_grid",
    "hann",
    "hamming",
    "blackman",
    "rectangular",
    "convolve",
    "correlate",
    "fft_convolve",
    "overlap_add_filter",
    "overlap_save_filter",
    "stft",
    "istft",
    "spectrogram",
    "fir_window_design",
    "apply_fir",
    "butterworth",
    "cheby1",
    "tf2sos",
    "sosfilt",
    "lfilter",
    "filtfilt",
    "sosfiltfilt",
]
