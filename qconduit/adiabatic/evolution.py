"""Adiabatic evolution functions for discrete adiabatic quantum computing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from qconduit.circuit import QuantumCircuit
from qconduit.core.device import Device, default_device
from qconduit.operators import PauliSum, PauliTerm
from qconduit.time_evolution import build_trotter_circuit, time_evolve_state


def interpolate_paulisum(
    h_initial: PauliSum,
    h_final: PauliSum,
    s: float,
) -> PauliSum:
    """
    Linearly interpolate between two PauliSum Hamiltonians:

        H(s) = (1 - s) * H_initial + s * H_final.

    The two Hamiltonians must have compatible PauliTerm structure; if the
    underlying PauliSum implementation supports arbitrary addition and
    scalar multiplication, this function will delegate to that. Otherwise,
    it will combine coefficients term-wise, assuming they share the same
    PauliTerm ordering.

    Parameters
    ----------
    h_initial:
        Initial Hamiltonian H(0).
    h_final:
        Final Hamiltonian H(1).
    s:
        Interpolation parameter in [0,1].

    Returns
    -------
    PauliSum
        Interpolated Hamiltonian H(s).

    Raises
    ------
    ValueError
        If s is outside [0,1], or if the Hamiltonians have incompatible
        PauliTerm structures (different number of terms or mismatched Pauli strings).
    """
    # Validate s
    if s < -1e-10 or s > 1.0 + 1e-10:
        raise ValueError(f"Interpolation parameter s must be in [0,1], got {s}.")

    s = max(0.0, min(1.0, float(s)))  # Clamp to [0,1]

    # Try idiomatic arithmetic first (in case it's added in the future)
    try:
        result = (1.0 - s) * h_initial + s * h_final  # type: ignore[operator]
        if isinstance(result, PauliSum):
            return result
    except (TypeError, AttributeError):
        # Fall back to manual combination
        pass

    # Manual fallback: combine all terms by matching Pauli strings
    # Validate n_qubits match
    if h_initial.n_qubits() != h_final.n_qubits():
        raise ValueError(
            f"h_initial and h_final must act on the same number of qubits. "
            f"Got {h_initial.n_qubits()} and {h_final.n_qubits()} qubits respectively."
        )

    # Build a dictionary mapping Pauli strings to coefficients
    coeff_map: dict[tuple[str, ...], float] = {}

    # Add terms from h_initial with weight (1-s)
    for term in h_initial.terms:
        if term.paulis in coeff_map:
            coeff_map[term.paulis] += (1.0 - s) * term.coeff
        else:
            coeff_map[term.paulis] = (1.0 - s) * term.coeff

    # Add terms from h_final with weight s
    for term in h_final.terms:
        if term.paulis in coeff_map:
            coeff_map[term.paulis] += s * term.coeff
        else:
            coeff_map[term.paulis] = s * term.coeff

    # Build combined terms, dropping those with zero coefficient
    combined_terms: list[PauliTerm] = []
    for paulis, coeff in coeff_map.items():
        if abs(coeff) > 1e-12:  # Drop near-zero terms
            combined_terms.append(PauliTerm(coeff=coeff, paulis=paulis))

    return PauliSum(terms=combined_terms)


@dataclass(frozen=True)
class AdiabaticConfig:
    """
    Configuration for discrete adiabatic evolution between two PauliSum
    Hamiltonians H_initial and H_final.

    The continuous-time Hamiltonian is:

        H(s) = (1 - s) H_initial + s H_final,   s ∈ [0,1],

    and we discretize s(t) via a schedule on num_steps points.
    """

    total_time: float
    num_steps: int
    schedule: torch.Tensor
    trotter_steps_per_interval: int = 1

    def __post_init__(self) -> None:
        """Validate AdiabaticConfig invariants."""
        if self.total_time <= 0:
            raise ValueError(f"total_time must be positive, got {self.total_time}.")

        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}.")

        if self.schedule.ndim != 1:
            raise ValueError(
                f"schedule must be 1D tensor, got shape {self.schedule.shape}."
            )

        if self.schedule.numel() != self.num_steps:
            raise ValueError(
                f"schedule length ({self.schedule.numel()}) must match "
                f"num_steps ({self.num_steps})."
            )

        # Validate schedule values are in [0, 1]
        s_min = torch.min(self.schedule).item()
        s_max = torch.max(self.schedule).item()
        if s_min < -1e-6 or s_max > 1.0 + 1e-6:
            raise ValueError(
                f"schedule values must be in [0, 1], got range [{s_min}, {s_max}]."
            )

        if self.trotter_steps_per_interval < 1:
            raise ValueError(
                f"trotter_steps_per_interval must be >= 1, got {self.trotter_steps_per_interval}."
            )


def adiabatic_evolve_state(
    initial_state: torch.Tensor,
    h_initial: PauliSum,
    h_final: PauliSum,
    config: AdiabaticConfig,
    device: Device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Perform discrete adiabatic evolution of a statevector under the
    interpolating Hamiltonian H(s) = (1 - s) H_initial + s H_final.

    The evolution is approximated via piecewise-constant Hamiltonians:

        for k = 0 .. num_steps-1:
            s_k = config.schedule[k]
            H_k = interpolate_paulisum(H_initial, H_final, s_k)
            |ψ_{k+1}> ≈ exp(-i Δt H_k) |ψ_k>,

    where Δt = total_time / num_steps and exp(-i Δt H_k) is implemented
    via existing PauliSum time-evolution (e.g., Trotter decomposition).

    This is a piecewise-constant adiabatic approximation, not an exact
    continuous-time solution.

    Parameters
    ----------
    initial_state:
        Complex statevector of shape (2**n_qubits,) normalized to 1.
    h_initial:
        PauliSum Hamiltonian at s = 0.
    h_final:
        PauliSum Hamiltonian at s = 1.
    config:
        AdiabaticConfig specifying total time, schedule, and discretization.
    device:
        Optional device. Defaults to `default_device()`.
    dtype:
        Optional dtype for the internal state. If None, uses initial_state.dtype.

    Returns
    -------
    torch.Tensor
        Final statevector |ψ_final> of shape (2**n_qubits,).

    Raises
    ------
    ValueError
        If initial_state is not 1D, if its length is not a power of 2, or if
        the Hamiltonians have incompatible structures.
    """
    # Validate state dimension
    if initial_state.ndim != 1:
        raise ValueError(
            f"initial_state must be 1D, got shape {initial_state.shape}."
        )

    dim = initial_state.shape[0]
    # Check if dim is a power of 2
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(
            f"initial_state length must be a power of 2, got {dim}."
        )

    # Device & dtype
    if device is None:
        dev = default_device()
    else:
        dev = device

    if dtype is None:
        dtype = initial_state.dtype

    # Move/cast state
    state = initial_state.to(device=dev.as_torch_device(), dtype=dtype)

    # Validate Hamiltonian dimensions
    if h_initial.n_qubits() != n_qubits:
        raise ValueError(
            f"h_initial acts on {h_initial.n_qubits()} qubits, "
            f"but initial_state corresponds to {n_qubits} qubits."
        )
    if h_final.n_qubits() != n_qubits:
        raise ValueError(
            f"h_final acts on {h_final.n_qubits()} qubits, "
            f"but initial_state corresponds to {n_qubits} qubits."
        )

    # Time step
    dt = config.total_time / float(config.num_steps)

    # Iterate over schedule
    for k in range(config.num_steps):
        s_k = float(config.schedule[k].item())
        H_k = interpolate_paulisum(h_initial, h_final, s_k)  # noqa: N806

        # Evolve state using time_evolve_state
        state = time_evolve_state(
            state=state,
            hamiltonian=H_k,
            t=dt,
            n_steps=config.trotter_steps_per_interval,
            n_qubits=n_qubits,
            order=1,
            device=dev,
        )

    return state


def build_adiabatic_circuit(
    n_qubits: int,
    h_initial: PauliSum,
    h_final: PauliSum,
    config: AdiabaticConfig,
) -> QuantumCircuit:
    """
    Build a QuantumCircuit implementing a discretized adiabatic evolution
    from H_initial to H_final using PauliSum Trotterization.

    At each step k, the circuit appends a Trotterized unitary approximating:

        U_k ≈ exp(-i Δt H(s_k)),

    with Δt = total_time / num_steps and s_k = config.schedule[k].

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    h_initial:
        PauliSum for H(0).
    h_final:
        PauliSum for H(1).
    config:
        AdiabaticConfig describing the schedule and Trotterization.

    Returns
    -------
    QuantumCircuit
        Circuit that, when applied to an initial state |ψ_0>, approximates the
        adiabatic evolution to |ψ_final>.

    Raises
    ------
    ValueError
        If n_qubits < 1, or if the Hamiltonians act on a different number of qubits.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")

    if h_initial.n_qubits() != n_qubits:
        raise ValueError(
            f"h_initial acts on {h_initial.n_qubits()} qubits, "
            f"but n_qubits={n_qubits}."
        )
    if h_final.n_qubits() != n_qubits:
        raise ValueError(
            f"h_final acts on {h_final.n_qubits()} qubits, "
            f"but n_qubits={n_qubits}."
        )

    circuit = QuantumCircuit(n_qubits=n_qubits)
    dt = config.total_time / float(config.num_steps)

    # Loop over schedule steps
    for k in range(config.num_steps):
        s_k = float(config.schedule[k].item())
        H_k = interpolate_paulisum(h_initial, h_final, s_k)  # noqa: N806

        # Build Trotter circuit for this step
        step_circuit = build_trotter_circuit(
            hamiltonian=H_k,
            t=dt,
            n_steps=config.trotter_steps_per_interval,
            n_qubits=n_qubits,
            order=1,
        )

        # Append gates from step_circuit to main circuit
        for op in step_circuit.ops:
            circuit.add_gate(
                op.name,
                list(op.qubits),
                params=list(op.params) if op.params is not None else None,
            )

    return circuit


def build_x_mixer_hamiltonian(num_qubits: int, strength: float = 1.0) -> PauliSum:
    """
    Construct the standard transverse-field X mixer Hamiltonian:

        H_mixer = -strength * sum_i X_i

    on `num_qubits` qubits.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    strength:
        Strength of the mixer field. Defaults to 1.0.

    Returns
    -------
    PauliSum
        The X mixer Hamiltonian.

    Raises
    ------
    ValueError
        If num_qubits < 1.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}.")

    terms: list[PauliTerm] = []
    for i in range(num_qubits):
        # Create a PauliTerm with X on qubit i and I on all others
        paulis = ["I"] * num_qubits
        paulis[i] = "X"
        term = PauliTerm(coeff=-strength, paulis=tuple(paulis))
        terms.append(term)

    return PauliSum(terms=terms)


def adiabatic_x_mixer_to_problem_state(
    initial_state: torch.Tensor,
    h_problem: PauliSum,
    config: AdiabaticConfig,
    device: Device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Convenience wrapper: evolve from the standard X mixer Hamiltonian to a
    user-specified problem Hamiltonian H_problem.

    H_initial = H_mixer, H_final = H_problem.

    Parameters
    ----------
    initial_state:
        Initial statevector.
    h_problem:
        Problem Hamiltonian.
    config:
        AdiabaticConfig.
    device, dtype:
        As in `adiabatic_evolve_state`.

    Returns
    -------
    torch.Tensor
        Final statevector.

    Raises
    ------
    ValueError
        If initial_state length is not a power of 2, or if h_problem acts on
        a different number of qubits than inferred from initial_state.
    """
    # Infer n_qubits from initial_state
    if initial_state.ndim != 1:
        raise ValueError(
            f"initial_state must be 1D, got shape {initial_state.shape}."
        )

    dim = initial_state.shape[0]
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(
            f"initial_state length must be a power of 2, got {dim}."
        )

    # Build mixer Hamiltonian
    h_mixer = build_x_mixer_hamiltonian(n_qubits)

    # Call adiabatic_evolve_state
    return adiabatic_evolve_state(
        initial_state=initial_state,
        h_initial=h_mixer,
        h_final=h_problem,
        config=config,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "interpolate_paulisum",
    "AdiabaticConfig",
    "adiabatic_evolve_state",
    "build_adiabatic_circuit",
    "build_x_mixer_hamiltonian",
    "adiabatic_x_mixer_to_problem_state",
]

