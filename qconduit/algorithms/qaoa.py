"""QAOA (Quantum Approximate Optimization Algorithm) for Ising/MaxCut problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from qconduit.backend.statevector import apply_gate, apply_two_qubit_gate, zero_state
from qconduit.circuit import QuantumCircuit
from qconduit.gates.standard import CNOT, H, RX, RZ
from qconduit.layers.ansatzes import ParametricAnsatz
from qconduit.operators import PauliTerm, PauliSum


@dataclass(frozen=True)
class Edge:
    """Simple data structure for an undirected, weighted edge.

    Parameters
    ----------
    u:
        One endpoint of the edge (0 <= u < num_nodes).
    v:
        The other endpoint of the edge (0 <= v < num_nodes).
    weight:
        Non-negative real edge weight. For unweighted graphs, this is 1.0.
    """

    u: int
    v: int
    weight: float = 1.0


def ising_maxcut_hamiltonian(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]] | Sequence[Edge],
    weights: Optional[Sequence[float]] = None,
    include_constant: bool = True,
) -> PauliSum:
    """
    Build a standard Ising Hamiltonian for MaxCut on an undirected graph.

    For a graph with vertex set {0, ..., num_nodes - 1} and edge set E,
    we construct:

        H = sum_{(i, j) in E} w_ij * (1 - Z_i Z_j) / 2

    where w_ij >= 0 is the edge weight. This Hamiltonian has larger
    expectation values for cuts that cut more (or more heavily weighted)
    edges. The constant term sum w_ij / 2 can optionally be included or
    omitted; including it makes H equal to the expected cut value, while
    omitting it simply shifts the energy by a constant.

    Parameters
    ----------
    num_nodes:
        Number of vertices in the graph.
    edges:
        Either a sequence of (u, v) pairs or a sequence of Edge objects.
        Each edge is undirected; (u, v) and (v, u) are treated the same.
    weights:
        Optional sequence of edge weights (same length as `edges`).
        If provided, `edges` must be a sequence of (u, v) pairs and
        each weight[i] corresponds to edges[i]. If edges are provided
        as Edge objects, this argument must be None.
    include_constant:
        If True, include the constant term sum w_ij / 2. If False,
        only the Z_i Z_j terms are included; the constant is effectively
        dropped, which does not affect the argmin/argmax of the energy
        but does shift its absolute value.

    Returns
    -------
    PauliSum
        The MaxCut Ising Hamiltonian as a sum of Pauli terms.

    Raises
    ------
    ValueError:
        If num_nodes <= 0, edge weights are negative, edges are out of bounds,
        or weights/edges have incompatible lengths.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    pauli_terms: List[PauliTerm] = []

    # Normalize edges and weights into a list of Edge objects.
    edge_list: List[Edge] = []
    if len(edges) > 0 and isinstance(edges[0], Edge):  # type: ignore[index]
        if weights is not None:
            raise ValueError("weights must be None when edges are Edge objects.")
        edge_list = list(edges)  # type: ignore[arg-type]
    else:
        # edges are (u, v) pairs
        if weights is None:
            weights_iter: Iterable[float] = (1.0 for _ in edges)
        else:
            if len(weights) != len(edges):
                raise ValueError("weights must have the same length as edges.")
            weights_iter = weights

        for (u, v), w in zip(edges, weights_iter):
            edge_list.append(Edge(u=int(u), v=int(v), weight=float(w)))

    # Validate edges and weights; accumulate total for constant term.
    constant_coeff = 0.0
    for e in edge_list:
        if e.weight < 0.0:
            raise ValueError("Edge weights must be non-negative.")
        if e.u < 0 or e.u >= num_nodes or e.v < 0 or e.v >= num_nodes:
            raise ValueError(
                f"Edge ({e.u}, {e.v}) is out of bounds for num_nodes={num_nodes}."
            )
        if e.u == e.v:
            # Self-loops are not meaningful for MaxCut; skip them explicitly.
            continue
        constant_coeff += e.weight / 2.0

        # Build the Z_i Z_j term with coefficient -w_ij / 2
        paulis = ["I"] * num_nodes
        paulis[e.u] = "Z"
        paulis[e.v] = "Z"
        term = PauliTerm(coeff=-e.weight / 2.0, paulis=tuple(paulis))
        pauli_terms.append(term)

    if include_constant and constant_coeff != 0.0:
        # Constant term (all I) with coefficient equal to sum w_ij / 2
        paulis = tuple("I" for _ in range(num_nodes))
        const_term = PauliTerm(coeff=constant_coeff, paulis=paulis)
        pauli_terms.append(const_term)

    return PauliSum(pauli_terms)


class QAOAAnsatz(ParametricAnsatz):
    """
    Standard QAOA ansatz for Ising / MaxCut Hamiltonians.

    This ansatz prepares the state

        |psi(γ, β)> = U_B(β_{p-1}) U_C(γ_{p-1}) ... U_B(β_0) U_C(γ_0) |+>^{⊗ n},

    where the cost Hamiltonian H_C is provided as a PauliSum that is
    diagonal in the computational basis (only 'Z' and 'I' Paulis).
    The mixer is the standard X-field mixer:

        H_B = sum_i X_i.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    problem_hamiltonian:
        PauliSum representing the cost Hamiltonian. It is expected to
        be diagonal in the computational basis (i.e., each PauliTerm
        contains only 'Z' and 'I' entries). If this is not the case,
        QAOAAnsatz will raise a ValueError when building the circuit.
    p:
        Number of QAOA layers (depth).

    Attributes
    ----------
    n_qubits:
        Number of qubits in the QAOA circuit.
    p:
        QAOA depth (number of alternating problem/mixer layers).
    num_parameters:
        Total number of scalar parameters (2 * p).
    problem_hamiltonian:
        The Ising/MaxCut cost Hamiltonian used by this ansatz.
    """

    def __init__(self, n_qubits: int, problem_hamiltonian: PauliSum, p: int) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if p <= 0:
            raise ValueError("p (number of QAOA layers) must be positive.")

        # Optionally, validate that problem_hamiltonian.n_qubits == n_qubits
        if hasattr(problem_hamiltonian, "n_qubits"):
            if problem_hamiltonian.n_qubits() != n_qubits:
                raise ValueError(
                    "problem_hamiltonian.n_qubits does not match n_qubits."
                )

        super().__init__(n_qubits=n_qubits, device=None)

        self._p = int(p)
        self._problem_hamiltonian = problem_hamiltonian

        # QAOA uses 2p parameters: gamma_0..gamma_{p-1} and beta_0..beta_{p-1}
        self._num_parameters = 2 * self._p

        # Validate Hamiltonian structure
        self._validate_hamiltonian()

    @property
    def p(self) -> int:
        """QAOA depth (number of alternating problem/mixer layers)."""
        return self._p

    @property
    def num_parameters(self) -> int:
        """Total number of scalar parameters (2 * p)."""
        return self._num_parameters

    @property
    def problem_hamiltonian(self) -> PauliSum:
        """The Ising/MaxCut cost Hamiltonian used by this ansatz."""
        return self._problem_hamiltonian

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Build the quantum state from parameters using backend operations.

        This method directly applies gates using backend operations to preserve
        gradients, unlike build_circuit which converts parameters to floats.

        Parameters
        ----------
        params:
            1D tensor of shape (2 * p,) containing [gamma_0, ..., gamma_{p-1},
            beta_0, ..., beta_{p-1}].

        Returns
        -------
        torch.Tensor
            Complex statevector tensor of shape (2**n_qubits,).
        """
        if params.ndim != 1:
            raise ValueError("QAOA parameters must be a 1D tensor.")
        if params.numel() != self._num_parameters:
            raise ValueError(
                f"Expected {self._num_parameters} parameters, got {params.numel()}."
            )

        # Ensure params is on the correct device and dtype
        params = params.to(dtype=torch.float32, device=self.qdevice.as_torch_device())

        # Extract gammas and betas
        gammas = params[: self._p]
        betas = params[self._p :]

        # Initialize state |0...0>
        state = zero_state(
            n_qubits=self.n_qubits,
            batch_shape=None,
            device=self.qdevice,
            dtype=torch.complex64,
        )

        # 1. Prepare |+>^n by applying H to each qubit.
        gate_h = H(dtype=state.dtype, device=state.device)
        for q in range(self.n_qubits):
            state = apply_gate(state, gate_h, qubit=q, n_qubits=self.n_qubits)

        # 2. Alternate problem and mixer layers.
        for layer in range(self._p):
            gamma_layer = gammas[layer]
            beta_layer = betas[layer]

            # Problem unitary U_C(gamma_layer)
            state = self._apply_problem_unitary(state, gamma_layer)

            # Mixer unitary U_B(beta_layer)
            state = self._apply_mixer_unitary(state, beta_layer)

        return state

    def build_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """
        Build a QuantumCircuit implementing the QAOA ansatz for the
        given parameter vector.

        Parameters
        ----------
        params:
            1D tensor of shape (2 * p,) containing [gamma_0, ..., gamma_{p-1},
            beta_0, ..., beta_{p-1}].

        Returns
        -------
        QuantumCircuit
            Circuit that prepares the QAOA state for the given parameters.

        Raises
        ------
        ValueError:
            If params has incorrect shape or the problem Hamiltonian contains
            non-diagonal (non-Z/I) Pauli operators.
        """
        if params.ndim != 1:
            raise ValueError("QAOA parameters must be a 1D tensor.")
        if params.numel() != self._num_parameters:
            raise ValueError(
                f"Expected {self._num_parameters} parameters, got {params.numel()}."
            )

        # Extract gammas and betas as Python floats for circuit parameters.
        gammas = [float(x) for x in params[: self._p]]
        betas = [float(x) for x in params[self._p :]]

        circuit = QuantumCircuit(n_qubits=self.n_qubits)

        # 1. Prepare |+>^n by applying H to each qubit.
        for q in range(self.n_qubits):
            circuit.add_gate("H", [q])

        # 2. Alternate problem and mixer layers.
        for layer in range(self._p):
            gamma_layer = gammas[layer]
            beta_layer = betas[layer]

            # Problem unitary U_C(gamma_layer)
            self._append_problem_unitary_layer(circuit, gamma_layer)

            # Mixer unitary U_B(beta_layer)
            self._append_mixer_unitary_layer(circuit, beta_layer)

        return circuit

    # --- Internal helpers -------------------------------------------------

    def _validate_hamiltonian(self) -> None:
        """
        Validate that the problem Hamiltonian is diagonal in Z and only contains
        1- and 2-body terms.
        """
        for term in self._problem_hamiltonian.terms:
            paulis = term.paulis
            num_qubits = len(paulis)
            if num_qubits != self.n_qubits:
                raise ValueError(
                    "All PauliTerms in problem_hamiltonian must have length "
                    f"{self.n_qubits}."
                )

            # Check that term is diagonal in Z.
            non_identity_indices: List[int] = []
            for idx, p in enumerate(paulis):
                label = p.upper()
                if label not in ("I", "Z"):
                    raise ValueError(
                        "QAOAAnsatz only supports diagonal (Z/I) Hamiltonians; "
                        f"found Pauli {label!r}."
                    )
                if label == "Z":
                    non_identity_indices.append(idx)

            # Check coefficient is real
            coeff = term.coeff
            if isinstance(coeff, complex) and abs(coeff.imag) > 1e-12:
                raise ValueError(
                    "problem_hamiltonian coefficients must be real; "
                    f"got {term.coeff}."
                )

            # Check that we only have 1- and 2-body terms (skip identity terms)
            if len(non_identity_indices) > 2:
                raise ValueError(
                    "QAOAAnsatz only supports 1- and 2-body Z terms "
                    "in the problem Hamiltonian."
                )

    def _append_problem_unitary_layer(
        self,
        circuit: QuantumCircuit,
        gamma: float,
    ) -> None:
        """
        Append the problem unitary U_C(gamma) = exp(-i * gamma * H_C)
        to the circuit. Assumes H_C is diagonal in Z.
        """
        if abs(gamma) == 0.0:
            return

        for term in self._problem_hamiltonian.terms:
            paulis = term.paulis

            # Find non-identity indices (Hamiltonian is already validated to be Z/I only)
            non_identity_indices: List[int] = []
            for idx, p in enumerate(paulis):
                if p.upper() == "Z":
                    non_identity_indices.append(idx)

            # Skip all-identity terms (global phase).
            if not non_identity_indices:
                continue

            # Get coefficient (already validated to be real)
            coeff = term.coeff
            coeff_real = float(coeff.real) if isinstance(coeff, complex) else float(coeff)

            # Effective angle for this term.
            angle = 2.0 * coeff_real * gamma
            if abs(angle) == 0.0:
                continue

            # Apply gates based on number of Z operators (already validated to be 1 or 2)
            if len(non_identity_indices) == 1:
                # Single-qubit Z term: exp(-i gamma c Z) = RZ(2 gamma c) up to phase.
                q = non_identity_indices[0]
                circuit.add_gate("RZ", [q], params=[angle])
            else:  # len(non_identity_indices) == 2
                # Two-qubit ZZ term: use CNOT ladder
                q0, q1 = non_identity_indices
                # Use (q0 -> q1) convention
                control, target = q0, q1
                circuit.add_gate("CNOT", [control, target])
                circuit.add_gate("RZ", [target], params=[angle])
                circuit.add_gate("CNOT", [control, target])

    def _apply_problem_unitary(
        self,
        state: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the problem unitary U_C(gamma) = exp(-i * gamma * H_C) to the state.
        Assumes H_C is diagonal in Z. This method returns a new state tensor
        via backend operations that preserve gradients.
        """
        for term in self._problem_hamiltonian.terms:
            paulis = term.paulis

            # Find non-identity indices
            non_identity_indices: List[int] = []
            for idx, p in enumerate(paulis):
                if p.upper() == "Z":
                    non_identity_indices.append(idx)

            # Skip all-identity terms
            if not non_identity_indices:
                continue

            # Get coefficient (already validated to be real)
            coeff = term.coeff
            coeff_real = float(coeff.real) if isinstance(coeff, complex) else float(coeff)

            # Effective angle for this term: 2 * coeff * gamma
            angle = 2.0 * coeff_real * gamma

            # Apply gates based on number of Z operators
            if len(non_identity_indices) == 1:
                # Single-qubit Z term: RZ(2 * coeff * gamma)
                q = non_identity_indices[0]
                gate_rz = RZ(angle, dtype=state.dtype, device=state.device)
                state = apply_gate(state, gate_rz, qubit=q, n_qubits=self.n_qubits)
            else:  # len(non_identity_indices) == 2
                # Two-qubit ZZ term: CNOT, RZ, CNOT
                q0, q1 = non_identity_indices
                control, target = q0, q1
                gate_cnot = CNOT(
                    dtype=state.dtype, device=state.device, control_first=control < target
                )
                state = apply_two_qubit_gate(
                    state, gate_cnot, qubit1=control, qubit2=target, n_qubits=self.n_qubits
                )
                gate_rz = RZ(angle, dtype=state.dtype, device=state.device)
                state = apply_gate(state, gate_rz, qubit=target, n_qubits=self.n_qubits)
                state = apply_two_qubit_gate(
                    state, gate_cnot, qubit1=control, qubit2=target, n_qubits=self.n_qubits
                )

        return state

    def _apply_mixer_unitary(
        self,
        state: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the standard X-field mixer U_B(beta) = prod_i exp(-i beta X_i)
        to the state. Implemented via RX(2 * beta) rotations. This method
        returns a new state tensor via backend operations that preserve gradients.
        """
        angle = 2.0 * beta
        gate_rx = RX(angle, dtype=state.dtype, device=state.device)
        for q in range(self.n_qubits):
            state = apply_gate(state, gate_rx, qubit=q, n_qubits=self.n_qubits)
        return state

    def _append_mixer_unitary_layer(
        self,
        circuit: QuantumCircuit,
        beta: float,
    ) -> None:
        """
        Append the standard X-field mixer U_B(beta) = prod_i exp(-i beta X_i)
        to the circuit. Implemented via RX(2 * beta) rotations.
        """
        if abs(beta) == 0.0:
            return

        angle = 2.0 * beta
        for q in range(self.n_qubits):
            circuit.add_gate("RX", [q], params=[angle])


__all__ = [
    "Edge",
    "ising_maxcut_hamiltonian",
    "QAOAAnsatz",
]

