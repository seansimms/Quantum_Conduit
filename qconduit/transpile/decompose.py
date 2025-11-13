"""Gate decomposition rules for basis transpilation."""

from __future__ import annotations

from math import pi, isclose
from typing import Sequence

from qconduit.circuit import QuantumCircuit


def _is_multiple_of_pi_over_4(theta: float, atol: float = 1e-8) -> bool:
    """
    Check whether theta is approximately a multiple of π/4.

    This is useful for identifying when a Z-rotation can be expressed exactly
    using S and T gates in a Clifford+T-style decomposition.

    Parameters
    ----------
    theta:
        Angle in radians to check.
    atol:
        Absolute tolerance for the check.

    Returns
    -------
    bool
        True if theta is approximately a multiple of π/4, False otherwise.
    """
    k = round(theta / (pi / 4.0))
    return isclose(theta, k * (pi / 4.0), abs_tol=atol)


def decompose_h_to_rz_rx_rz(
    circuit: QuantumCircuit,
    qubit: int,
) -> None:
    """
    Decompose the Hadamard gate H on a single qubit into Z-Y-Z-style rotations.

    One standard decomposition (up to global phase) is:

        H = Rz(π/2) Rx(π/2) Rz(π/2).

    This function appends the three rotation gates to `circuit` in that order.

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gates to.
    qubit:
        Qubit index on which to apply the decomposition.
    """
    circuit.add_gate("RZ", [qubit], [pi / 2.0])
    circuit.add_gate("RX", [qubit], [pi / 2.0])
    circuit.add_gate("RZ", [qubit], [pi / 2.0])


def decompose_x_to_rx(
    circuit: QuantumCircuit,
    qubit: int,
) -> None:
    """
    Decompose X as a rotation Rx(π) (up to global phase).

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gate to.
    qubit:
        Qubit index on which to apply the decomposition.
    """
    circuit.add_gate("RX", [qubit], [pi])


def decompose_y_to_ry(
    circuit: QuantumCircuit,
    qubit: int,
) -> None:
    """
    Decompose Y as a rotation Ry(π) (up to global phase).

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gate to.
    qubit:
        Qubit index on which to apply the decomposition.
    """
    circuit.add_gate("RY", [qubit], [pi])


def decompose_z_to_rz(
    circuit: QuantumCircuit,
    qubit: int,
) -> None:
    """
    Decompose Z as a rotation Rz(π) (up to global phase).

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gate to.
    qubit:
        Qubit index on which to apply the decomposition.
    """
    circuit.add_gate("RZ", [qubit], [pi])


def decompose_y_to_rz_rx_rz(
    circuit: QuantumCircuit,
    qubit: int,
) -> None:
    """
    Decompose Y as Y = Rz(π/2) Rx(π) Rz(-π/2) (up to global phase).

    This allows Y to be decomposed into {RZ, RX} basis.

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gates to.
    qubit:
        Qubit index on which to apply the decomposition.
    """
    circuit.add_gate("RZ", [qubit], [pi / 2.0])
    circuit.add_gate("RX", [qubit], [pi])
    circuit.add_gate("RZ", [qubit], [-pi / 2.0])


def decompose_rz_to_clifford_t(
    circuit: QuantumCircuit,
    qubit: int,
    theta: float,
    atol: float = 1e-8,
) -> bool:
    """
    Decompose Rz(theta) into a sequence of S and T gates when theta is an
    integer multiple of π/4.

    For example:
        theta = π/2  -> S or S^\\dagger
        theta = π/4  -> T or T^\\dagger

    If theta is not (within atol) a multiple of π/4, this function returns
    False and does not modify the circuit. If it succeeds, it returns True.

    The decomposition is exact up to a global phase and uses only S and T (and
    their daggers) applied repeatedly.

    Parameters
    ----------
    circuit:
        Circuit to append the decomposition gates to.
    qubit:
        Qubit index on which to apply the decomposition.
    theta:
        Rotation angle in radians.
    atol:
        Absolute tolerance for checking if theta is a multiple of π/4.

    Returns
    -------
    bool
        True if the decomposition was successful (theta was a multiple of π/4),
        False otherwise.
    """
    if not _is_multiple_of_pi_over_4(theta, atol):
        return False

    k = round(theta / (pi / 4.0))
    # Reduce modulo 8 since Z rotations are 2π-periodic
    k_mod = k % 8

    if k_mod == 0:
        # Identity (up to global phase) - no gates needed
        pass
    elif k_mod == 1:
        # T gate
        circuit.add_gate("T", [qubit])
    elif k_mod == 2:
        # S gate (Rz(π/2))
        circuit.add_gate("S", [qubit])
    elif k_mod == 3:
        # S · T
        circuit.add_gate("S", [qubit])
        circuit.add_gate("T", [qubit])
    elif k_mod == 4:
        # Z gate (S · S)
        circuit.add_gate("S", [qubit])
        circuit.add_gate("S", [qubit])
    elif k_mod == 5:
        # Z · T (S · S · T)
        circuit.add_gate("S", [qubit])
        circuit.add_gate("S", [qubit])
        circuit.add_gate("T", [qubit])
    elif k_mod == 6:
        # S^\dagger (S^3, which is S · S · S)
        circuit.add_gate("S", [qubit])
        circuit.add_gate("S", [qubit])
        circuit.add_gate("S", [qubit])
    elif k_mod == 7:
        # T^\dagger (T^7, which is T · T · T · T · T · T · T)
        # For simplicity, we use T^7 = T · T · T · T · T · T · T
        for _ in range(7):
            circuit.add_gate("T", [qubit])
    else:
        # Should never happen, but handle for completeness
        return False

    return True


def decompose_gate_to_basis(
    source_circuit: QuantumCircuit,
    gate_index: int,
    target_basis: Sequence[str],
) -> QuantumCircuit:
    """
    Decompose a single gate from `source_circuit` at the given index into a
    sequence of gates supported by `target_basis`.

    This function returns a NEW QuantumCircuit with the same number of qubits
    containing ONLY the decomposition for that one gate (no surrounding gates).

    Parameters
    ----------
    source_circuit:
        Circuit containing the gate to be decomposed.
    gate_index:
        Index of the gate in the circuit's gate list.
    target_basis:
        Iterable of gate names that are allowed in the target basis, e.g.
        ["RX", "RZ", "CNOT"] or ["H", "S", "T", "CNOT"].

    Returns
    -------
    QuantumCircuit
        New circuit with the decomposition of the selected gate into the
        target basis.

    Raises
    ------
    ValueError
        If the gate cannot be decomposed into the given basis.
    IndexError
        If gate_index is out of range.
    """
    if gate_index < 0 or gate_index >= len(source_circuit.ops):
        raise IndexError(
            f"Gate index {gate_index} is out of range for circuit with "
            f"{len(source_circuit.ops)} gates."
        )

    gate = source_circuit.ops[gate_index]
    name = gate.name.upper()
    target_basis_upper = {b.upper() for b in target_basis}

    # Create new circuit with same number of qubits
    decomp = QuantumCircuit(source_circuit.n_qubits)

    # If gate is already in target basis, copy it
    if name in target_basis_upper:
        decomp.add_gate(name, gate.qubits, gate.params)
        return decomp

    # Handle single-qubit gates
    if len(gate.qubits) == 1:
        qubit = gate.qubits[0]

        # H decomposition
        if name == "H":
            if "RZ" in target_basis_upper and "RX" in target_basis_upper:
                decompose_h_to_rz_rx_rz(decomp, qubit)
                return decomp

        # X decomposition
        if name == "X":
            if "RX" in target_basis_upper:
                decompose_x_to_rx(decomp, qubit)
                return decomp

        # Y decomposition
        if name == "Y":
            if "RY" in target_basis_upper:
                decompose_y_to_ry(decomp, qubit)
                return decomp
            # Alternative: Y = Rz(π/2) Rx(π) Rz(-π/2)
            if "RZ" in target_basis_upper and "RX" in target_basis_upper:
                decompose_y_to_rz_rx_rz(decomp, qubit)
                return decomp

        # Z decomposition
        if name == "Z":
            if "RZ" in target_basis_upper:
                decompose_z_to_rz(decomp, qubit)
                return decomp

        # RZ to Clifford+T decomposition
        if name == "RZ":
            if gate.params is None or len(gate.params) != 1:
                raise ValueError(f"RZ gate requires exactly one parameter, got {gate.params}.")
            theta = gate.params[0]

            # Check if we're targeting Clifford+T basis
            if "S" in target_basis_upper and "T" in target_basis_upper:
                if decompose_rz_to_clifford_t(decomp, qubit, theta):
                    return decomp
                # If decomposition failed and RZ is not in basis, raise error
                if "RZ" not in target_basis_upper:
                    raise ValueError(
                        f"Cannot decompose RZ({theta}) into Clifford+T basis "
                        f"(theta is not a multiple of π/4)."
                    )

            # If RZ is in target basis, just copy it
            if "RZ" in target_basis_upper:
                decomp.add_gate("RZ", [qubit], [theta])
                return decomp

        # RX, RY, RZ are already handled if in basis
        if name in ("RX", "RY", "RZ"):
            if name in target_basis_upper:
                decomp.add_gate(name, gate.qubits, gate.params)
                return decomp

        # S, T, H are already handled if in basis
        if name in ("S", "T", "H"):
            if name in target_basis_upper:
                decomp.add_gate(name, gate.qubits, gate.params)
                return decomp

    # Handle two-qubit gates
    if len(gate.qubits) == 2:
        # CNOT
        if name == "CNOT":
            if "CNOT" in target_basis_upper:
                decomp.add_gate("CNOT", gate.qubits, gate.params)
                return decomp

    # If we get here, no decomposition rule applies
    raise ValueError(
        f"Cannot decompose gate '{name}' into target basis {sorted(target_basis_upper)}."
    )


__all__ = [
    "decompose_h_to_rz_rx_rz",
    "decompose_x_to_rx",
    "decompose_y_to_ry",
    "decompose_y_to_rz_rx_rz",
    "decompose_z_to_rz",
    "decompose_rz_to_clifford_t",
    "decompose_gate_to_basis",
]

