"""Bloch sphere coordinate computation and visualization.

This module provides utilities to compute Bloch coordinates from statevectors
or density matrices, and optional plotting helpers using matplotlib.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# Type hint for matplotlib Axes (optional dependency)
try:
    from matplotlib.axes import Axes
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Dummy type for type checking
    if False:
        from matplotlib.axes import Axes


def partial_trace_statevector(
    psi: torch.Tensor,
    qubit_index: int,
    n_qubits: int,
) -> torch.Tensor:
    """
    Compute the reduced density matrix for a single qubit by tracing out others.

    For a multi-qubit pure state |ψ⟩, compute ρ = Tr_{others}(|ψ⟩⟨ψ|) for
    the target qubit.

    Parameters
    ----------
    psi:
        Statevector tensor of shape (2**n_qubits,).
    qubit_index:
        Index of the qubit to extract (0-indexed, 0 = LSB).
    n_qubits:
        Total number of qubits.

    Returns
    -------
    torch.Tensor
        Reduced density matrix of shape (2, 2).
    """
    if psi.shape != (2**n_qubits,):
        raise ValueError(
            f"psi must have shape (2**n_qubits,), got {psi.shape}"
        )

    if qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError(
            f"qubit_index {qubit_index} out of range [0, {n_qubits})"
        )

    # Build full density matrix: rho_full = |psi><psi|
    rho_full = torch.outer(psi, psi.conj())

    # Perform partial trace: trace out all qubits except qubit_index
    # For qubit_index i, we need to sum over all basis states where
    # the other qubits match.
    dim = 2**n_qubits
    rho_reduced = torch.zeros((2, 2), dtype=rho_full.dtype, device=rho_full.device)

    for i in range(dim):
        for j in range(dim):
            # Extract bit values for qubit_index from indices i and j
            bit_i = (i >> qubit_index) & 1
            bit_j = (j >> qubit_index) & 1

            # Check if all other qubits match
            other_bits_match = True
            for q in range(n_qubits):
                if q != qubit_index:
                    bit_i_q = (i >> q) & 1
                    bit_j_q = (j >> q) & 1
                    if bit_i_q != bit_j_q:
                        other_bits_match = False
                        break

            if other_bits_match:
                rho_reduced[bit_i, bit_j] += rho_full[i, j]

    return rho_reduced


def bloch_coords_from_density(rho: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute Bloch coordinates (x, y, z) from a single-qubit density matrix.

    For a density matrix ρ = [[a, b], [b*, d]], the Bloch coordinates are:
        x = 2 * Re(b)
        y = 2 * Im(b)
        z = a - d

    Parameters
    ----------
    rho:
        Density matrix of shape (2, 2).

    Returns
    -------
    Tuple[float, float, float]
        Bloch coordinates (x, y, z).
    """
    if rho.shape != (2, 2):
        raise ValueError(f"rho must have shape (2, 2), got {rho.shape}")

    # Extract matrix elements (as complex numbers)
    a_complex = rho[0, 0].item()
    b_complex = rho[0, 1].item()
    d_complex = rho[1, 1].item()

    # Extract real parts (density matrices are Hermitian, so diagonal is real)
    a = float(a_complex.real)
    b = complex(b_complex)  # Off-diagonal can be complex
    d = float(d_complex.real)

    # Compute Bloch coordinates
    x = 2 * b.real
    y = 2 * b.imag
    z = a - d

    return (float(x), float(y), float(z))


def bloch_coords_from_statevector(
    psi: torch.Tensor,
    qubit_index: int,
    n_qubits: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute Bloch coordinates for a single qubit from a multi-qubit statevector.

    This computes the reduced density matrix for the target qubit and then
    extracts Bloch coordinates.

    Parameters
    ----------
    psi:
        Statevector tensor of shape (2**n_qubits,).
    qubit_index:
        Index of the qubit to analyze (0-indexed, 0 = LSB).
    n_qubits:
        Number of qubits. If None, inferred from psi.shape.

    Returns
    -------
    Tuple[float, float, float]
        Bloch coordinates (x, y, z).
    """
    dim = psi.shape[-1]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(
                f"psi dimension {dim} is not a power of 2. "
                "Please specify n_qubits explicitly."
            )

    if qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError(
            f"qubit_index {qubit_index} out of range [0, {n_qubits})"
        )

    # Flatten psi to ensure 1D
    psi_flat = psi.flatten()
    if psi_flat.shape != (dim,):
        raise ValueError(
            f"psi must have shape (2**n_qubits,), got {psi.shape}"
        )

    # Compute reduced density matrix
    rho = partial_trace_statevector(psi_flat, qubit_index, n_qubits)

    # Extract Bloch coordinates
    return bloch_coords_from_density(rho)


def plot_bloch_vector(
    coords: Tuple[float, float, float],
    ax: Optional["Axes"] = None,
) -> "Axes":
    """
    Plot a Bloch vector as an arrow on a Bloch sphere (2D projection).

    This function requires matplotlib. If matplotlib is not available,
    raises RuntimeError.

    Parameters
    ----------
    coords:
        Bloch coordinates (x, y, z).
    ax:
        Matplotlib axes to plot on. If None, creates a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object used for plotting.

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib required for plotting; install with pip install matplotlib"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    x, y, z = coords

    # Plot 2D projection (X-Y plane)
    ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc="blue", ec="blue")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Bloch Vector (x={x:.3f}, y={y:.3f}, z={z:.3f})")
    ax.grid(True)
    ax.set_aspect("equal")

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray")
    ax.add_patch(circle)

    return ax


def plot_bloch_projections(
    coords: Tuple[float, float, float],
    ax: Optional["Axes"] = None,
) -> "Axes":
    """
    Plot 2D projections of a Bloch vector (X-Y, X-Z, Y-Z planes).

    This function requires matplotlib. If matplotlib is not available,
    raises RuntimeError.

    Parameters
    ----------
    coords:
        Bloch coordinates (x, y, z).
    ax:
        Matplotlib axes to plot on. If None, creates a new figure with subplots.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object (or subplot axes if ax was None).

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib required for plotting; install with pip install matplotlib"
        )

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        axes = [ax]

    x, y, z = coords

    # X-Y projection
    if len(axes) > 0:
        ax_xy = axes[0]
        ax_xy.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc="blue", ec="blue")
        ax_xy.set_xlim(-1.2, 1.2)
        ax_xy.set_ylim(-1.2, 1.2)
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("X-Y Projection")
        ax_xy.grid(True)
        ax_xy.set_aspect("equal")
        circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray")
        ax_xy.add_patch(circle)

    # X-Z projection
    if len(axes) > 1:
        ax_xz = axes[1]
        ax_xz.arrow(0, 0, x, z, head_width=0.1, head_length=0.1, fc="green", ec="green")
        ax_xz.set_xlim(-1.2, 1.2)
        ax_xz.set_ylim(-1.2, 1.2)
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_title("X-Z Projection")
        ax_xz.grid(True)
        ax_xz.set_aspect("equal")
        circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray")
        ax_xz.add_patch(circle)

    # Y-Z projection
    if len(axes) > 2:
        ax_yz = axes[2]
        ax_yz.arrow(0, 0, y, z, head_width=0.1, head_length=0.1, fc="red", ec="red")
        ax_yz.set_xlim(-1.2, 1.2)
        ax_yz.set_ylim(-1.2, 1.2)
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("Y-Z Projection")
        ax_yz.grid(True)
        ax_yz.set_aspect("equal")
        circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray")
        ax_yz.add_patch(circle)

    if ax is None:
        plt.tight_layout()
        return axes[0]  # Return first subplot for compatibility
    return ax

