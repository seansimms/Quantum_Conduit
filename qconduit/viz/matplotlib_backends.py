"""Optional matplotlib styling helpers for visualization.

This module provides aesthetic styling utilities for Bloch sphere and
circuit visualization plots. Matplotlib is an optional dependency.
"""

from __future__ import annotations

from typing import Tuple

# Type hints for matplotlib (optional dependency)
try:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Dummy types for type checking
    if False:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure


def create_bloch_figure(size: int = 4) -> "Figure":
    """
    Create a matplotlib figure with aesthetic styling for Bloch sphere plots.

    Parameters
    ----------
    size:
        Figure size (width and height in inches).

    Returns
    -------
    matplotlib.figure.Figure
        Configured figure.

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib required for plotting; install with pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    return fig


def create_projection_axes() -> Tuple["Axes", "Axes", "Axes"]:
    """
    Create three matplotlib axes for Bloch sphere projections (X-Y, X-Z, Y-Z).

    Returns
    -------
    Tuple[Axes, Axes, Axes]
        Three axes objects for X-Y, X-Z, and Y-Z projections.

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib required for plotting; install with pip install matplotlib"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax in axes:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    axes[0].set_xlabel("X", fontsize=10)
    axes[0].set_ylabel("Y", fontsize=10)
    axes[0].set_title("X-Y Projection", fontsize=11)

    axes[1].set_xlabel("X", fontsize=10)
    axes[1].set_ylabel("Z", fontsize=10)
    axes[1].set_title("X-Z Projection", fontsize=11)

    axes[2].set_xlabel("Y", fontsize=10)
    axes[2].set_ylabel("Z", fontsize=10)
    axes[2].set_title("Y-Z Projection", fontsize=11)

    plt.tight_layout()

    return axes[0], axes[1], axes[2]



