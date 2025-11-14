"""Tests for Ising/MaxCut Hamiltonian construction."""

from __future__ import annotations

import math

import pytest
import torch

from qconduit.operators import PauliTerm, PauliSum
from qconduit.algorithms import Edge, ising_maxcut_hamiltonian


def test_ising_maxcut_hamiltonian_two_nodes_single_edge() -> None:
    """Test MaxCut Hamiltonian for a 2-node graph with a single edge."""
    num_nodes = 2
    edges = [(0, 1)]

    H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)
    assert isinstance(H, PauliSum)
    assert len(H.terms) == 2  # one ZZ term + one constant term

    # Extract terms for inspection.
    coeffs = {}
    for term in H.terms:
        key = term.paulis
        coeffs[key] = term.coeff

    # Constant term (I, I) with coeff 0.5
    assert ("I", "I") in coeffs
    assert abs(float(coeffs[("I", "I")]) - 0.5) < 1e-8

    # ZZ term (Z, Z) with coeff -0.5
    assert ("Z", "Z") in coeffs
    assert abs(float(coeffs[("Z", "Z")]) + 0.5) < 1e-8


def test_ising_maxcut_hamiltonian_weighted_triangle() -> None:
    """Test MaxCut Hamiltonian for a weighted triangle graph."""
    num_nodes = 3
    edges = [(0, 1), (1, 2), (0, 2)]
    weights = [1.0, 2.0, 3.0]

    H = ising_maxcut_hamiltonian(
        num_nodes=num_nodes,
        edges=edges,
        weights=weights,
        include_constant=True,
    )

    # Expect one constant term with coeff = sum w_ij / 2 = (1 + 2 + 3) / 2 = 3.0
    const_coeff = 0.0
    zz_coeff_sum = 0.0
    for term in H.terms:
        if all(p == "I" for p in term.paulis):
            const_coeff += float(term.coeff)
        else:
            zz_coeff_sum += float(term.coeff)

    assert abs(const_coeff - 3.0) < 1e-8
    # ZZ terms sum should be -const_coeff (because each is -w_ij / 2)
    assert abs(zz_coeff_sum + const_coeff) < 1e-8


def test_ising_maxcut_hamiltonian_with_edge_objects() -> None:
    """Test MaxCut Hamiltonian construction using Edge objects."""
    num_nodes = 2
    edges = [Edge(0, 1, weight=2.0)]

    H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

    # Should have constant term with coeff 1.0 and ZZ term with coeff -1.0
    const_coeff = 0.0
    zz_coeff = 0.0
    for term in H.terms:
        if all(p == "I" for p in term.paulis):
            const_coeff += float(term.coeff)
        else:
            zz_coeff += float(term.coeff)

    assert abs(const_coeff - 1.0) < 1e-8
    assert abs(zz_coeff + 1.0) < 1e-8


def test_ising_maxcut_hamiltonian_no_constant() -> None:
    """Test MaxCut Hamiltonian without constant term."""
    num_nodes = 2
    edges = [(0, 1)]

    H = ising_maxcut_hamiltonian(
        num_nodes=num_nodes, edges=edges, include_constant=False
    )

    # Should only have ZZ term, no constant
    const_terms = [t for t in H.terms if all(p == "I" for p in t.paulis)]
    assert len(const_terms) == 0

    # Should have one ZZ term
    zz_terms = [t for t in H.terms if not all(p == "I" for p in t.paulis)]
    assert len(zz_terms) == 1
    assert abs(float(zz_terms[0].coeff) + 0.5) < 1e-8


def test_ising_maxcut_hamiltonian_negative_num_nodes() -> None:
    """Test that negative num_nodes raises ValueError."""
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        ising_maxcut_hamiltonian(num_nodes=0, edges=[(0, 1)])


def test_ising_maxcut_hamiltonian_negative_weight() -> None:
    """Test that negative edge weights raise ValueError."""
    with pytest.raises(ValueError, match="Edge weights must be non-negative"):
        ising_maxcut_hamiltonian(num_nodes=2, edges=[(0, 1)], weights=[-1.0])


def test_ising_maxcut_hamiltonian_out_of_bounds_edge() -> None:
    """Test that out-of-bounds edges raise ValueError."""
    with pytest.raises(ValueError, match="out of bounds"):
        ising_maxcut_hamiltonian(num_nodes=2, edges=[(0, 2)])


def test_ising_maxcut_hamiltonian_weights_length_mismatch() -> None:
    """Test that mismatched weights/edges lengths raise ValueError."""
    with pytest.raises(ValueError, match="weights must have the same length"):
        ising_maxcut_hamiltonian(
            num_nodes=2, edges=[(0, 1), (0, 1)], weights=[1.0]
        )


def test_ising_maxcut_hamiltonian_weights_with_edge_objects() -> None:
    """Test that providing weights with Edge objects raises ValueError."""
    with pytest.raises(ValueError, match="weights must be None when edges are Edge objects"):
        ising_maxcut_hamiltonian(
            num_nodes=2, edges=[Edge(0, 1)], weights=[1.0]
        )


def test_ising_maxcut_hamiltonian_self_loop_skipped() -> None:
    """Test that self-loops are skipped."""
    num_nodes = 2
    edges = [(0, 0), (0, 1)]  # self-loop + valid edge

    H = ising_maxcut_hamiltonian(num_nodes=num_nodes, edges=edges, include_constant=True)

    # Should only have terms from the (0, 1) edge, not the self-loop
    # Constant term: 0.5, ZZ term: -0.5
    const_coeff = 0.0
    zz_coeff = 0.0
    for term in H.terms:
        if all(p == "I" for p in term.paulis):
            const_coeff += float(term.coeff)
        else:
            zz_coeff += float(term.coeff)

    assert abs(const_coeff - 0.5) < 1e-8
    assert abs(zz_coeff + 0.5) < 1e-8


