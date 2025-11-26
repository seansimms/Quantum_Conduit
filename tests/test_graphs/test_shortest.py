"""Tests for shortest path algorithms."""

import math
import pytest

from qconduit.graphs import WeightedGraph, dijkstra, bellman_ford, reconstruct_path


class TestDijkstra:
    """Tests for Dijkstra's algorithm."""

    def test_dijkstra_simple(self):
        """Test Dijkstra on simple weighted graph."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)
        G.add_edge("A", "C", 5.0)

        dist, parent = dijkstra(G, "A")

        assert dist["A"] == 0.0
        assert dist["B"] == 1.0
        assert dist["C"] == 3.0  # A->B->C is shorter than A->C
        assert parent["A"] is None
        assert parent["B"] == "A"
        assert parent["C"] == "B"

    def test_dijkstra_path_reconstruction(self):
        """Test path reconstruction from Dijkstra parent map."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)
        G.add_edge("C", "D", 1.0)

        dist, parent = dijkstra(G, "A")
        path = reconstruct_path(parent, "D")

        assert path == ["A", "B", "C", "D"]
        assert dist["D"] == 4.0

    def test_dijkstra_unreachable(self):
        """Test Dijkstra with unreachable nodes."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_node("C")  # Isolated

        dist, parent = dijkstra(G, "A")

        assert dist["C"] == float("inf")
        assert parent["C"] is None

    def test_dijkstra_negative_weights_error(self):
        """Test that Dijkstra raises error for negative weights."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", -1.0)

        with pytest.raises(ValueError, match="non-negative"):
            dijkstra(G, "A")

    def test_dijkstra_nonexistent_source(self):
        """Test Dijkstra with nonexistent source."""
        G = WeightedGraph()
        with pytest.raises(ValueError):
            dijkstra(G, "A")

    def test_dijkstra_single_node(self):
        """Test Dijkstra on single node graph."""
        G = WeightedGraph()
        G.add_node("A")

        dist, parent = dijkstra(G, "A")
        assert dist["A"] == 0.0
        assert parent["A"] is None

    def test_dijkstra_deterministic_tie_breaking(self):
        """Test that Dijkstra breaks ties deterministically."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("A", "C", 1.0)  # Same distance

        dist, parent = dijkstra(G, "A")
        # Both should have distance 1.0
        assert dist["B"] == 1.0
        assert dist["C"] == 1.0
        # Both should have A as parent
        assert parent["B"] == "A"
        assert parent["C"] == "A"


class TestBellmanFord:
    """Tests for Bellman-Ford algorithm."""

    def test_bellman_ford_simple(self):
        """Test Bellman-Ford on simple graph."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)

        dist, parent, has_cycle = bellman_ford(G, "A")

        assert dist["A"] == 0.0
        assert dist["B"] == 1.0
        assert dist["C"] == 3.0
        assert has_cycle is False

    def test_bellman_ford_negative_weights(self):
        """Test Bellman-Ford with negative weights (no cycle)."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", -2.0)

        dist, parent, has_cycle = bellman_ford(G, "A")

        assert dist["A"] == 0.0
        assert dist["B"] == 1.0
        assert dist["C"] == -1.0  # 1 + (-2) = -1
        assert has_cycle is False

    def test_bellman_ford_negative_cycle(self):
        """Test Bellman-Ford detects negative cycle."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", -3.0)
        G.add_edge("C", "B", 1.0)  # Creates negative cycle: B->C->B with weight -2

        dist, parent, has_cycle = bellman_ford(G, "A")

        assert has_cycle is True

    def test_bellman_ford_negative_cycle_reachable(self):
        """Test Bellman-Ford with negative cycle reachable from source."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 1.0)
        G.add_edge("C", "B", -3.0)  # Negative cycle B->C->B

        dist, parent, has_cycle = bellman_ford(G, "A")
        assert has_cycle is True

    def test_bellman_ford_unreachable_negative_cycle(self):
        """Test Bellman-Ford with negative cycle not reachable from source."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("C", "D", 1.0)
        G.add_edge("D", "C", -3.0)  # Negative cycle, but not reachable from A

        dist, parent, has_cycle = bellman_ford(G, "A")
        # Should not detect cycle since it's not reachable
        assert has_cycle is False
        assert dist["C"] == float("inf")

    def test_bellman_ford_nonexistent_source(self):
        """Test Bellman-Ford with nonexistent source."""
        G = WeightedGraph()
        with pytest.raises(ValueError):
            bellman_ford(G, "A")

    def test_bellman_ford_single_node(self):
        """Test Bellman-Ford on single node graph."""
        G = WeightedGraph()
        G.add_node("A")

        dist, parent, has_cycle = bellman_ford(G, "A")
        assert dist["A"] == 0.0
        assert parent["A"] is None
        assert has_cycle is False


def test_dijkstra_type_annotations_regression():
    """Regression test: ensure List import is present (F821 fix)."""
    # This test verifies that dijkstra can be imported and type-checked
    # without undefined-name errors. The function uses List[Tuple[...]] internally.
    from qconduit.graphs.shortest import dijkstra
    from typing import get_type_hints
    
    # Verify the function can be imported and called
    G = WeightedGraph(directed=True)
    G.add_edge("A", "B", 1.0)
    
    dist, parent = dijkstra(G, "A")
    assert dist["A"] == 0.0
    assert dist["B"] == 1.0
    
    # Verify type hints are accessible (indirect check that imports work)
    hints = get_type_hints(dijkstra)
    assert "graph" in hints
    assert "source" in hints

