"""Tests for all-pairs shortest path algorithms."""

import math
import pytest

from qconduit.graphs import floyd_warshall, reconstruct_path


class TestFloydWarshall:
    """Tests for Floyd-Warshall algorithm."""

    def test_floyd_warshall_simple(self):
        """Test Floyd-Warshall on simple graph."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1.0), ("B", "C", 2.0)]

        dist, path = floyd_warshall(nodes, edges)

        assert dist[("A", "A")] == 0.0
        assert dist[("A", "B")] == 1.0
        assert dist[("A", "C")] == 3.0
        assert dist[("B", "C")] == 2.0
        assert dist[("C", "A")] == float("inf")  # No path

        assert path[("A", "C")] == ["A", "B", "C"]
        assert path[("A", "A")] == ["A"]

    def test_floyd_warshall_path_reconstruction(self):
        """Test path reconstruction in Floyd-Warshall."""
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B", 1.0), ("B", "C", 2.0), ("C", "D", 1.0)]

        dist, path = floyd_warshall(nodes, edges)

        assert path[("A", "D")] == ["A", "B", "C", "D"]
        assert dist[("A", "D")] == 4.0

    def test_floyd_warshall_negative_weights(self):
        """Test Floyd-Warshall with negative weights (no cycle)."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1.0), ("B", "C", -2.0)]

        dist, path = floyd_warshall(nodes, edges)

        assert dist[("A", "C")] == -1.0
        assert path[("A", "C")] == ["A", "B", "C"]

    def test_floyd_warshall_complete_graph(self):
        """Test Floyd-Warshall on complete graph."""
        nodes = ["A", "B", "C"]
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 2.0),
            ("A", "C", 4.0),  # Direct path is longer
        ]

        dist, path = floyd_warshall(nodes, edges)

        # A->B->C (3.0) is shorter than A->C (4.0)
        assert dist[("A", "C")] == 3.0
        assert path[("A", "C")] == ["A", "B", "C"]

    def test_floyd_warshall_single_node(self):
        """Test Floyd-Warshall on single node."""
        nodes = ["A"]
        edges = []

        dist, path = floyd_warshall(nodes, edges)

        assert dist[("A", "A")] == 0.0
        assert path[("A", "A")] == ["A"]

    def test_floyd_warshall_disconnected(self):
        """Test Floyd-Warshall on disconnected graph."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1.0)]  # C is isolated

        dist, path = floyd_warshall(nodes, edges)

        assert dist[("A", "B")] == 1.0
        assert dist[("A", "C")] == float("inf")
        assert path[("A", "C")] is None

    def test_floyd_warshall_deterministic_ordering(self):
        """Test that Floyd-Warshall uses deterministic node ordering."""
        nodes = ["C", "A", "B"]  # Unsorted
        edges = [("A", "B", 1.0), ("B", "C", 2.0)]

        dist, path = floyd_warshall(nodes, edges)

        # Should work regardless of input order
        assert dist[("A", "C")] == 3.0
        assert path[("A", "C")] == ["A", "B", "C"]

