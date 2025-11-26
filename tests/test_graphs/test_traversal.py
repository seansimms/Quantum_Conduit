"""Tests for graph traversal algorithms."""

import pytest

from qconduit.graphs import Graph, bfs, dfs_recursive, dfs_iterative


class TestBFS:
    """Tests for breadth-first search."""

    def test_bfs_simple(self):
        """Test BFS on simple graph."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")

        order, dist, parent = bfs(G, "A")

        assert order[0] == "A"
        assert dist["A"] == 0
        assert dist["B"] == 1
        assert dist["C"] == 1
        assert dist["D"] == 2
        assert parent["A"] is None
        assert parent["B"] == "A"
        assert parent["C"] == "A"
        assert parent["D"] == "B"

    def test_bfs_deterministic_order(self):
        """Test that BFS order is deterministic."""
        G = Graph()
        G.add_edge("A", "C")
        G.add_edge("A", "B")

        order, _, _ = bfs(G, "A")
        # B and C should be visited in sorted order
        assert order == ["A", "B", "C"]

    def test_bfs_disconnected(self):
        """Test BFS on disconnected graph."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_node("C")  # Isolated node

        order, dist, parent = bfs(G, "A")

        assert "A" in order
        assert "B" in order
        assert "C" not in order  # Unreachable
        assert dist["C"] == float("inf")
        assert parent["C"] is None

    def test_bfs_nonexistent_source(self):
        """Test BFS with nonexistent source."""
        G = Graph()
        order, dist, parent = bfs(G, "A")
        assert len(order) == 0
        assert len(dist) == 0
        assert len(parent) == 0

    def test_bfs_single_node(self):
        """Test BFS on single node graph."""
        G = Graph()
        G.add_node("A")

        order, dist, parent = bfs(G, "A")
        assert order == ["A"]
        assert dist["A"] == 0
        assert parent["A"] is None


class TestDFS:
    """Tests for depth-first search."""

    def test_dfs_recursive_simple(self):
        """Test recursive DFS on simple graph."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")

        pre, post, parent = dfs_recursive(G, "A")

        assert "A" in pre
        assert "A" in post
        assert pre[0] == "A"  # A is first in preorder
        assert post[-1] == "A"  # A is last in postorder
        assert parent["A"] is None
        assert parent["B"] == "A"

    def test_dfs_iterative_simple(self):
        """Test iterative DFS on simple graph."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")

        pre, post, parent = dfs_iterative(G, "A")

        assert "A" in pre
        assert "A" in post
        assert pre[0] == "A"
        assert post[-1] == "A"
        assert parent["A"] is None
        assert parent["B"] == "A"

    def test_dfs_recursive_vs_iterative(self):
        """Test that recursive and iterative DFS produce same parent map."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "E")

        pre1, post1, parent1 = dfs_recursive(G, "A")
        pre2, post2, parent2 = dfs_iterative(G, "A")

        # Parent maps should be equivalent (same reachable nodes)
        assert set(parent1.keys()) == set(parent2.keys())
        for node in parent1:
            if parent1[node] is None:
                assert parent2[node] is None
            else:
                assert parent1[node] == parent2[node]

    def test_dfs_deterministic_order(self):
        """Test that DFS visits neighbors in sorted order."""
        G = Graph()
        G.add_edge("A", "Z")
        G.add_edge("A", "B")
        G.add_edge("A", "M")

        pre, _, _ = dfs_recursive(G, "A")
        # After A, should visit B first (sorted order)
        assert pre[1] == "B"

    def test_dfs_disconnected(self):
        """Test DFS on disconnected graph."""
        G = Graph()
        G.add_edge("A", "B")
        G.add_node("C")

        pre, post, parent = dfs_recursive(G, "A")

        assert "A" in pre
        assert "B" in pre
        assert "C" not in pre
        assert parent.get("C") is None

    def test_dfs_nonexistent_source(self):
        """Test DFS with nonexistent source."""
        G = Graph()
        pre, post, parent = dfs_recursive(G, "A")
        assert len(pre) == 0
        assert len(post) == 0
        assert len(parent) == 0

