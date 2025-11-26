"""Tests for minimum spanning tree algorithms."""

import pytest

from qconduit.graphs import WeightedGraph, kruskal_mst, prim_mst


class TestKruskal:
    """Tests for Kruskal's algorithm."""

    def test_kruskal_simple(self):
        """Test Kruskal on simple graph."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1.0), ("B", "C", 2.0), ("A", "C", 3.0)]

        mst = kruskal_mst(nodes, edges)

        # MST should have 2 edges (n-1 for n=3)
        assert len(mst) == 2
        # Should include A-B and B-C (total weight 3.0)
        edge_set = {(u, v) for u, v, w in mst}
        assert ("A", "B") in edge_set or ("B", "A") in edge_set
        assert ("B", "C") in edge_set or ("C", "B") in edge_set

        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 3.0

    def test_kruskal_classic_example(self):
        """Test Kruskal on classic triangle example."""
        nodes = ["A", "B", "C"]
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 2.0),
            ("A", "C", 3.0),
        ]

        mst = kruskal_mst(nodes, edges)
        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 3.0  # A-B (1) + B-C (2)

    def test_kruskal_disconnected_forest(self):
        """Test Kruskal on disconnected graph (returns forest)."""
        nodes = ["A", "B", "C", "D"]
        edges = [
            ("A", "B", 1.0),  # Component 1
            ("C", "D", 2.0),  # Component 2
        ]

        mst = kruskal_mst(nodes, edges)
        # Should have 2 edges (one per component)
        assert len(mst) == 2
        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 3.0

    def test_kruskal_deterministic_tie_breaking(self):
        """Test that Kruskal breaks ties deterministically."""
        nodes = ["A", "B", "C", "D"]
        edges = [
            ("A", "B", 1.0),
            ("C", "D", 1.0),  # Same weight
            ("A", "C", 2.0),
        ]

        mst = kruskal_mst(nodes, edges)
        # Should include both weight-1 edges
        weights = [w for _, _, w in mst]
        assert 1.0 in weights
        assert len([w for w in weights if w == 1.0]) == 2

    def test_kruskal_single_node(self):
        """Test Kruskal on single node."""
        nodes = ["A"]
        edges = []

        mst = kruskal_mst(nodes, edges)
        assert len(mst) == 0

    def test_kruskal_empty(self):
        """Test Kruskal on empty graph."""
        nodes = []
        edges = []

        mst = kruskal_mst(nodes, edges)
        assert len(mst) == 0


class TestPrim:
    """Tests for Prim's algorithm."""

    def test_prim_simple(self):
        """Test Prim on simple graph."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)
        G.add_edge("A", "C", 3.0)

        mst = prim_mst(G, "A")

        assert len(mst) == 2
        total_weight = sum(w for _, _, w in mst)
        assert total_weight == 3.0

    def test_prim_vs_kruskal(self):
        """Test that Prim and Kruskal produce same MST weight."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)
        G.add_edge("A", "C", 3.0)
        G.add_edge("C", "D", 1.0)

        prim_result = prim_mst(G, "A")
        nodes = G.nodes()
        edges = G.edges()
        kruskal_result = kruskal_mst(nodes, edges)

        prim_weight = sum(w for _, _, w in prim_result)
        kruskal_weight = sum(w for _, _, w in kruskal_result)

        assert prim_weight == kruskal_weight

    def test_prim_disconnected_forest(self):
        """Test Prim on disconnected graph."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("C", "D", 2.0)

        mst = prim_mst(G, "A")
        # Should only include edges from component containing A
        assert len(mst) == 1
        assert mst[0][2] == 1.0

    def test_prim_deterministic_tie_breaking(self):
        """Test that Prim breaks ties deterministically."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("A", "C", 1.0)  # Same weight

        mst = prim_mst(G, "A")
        # Should include both edges (n-1 = 2 for n=3)
        assert len(mst) == 2
        assert all(w == 1.0 for _, _, w in mst)

    def test_prim_nonexistent_start(self):
        """Test Prim with nonexistent start node."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)

        with pytest.raises(ValueError):
            prim_mst(G, "C")

    def test_prim_single_node(self):
        """Test Prim on single node graph."""
        G = WeightedGraph()
        G.add_node("A")

        mst = prim_mst(G, "A")
        assert len(mst) == 0

    def test_prim_empty(self):
        """Test Prim on empty graph."""
        G = WeightedGraph()
        mst = prim_mst(G)
        assert len(mst) == 0

