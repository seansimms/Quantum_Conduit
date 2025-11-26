"""Tests for graph utility functions."""

import pytest

from qconduit.graphs import WeightedGraph, node_index_map, edges_from_weighted_graph, reconstruct_path


class TestNodeIndexMap:
    """Tests for node_index_map function."""

    def test_node_index_map_simple(self):
        """Test node index mapping on simple list."""
        nodes = ["C", "A", "B"]
        node_to_idx, idx_to_node = node_index_map(nodes)

        assert node_to_idx["A"] == 0
        assert node_to_idx["B"] == 1
        assert node_to_idx["C"] == 2
        assert idx_to_node == ["A", "B", "C"]

    def test_node_index_map_deterministic(self):
        """Test that node index mapping is deterministic."""
        nodes1 = ["C", "A", "B"]
        nodes2 = ["A", "B", "C"]

        map1 = node_index_map(nodes1)
        map2 = node_index_map(nodes2)

        assert map1[0] == map2[0]  # Same node_to_index
        assert map1[1] == map2[1]  # Same idx_to_node

    def test_node_index_map_duplicates(self):
        """Test that duplicates are handled."""
        nodes = ["A", "B", "A", "C"]
        node_to_idx, idx_to_node = node_index_map(nodes)

        assert len(node_to_idx) == 3
        assert len(idx_to_node) == 3
        assert "A" in node_to_idx
        assert "B" in node_to_idx
        assert "C" in node_to_idx


class TestEdgesFromWeightedGraph:
    """Tests for edges_from_weighted_graph function."""

    def test_edges_from_weighted_graph(self):
        """Test edge iteration from weighted graph."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 2.0)
        G.add_edge("A", "C", 3.0)

        edges = list(edges_from_weighted_graph(G))
        assert len(edges) == 3
        # Should be sorted deterministically
        edge_strs = [(str(u), str(v), w) for u, v, w in edges]
        assert edge_strs == sorted(edge_strs)

    def test_edges_from_weighted_graph_deterministic(self):
        """Test that edge iteration is deterministic."""
        G = WeightedGraph()
        G.add_edge("Z", "A", 1.0)
        G.add_edge("B", "C", 1.0)

        edges1 = list(edges_from_weighted_graph(G))
        edges2 = list(edges_from_weighted_graph(G))

        assert edges1 == edges2


class TestReconstructPath:
    """Tests for reconstruct_path function."""

    def test_reconstruct_path_simple(self):
        """Test path reconstruction on simple path."""
        parent = {"A": None, "B": "A", "C": "B"}
        path = reconstruct_path(parent, "C")

        assert path == ["A", "B", "C"]

    def test_reconstruct_path_source(self):
        """Test path reconstruction when target is source."""
        parent = {"A": None, "B": "A"}
        path = reconstruct_path(parent, "A")

        assert path == ["A"]

    def test_reconstruct_path_unreachable(self):
        """Test path reconstruction for unreachable node."""
        parent = {"A": None, "B": "A"}
        path = reconstruct_path(parent, "C")

        assert path is None

    def test_reconstruct_path_cycle_detection(self):
        """Test that cycle detection works in path reconstruction."""
        # Invalid parent map with cycle (shouldn't happen in real algorithms)
        parent = {"A": "B", "B": "A", "C": "B"}
        path = reconstruct_path(parent, "C")

        # Should detect cycle and return None
        assert path is None

    def test_reconstruct_path_empty_parent(self):
        """Test path reconstruction with empty parent map."""
        parent = {}
        path = reconstruct_path(parent, "A")

        assert path is None

