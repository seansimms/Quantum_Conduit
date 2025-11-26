"""Tests for core graph data structures."""

import pytest

from qconduit.graphs import Graph, WeightedGraph


class TestGraph:
    """Tests for unweighted Graph class."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        G = Graph()
        assert G.directed is False
        assert len(G.nodes()) == 0
        assert len(G.edges()) == 0

    def test_add_node(self):
        """Test adding nodes."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        nodes = G.nodes()
        assert len(nodes) == 2
        assert "A" in nodes
        assert "B" in nodes

    def test_add_edge_undirected(self):
        """Test adding edges in undirected graph."""
        G = Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        nodes = G.nodes()
        assert len(nodes) == 3
        assert "A" in nodes
        assert "B" in nodes
        assert "C" in nodes

        edges = G.edges()
        assert len(edges) == 2
        assert ("A", "B") in edges or ("B", "A") in edges
        assert ("B", "C") in edges or ("C", "B") in edges

        # Check neighbors
        assert "B" in G.neighbors("A")
        assert "A" in G.neighbors("B")
        assert "C" in G.neighbors("B")
        assert "B" in G.neighbors("C")

    def test_add_edge_directed(self):
        """Test adding edges in directed graph."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        edges = G.edges()
        assert ("A", "B") in edges
        assert ("B", "C") in edges
        assert ("B", "A") not in edges

        assert "B" in G.neighbors("A")
        assert "C" in G.neighbors("B")
        assert "A" not in G.neighbors("B")

    def test_deterministic_neighbors(self):
        """Test that neighbors are returned in sorted order."""
        G = Graph()
        G.add_edge("A", "Z")
        G.add_edge("A", "M")
        G.add_edge("A", "B")

        neighbors = G.neighbors("A")
        # Should be sorted
        assert neighbors == sorted(neighbors, key=lambda x: str(x))
        assert neighbors[0] == "B"
        assert neighbors[1] == "M"
        assert neighbors[2] == "Z"

    def test_edges_deterministic(self):
        """Test that edges are returned in deterministic order."""
        G = Graph()
        G.add_edge("Z", "A")
        G.add_edge("B", "C")
        G.add_edge("A", "B")

        edges = G.edges()
        # Should be sorted
        edge_strs = [(str(u), str(v)) for u, v in edges]
        assert edge_strs == sorted(edge_strs)

    def test_neighbors_nonexistent_node(self):
        """Test that querying neighbors of nonexistent node raises error."""
        G = Graph()
        with pytest.raises(KeyError):
            G.neighbors("A")


class TestWeightedGraph:
    """Tests for WeightedGraph class."""

    def test_empty_weighted_graph(self):
        """Test empty weighted graph creation."""
        G = WeightedGraph()
        assert G.directed is False
        assert len(G.nodes()) == 0
        assert len(G.edges()) == 0

    def test_add_edge_with_weight(self):
        """Test adding weighted edges."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.5)
        G.add_edge("B", "C", 2.0)

        edges = G.edges()
        assert ("A", "B", 1.5) in edges or ("B", "A", 1.5) in edges
        assert ("B", "C", 2.0) in edges or ("C", "B", 2.0) in edges

        neighbors = G.neighbors("A")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "B"
        assert neighbors[0][1] == 1.5

    def test_add_edge_directed_weighted(self):
        """Test adding weighted edges in directed graph."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "A", 2.0)

        edges = G.edges()
        assert ("A", "B", 1.0) in edges
        assert ("B", "A", 2.0) in edges

        assert G.neighbors("A")[0][1] == 1.0
        assert G.neighbors("B")[0][1] == 2.0

    def test_update_edge_weight(self):
        """Test updating existing edge weight."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("A", "B", 2.0)  # Update weight

        neighbors = G.neighbors("A")
        assert len(neighbors) == 1
        assert neighbors[0][1] == 2.0

    def test_deterministic_neighbors_weighted(self):
        """Test that neighbors are returned in sorted order."""
        G = WeightedGraph()
        G.add_edge("A", "Z", 1.0)
        G.add_edge("A", "M", 2.0)
        G.add_edge("A", "B", 1.5)

        neighbors = G.neighbors("A")
        # Should be sorted by (node_str, weight)
        assert neighbors[0][0] == "B"
        assert neighbors[1][0] == "M"
        assert neighbors[2][0] == "Z"

    def test_negative_weights(self):
        """Test that negative weights are allowed."""
        G = WeightedGraph()
        G.add_edge("A", "B", -1.0)
        neighbors = G.neighbors("A")
        assert neighbors[0][1] == -1.0

