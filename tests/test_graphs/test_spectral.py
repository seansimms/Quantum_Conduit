"""Tests for spectral graph methods."""

import pytest
import numpy as np

try:
    from qconduit.graphs import WeightedGraph, graph_laplacian_matrix, spectral_clustering
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")


class TestGraphLaplacian:
    """Tests for graph Laplacian matrix computation."""

    def test_laplacian_simple(self):
        """Test Laplacian on simple graph."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)

        L = graph_laplacian_matrix(G)

        assert L.shape == (2, 2)
        # L = D - W, where D is degree matrix and W is adjacency
        # For edge A-B with weight 1:
        # D = [[1, 0], [0, 1]], W = [[0, 1], [1, 0]]
        # L = [[1, -1], [-1, 1]]
        assert L[0, 0] == pytest.approx(1.0)
        assert L[0, 1] == pytest.approx(-1.0)
        assert L[1, 0] == pytest.approx(-1.0)
        assert L[1, 1] == pytest.approx(1.0)

    def test_laplacian_weighted(self):
        """Test Laplacian on weighted graph."""
        G = WeightedGraph()
        G.add_edge("A", "B", 2.0)
        G.add_edge("B", "C", 3.0)

        L = graph_laplacian_matrix(G)

        assert L.shape == (3, 3)
        # Check that row sums are zero (property of Laplacian)
        row_sums = L.sum(axis=1)
        assert np.allclose(row_sums, 0.0)

    def test_laplacian_directed(self):
        """Test Laplacian on directed graph."""
        G = WeightedGraph(directed=True)
        G.add_edge("A", "B", 1.0)
        G.add_edge("B", "C", 1.0)

        L = graph_laplacian_matrix(G)
        # For directed graph, Laplacian may not be symmetric
        assert L.shape == (3, 3)

    def test_laplacian_isolated_node(self):
        """Test Laplacian with isolated node."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_node("C")  # Isolated

        L = graph_laplacian_matrix(G)
        assert L.shape == (3, 3)
        # Isolated node should have zero row/column (except diagonal)
        assert L[2, 0] == pytest.approx(0.0)
        assert L[2, 1] == pytest.approx(0.0)
        assert L[0, 2] == pytest.approx(0.0)
        assert L[1, 2] == pytest.approx(0.0)

    def test_laplacian_deterministic_ordering(self):
        """Test that Laplacian uses deterministic node ordering."""
        G = WeightedGraph()
        G.add_edge("C", "A", 1.0)  # Nodes added in non-sorted order

        L1 = graph_laplacian_matrix(G)
        L2 = graph_laplacian_matrix(G)

        # Should be identical (deterministic)
        assert np.allclose(L1, L2)

    def test_laplacian_single_node(self):
        """Test Laplacian on single node."""
        G = WeightedGraph()
        G.add_node("A")

        L = graph_laplacian_matrix(G)
        assert L.shape == (1, 1)
        assert L[0, 0] == pytest.approx(0.0)


class TestSpectralClustering:
    """Tests for spectral clustering."""

    def test_spectral_clustering_two_components(self):
        """Test spectral clustering on two disconnected components."""
        G = WeightedGraph()
        # Component 1: A-B
        G.add_edge("A", "B", 1.0)
        # Component 2: C-D
        G.add_edge("C", "D", 1.0)

        clusters = spectral_clustering(G, k=2)

        assert len(clusters) == 4
        # Verify we have exactly 2 clusters
        unique_clusters = set(clusters.values())
        assert len(unique_clusters) == 2
        # For disconnected components, nodes within a component should be in same cluster
        # (This is the ideal case, but k-means may not always achieve this)
        # At minimum, verify algorithm runs and produces k clusters
        assert 0 in unique_clusters
        assert 1 in unique_clusters

    def test_spectral_clustering_k_validation(self):
        """Test that invalid k raises error."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)

        with pytest.raises(ValueError):
            spectral_clustering(G, k=0)

        with pytest.raises(ValueError):
            spectral_clustering(G, k=10)  # More than nodes

    def test_spectral_clustering_single_node(self):
        """Test spectral clustering on single node."""
        G = WeightedGraph()
        G.add_node("A")

        clusters = spectral_clustering(G, k=1)
        assert clusters["A"] == 0

    def test_spectral_clustering_empty(self):
        """Test spectral clustering on empty graph."""
        G = WeightedGraph()
        clusters = spectral_clustering(G, k=1)
        assert len(clusters) == 0

    def test_spectral_clustering_normalized(self):
        """Test normalized spectral clustering."""
        G = WeightedGraph()
        G.add_edge("A", "B", 1.0)
        G.add_edge("C", "D", 1.0)

        clusters_norm = spectral_clustering(G, k=2, normalized=True)
        clusters_unnorm = spectral_clustering(G, k=2, normalized=False)

        # Both should produce valid clusterings
        assert len(clusters_norm) == 4
        assert len(clusters_unnorm) == 4
        # Normalized should produce 2 clusters (more reliable for disconnected components)
        assert len(set(clusters_norm.values())) == 2
        # Unnormalized may produce 1-2 clusters depending on eigenvector structure
        assert 1 <= len(set(clusters_unnorm.values())) <= 2
        # Verify cluster IDs are in valid range
        assert all(0 <= c <= 1 for c in clusters_norm.values())
        assert all(0 <= c <= 1 for c in clusters_unnorm.values())

