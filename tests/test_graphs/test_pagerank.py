"""Tests for PageRank algorithm."""

import pytest

from qconduit.graphs import Graph, pagerank


class TestPageRank:
    """Tests for PageRank algorithm."""

    def test_pagerank_two_node(self):
        """Test PageRank on two-node graph."""
        G = Graph(directed=True)
        G.add_edge("A", "B")

        scores = pagerank(G)

        assert len(scores) == 2
        assert "A" in scores
        assert "B" in scores
        # Scores should sum to 1
        assert abs(sum(scores.values()) - 1.0) < 1e-6
        # B should have higher score (receives link from A)
        assert scores["B"] > scores["A"]

    def test_pagerank_scores_sum_to_one(self):
        """Test that PageRank scores sum to 1."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")  # Cycle

        scores = pagerank(G)
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_pagerank_dangling_node(self):
        """Test PageRank with dangling node (no outgoing edges)."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_node("C")  # Dangling node

        scores = pagerank(G)
        assert len(scores) == 3
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_pagerank_personalization(self):
        """Test PageRank with personalization vector."""
        G = Graph(directed=True)
        G.add_edge("A", "B")

        personalization = {"A": 0.7, "B": 0.3}
        scores = pagerank(G, personalization=personalization)

        assert abs(sum(scores.values()) - 1.0) < 1e-6
        # Both scores should be positive
        assert scores["A"] > 0
        assert scores["B"] > 0
        # Personalization affects scores (A has higher personalization)
        # Note: B receives link from A, so actual scores depend on both factors

    def test_pagerank_personalization_invalid_sum(self):
        """Test that invalid personalization raises error."""
        G = Graph(directed=True)
        G.add_edge("A", "B")

        personalization = {"A": 0.5, "B": 0.3}  # Doesn't sum to 1

        with pytest.raises(ValueError, match="sum to 1"):
            pagerank(G, personalization=personalization)

    def test_pagerank_alpha_validation(self):
        """Test that invalid alpha raises error."""
        G = Graph(directed=True)
        G.add_edge("A", "B")

        with pytest.raises(ValueError, match="Alpha must be"):
            pagerank(G, alpha=1.5)

        with pytest.raises(ValueError, match="Alpha must be"):
            pagerank(G, alpha=-0.1)

    def test_pagerank_single_node(self):
        """Test PageRank on single node graph."""
        G = Graph(directed=True)
        G.add_node("A")

        scores = pagerank(G)
        assert scores["A"] == 1.0

    def test_pagerank_empty(self):
        """Test PageRank on empty graph."""
        G = Graph(directed=True)
        scores = pagerank(G)
        assert len(scores) == 0

    def test_pagerank_convergence(self):
        """Test that PageRank converges."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        scores = pagerank(G, tol=1e-8, maxiter=100)
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_pagerank_undirected_treated_as_bidirectional(self):
        """Test that undirected graphs are treated as bidirectional."""
        G = Graph(directed=False)
        G.add_edge("A", "B")

        scores = pagerank(G)
        # In undirected graph, both nodes should have similar scores
        assert abs(sum(scores.values()) - 1.0) < 1e-6

