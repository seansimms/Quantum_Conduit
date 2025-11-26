"""Integration tests for graphs package within qconduit."""

import pytest

# Test that graphs integrates properly with main qconduit package
def test_graphs_import_from_main():
    """Test that graphs can be imported from main qconduit package."""
    import qconduit
    
    # Test direct import
    from qconduit import Graph, WeightedGraph
    assert Graph is not None
    assert WeightedGraph is not None
    
    # Test algorithm imports
    from qconduit import dijkstra, bfs, pagerank
    assert dijkstra is not None
    assert bfs is not None
    assert pagerank is not None


def test_graphs_in_all_exports():
    """Test that graph exports are in __all__."""
    import qconduit
    
    graph_exports = {
        'Graph', 'WeightedGraph', 'bfs', 'dfs_recursive', 'dfs_iterative',
        'dijkstra', 'bellman_ford', 'floyd_warshall', 'kruskal_mst', 'prim_mst',
        'pagerank', 'graph_laplacian_matrix', 'spectral_clustering',
        'node_index_map', 'edges_from_weighted_graph', 'reconstruct_path'
    }
    
    all_exports = set(qconduit.__all__)
    assert graph_exports.issubset(all_exports), "Graph exports missing from __all__"


def test_graphs_no_circular_imports():
    """Test that importing graphs doesn't break other qconduit imports."""
    # Import graphs first
    from qconduit import Graph, WeightedGraph, dijkstra
    
    # Then import other modules
    from qconduit import Device, QuantumCircuit
    from qconduit import VQE, QAOAAnsatz
    from qconduit import PauliSum, PauliTerm
    
    # All should work
    assert Graph is not None
    assert Device is not None
    assert VQE is not None
    assert PauliSum is not None


def test_graphs_functional_integration():
    """Test that graphs work in a realistic usage scenario."""
    from qconduit import WeightedGraph, dijkstra, reconstruct_path
    
    # Create a graph
    G = WeightedGraph(directed=True)
    G.add_edge('start', 'A', 1.0)
    G.add_edge('A', 'B', 2.0)
    G.add_edge('B', 'end', 1.0)
    G.add_edge('start', 'end', 5.0)  # Longer direct path
    
    # Find shortest path
    dist, parent = dijkstra(G, 'start')
    
    # Reconstruct path
    path = reconstruct_path(parent, 'end')
    
    assert dist['end'] == 4.0
    assert path == ['start', 'A', 'B', 'end']


def test_graphs_with_other_modules():
    """Test that graphs can be used alongside other qconduit features."""
    from qconduit import Graph, pagerank, default_device
    
    # Create a simple graph
    G = Graph(directed=True)
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    
    # Compute PageRank
    scores = pagerank(G)
    
    # Verify it works
    assert len(scores) == 3
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    
    # Device should still work
    device = default_device()
    assert device is not None

