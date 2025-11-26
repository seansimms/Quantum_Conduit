"""
Graph algorithms package for qconduit.

This package provides canonical textbook graph algorithms including:
- Graph data structures (Graph, WeightedGraph)
- Traversal algorithms (BFS, DFS)
- Shortest path algorithms (Dijkstra, Bellman-Ford)
- All-pairs shortest paths (Floyd-Warshall)
- Minimum spanning trees (Kruskal, Prim)
- PageRank
- Spectral methods (Laplacian, spectral clustering)

All algorithms are deterministic and use sorted node ordering for reproducibility.
"""

from .allpairs import floyd_warshall
from .core import Graph, WeightedGraph
from .mst import kruskal_mst, prim_mst
from .pagerank import pagerank
from .shortest import bellman_ford, dijkstra
from .spectral import graph_laplacian_matrix, spectral_clustering
from .traversal import bfs, dfs_iterative, dfs_recursive
from .utils import edges_from_weighted_graph, node_index_map, reconstruct_path

__all__ = [
    "Graph",
    "WeightedGraph",
    "bfs",
    "dfs_recursive",
    "dfs_iterative",
    "dijkstra",
    "bellman_ford",
    "floyd_warshall",
    "kruskal_mst",
    "prim_mst",
    "pagerank",
    "graph_laplacian_matrix",
    "spectral_clustering",
    "node_index_map",
    "edges_from_weighted_graph",
    "reconstruct_path",
]

# Example usage:
# from qconduit.graphs import WeightedGraph, dijkstra, reconstruct_path
#
# G = WeightedGraph(directed=True)
# G.add_edge('A', 'B', 1.0)
# G.add_edge('B', 'C', 2.0)
# dist, parent = dijkstra(G, 'A')
# path = reconstruct_path(parent, 'C')  # ['A', 'B', 'C']

