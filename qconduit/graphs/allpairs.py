"""
All-pairs shortest path algorithms: Floyd-Warshall.

Computes shortest paths between all pairs of nodes in a graph.

References:
    - Cormen, Leiserson, Rivest, Stein. "Introduction to Algorithms", 3rd ed.
      Chapter 25.2 (Floyd-Warshall).
"""

from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from .utils import node_index_map


def floyd_warshall(
    nodes: List[Hashable], edges: Iterable[Tuple[Hashable, Hashable, float]]
) -> Tuple[
    Dict[Tuple[Hashable, Hashable], float],
    Dict[Tuple[Hashable, Hashable], Optional[List[Hashable]]],
]:
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.

    Computes shortest distances and paths between all pairs of nodes.
    Handles negative edge weights but not negative cycles (distances may
    be incorrect if negative cycles exist).

    Args:
        nodes: List of all nodes in the graph.
        edges: Iterable of (u, v, weight) tuples.

    Returns:
        Tuple of:
        - dist: Dictionary mapping (u, v) -> shortest distance (float or inf)
        - path: Dictionary mapping (u, v) -> list of nodes on shortest path, or None if no path

    Complexity: O(n^3) where n is number of nodes.

    Example:
        >>> nodes = ['A', 'B', 'C']
        >>> edges = [('A', 'B', 1.0), ('B', 'C', 2.0)]
        >>> dist, path = floyd_warshall(nodes, edges)
        >>> dist[('A', 'C')]
        3.0
        >>> path[('A', 'C')]
        ['A', 'B', 'C']
    """
    # Create deterministic node index mapping
    node_to_idx, idx_to_node = node_index_map(nodes)
    n = len(node_to_idx)

    # Initialize distance matrix
    dist_matrix = [[float("inf")] * n for _ in range(n)]
    next_matrix: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    # Distance from node to itself is 0
    for i in range(n):
        dist_matrix[i][i] = 0.0
        next_matrix[i][i] = i

    # Initialize with direct edges
    for u, v, weight in edges:
        i = node_to_idx[u]
        j = node_to_idx[v]
        if weight < dist_matrix[i][j]:
            dist_matrix[i][j] = weight
            next_matrix[i][j] = j

    # Floyd-Warshall main loop
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][k] != float("inf") and dist_matrix[k][j] != float("inf"):
                    new_dist = dist_matrix[i][k] + dist_matrix[k][j]
                    if new_dist < dist_matrix[i][j]:
                        dist_matrix[i][j] = new_dist
                        next_matrix[i][j] = next_matrix[i][k]

    # Build result dictionaries
    dist: Dict[Tuple[Hashable, Hashable], float] = {}
    path: Dict[Tuple[Hashable, Hashable], Optional[List[Hashable]]] = {}

    def reconstruct_path_internal(i: int, j: int) -> Optional[List[Hashable]]:
        """Reconstruct path from i to j using next_matrix."""
        if dist_matrix[i][j] == float("inf"):
            return None
        if next_matrix[i][j] is None:
            return None

        path_list = [idx_to_node[i]]
        current = i
        while current != j:
            if next_matrix[current][j] is None:
                return None
            current = next_matrix[current][j]
            path_list.append(idx_to_node[current])
        return path_list

    for i in range(n):
        for j in range(n):
            u = idx_to_node[i]
            v = idx_to_node[j]
            dist[(u, v)] = dist_matrix[i][j]
            path[(u, v)] = reconstruct_path_internal(i, j)

    return dist, path

