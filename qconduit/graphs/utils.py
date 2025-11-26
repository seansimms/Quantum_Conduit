"""
Utility functions for graph algorithms.

Provides helpers for node indexing, edge iteration, and path reconstruction.
"""

from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from .core import WeightedGraph


def node_index_map(nodes: Iterable[Hashable]) -> Tuple[Dict[Hashable, int], List[Hashable]]:
    """
    Create deterministic mapping from nodes to indices 0..n-1.

    Nodes are sorted by string representation for deterministic ordering.

    Args:
        nodes: Iterable of hashable nodes.

    Returns:
        Tuple of (node_to_index dict, index_to_node list).
        The list provides the node ordering used for indexing.

    Example:
        >>> node_to_idx, idx_to_node = node_index_map(['c', 'a', 'b'])
        >>> node_to_idx
        {'a': 0, 'b': 1, 'c': 2}
        >>> idx_to_node
        ['a', 'b', 'c']
    """
    sorted_nodes = sorted(set(nodes), key=lambda x: str(x))
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}
    return node_to_index, sorted_nodes


def edges_from_weighted_graph(graph: WeightedGraph) -> Iterable[Tuple[Hashable, Hashable, float]]:
    """
    Return iterator of (u, v, weight) edges with deterministic ordering.

    Edges are sorted by (str(u), str(v), weight) for reproducibility.

    Args:
        graph: WeightedGraph instance.

    Yields:
        (u, v, weight) tuples in deterministic order.
    """
    edges = graph.edges()
    # Sort by string representation of nodes, then weight
    sorted_edges = sorted(edges, key=lambda x: (str(x[0]), str(x[1]), x[2]))
    yield from sorted_edges


def reconstruct_path(
    parent: Dict[Hashable, Optional[Hashable]], target: Hashable
) -> Optional[List[Hashable]]:
    """
    Reconstruct path from source to target using parent map.

    The parent map should come from a shortest-path algorithm (e.g., BFS,
    Dijkstra) where parent[node] is the previous node on the shortest path,
    or None if node is unreachable or is the source.

    Args:
        parent: Dictionary mapping node -> parent node (or None).
        target: Target node to reconstruct path to.

    Returns:
        List of nodes from source to target (inclusive), or None if target
        is unreachable.

    Example:
        >>> parent = {'A': None, 'B': 'A', 'C': 'B'}
        >>> reconstruct_path(parent, 'C')
        ['A', 'B', 'C']
        >>> reconstruct_path(parent, 'D')
        None
    """
    if target not in parent or parent[target] is None:
        # Check if target is the source (parent[target] is None) or unreachable
        if target in parent and parent[target] is None:
            return [target]
        return None

    path = []
    current = target
    # Build path backwards, handling cycles (shouldn't happen in shortest paths)
    visited = set()
    while current is not None:
        if current in visited:
            # Cycle detected (shouldn't happen in valid parent maps)
            return None
        visited.add(current)
        path.append(current)
        current = parent.get(current)

    path.reverse()
    return path

