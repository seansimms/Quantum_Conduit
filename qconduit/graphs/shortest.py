"""
Shortest path algorithms: Dijkstra and Bellman-Ford.

Dijkstra's algorithm for non-negative edge weights.
Bellman-Ford algorithm for graphs with negative weights (detects negative cycles).

References:
    - Cormen, Leiserson, Rivest, Stein. "Introduction to Algorithms", 3rd ed.
      Chapters 24.3 (Dijkstra) and 24.1 (Bellman-Ford).
"""

import heapq
from typing import Dict, Hashable, List, Optional, Tuple

from .core import WeightedGraph


def dijkstra(
    graph: WeightedGraph, source: Hashable
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Optional[Hashable]]]:
    """
    Dijkstra's algorithm for single-source shortest paths.

    Computes shortest paths from source to all reachable nodes in a graph
    with non-negative edge weights.

    Args:
        graph: WeightedGraph with non-negative edge weights.
        source: Source node.

    Returns:
        Tuple of:
        - dist: Dictionary mapping node -> shortest distance from source (float or inf)
        - parent: Dictionary mapping node -> previous node on shortest path (None if unreachable)

    Raises:
        ValueError: If source is not in graph.
        ValueError: If graph contains negative edge weights.

    Complexity: O(E log V) using binary heap priority queue.

    Example:
        >>> G = WeightedGraph(directed=True)
        >>> G.add_edge('A', 'B', 1.0)
        >>> G.add_edge('B', 'C', 2.0)
        >>> dist, parent = dijkstra(G, 'A')
        >>> dist['C']
        3.0
    """
    if source not in graph.adj:
        raise ValueError(f"Source node {source} not in graph")

    # Check for negative weights
    for u in graph.nodes():
        for v, weight in graph.neighbors(u):
            if weight < 0:
                raise ValueError(
                    f"Dijkstra requires non-negative weights. "
                    f"Found negative weight {weight} on edge ({u}, {v})"
                )

    dist: Dict[Hashable, float] = {}
    parent: Dict[Hashable, Optional[Hashable]] = {}

    # Initialize distances
    for node in graph.nodes():
        dist[node] = float("inf")
        parent[node] = None

    dist[source] = 0.0
    parent[source] = None

    # Priority queue: (distance, node_str, node) for deterministic tie-breaking
    pq: List[Tuple[float, str, Hashable]] = [(0.0, str(source), source)]
    visited: set = set()

    while pq:
        d, _, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)

        # Relax edges from u
        for v, weight in graph.neighbors(u):
            if v in visited:
                continue

            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                # Use (dist, str(node), node) for deterministic tie-breaking
                heapq.heappush(pq, (new_dist, str(v), v))

    return dist, parent


def bellman_ford(
    graph: WeightedGraph, source: Hashable
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Optional[Hashable]], bool]:
    """
    Bellman-Ford algorithm for single-source shortest paths.

    Computes shortest paths from source to all reachable nodes, allowing
    negative edge weights. Detects negative cycles reachable from source.

    Args:
        graph: WeightedGraph (may have negative weights).
        source: Source node.

    Returns:
        Tuple of:
        - dist: Dictionary mapping node -> shortest distance from source (float or inf)
        - parent: Dictionary mapping node -> previous node on shortest path (None if unreachable)
        - has_negative_cycle: True if negative cycle reachable from source is detected

    Raises:
        ValueError: If source is not in graph.

    Complexity: O(VE) where V is vertices and E is edges.

    Example:
        >>> G = WeightedGraph(directed=True)
        >>> G.add_edge('A', 'B', 1.0)
        >>> G.add_edge('B', 'C', -2.0)
        >>> dist, parent, has_cycle = bellman_ford(G, 'A')
        >>> has_cycle
        False
        >>> dist['C']
        -1.0
    """
    if source not in graph.adj:
        raise ValueError(f"Source node {source} not in graph")

    dist: Dict[Hashable, float] = {}
    parent: Dict[Hashable, Optional[Hashable]] = {}

    # Initialize distances
    for node in graph.nodes():
        dist[node] = float("inf")
        parent[node] = None

    dist[source] = 0.0
    parent[source] = None

    # Get all edges in deterministic order
    edges = list(graph.edges())
    n = len(graph.nodes())

    # Relax edges n-1 times
    for _ in range(n - 1):
        for u, v, weight in edges:
            if dist[u] != float("inf") and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                parent[v] = u

    # Check for negative cycles
    has_negative_cycle = False
    for u, v, weight in edges:
        if dist[u] != float("inf") and dist[u] + weight < dist[v]:
            has_negative_cycle = True
            break

    return dist, parent, has_negative_cycle

