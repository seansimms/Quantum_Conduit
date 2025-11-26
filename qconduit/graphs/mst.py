"""
Minimum spanning tree algorithms: Kruskal and Prim.

Kruskal uses union-find data structure. Prim uses priority queue.

References:
    - Cormen, Leiserson, Rivest, Stein. "Introduction to Algorithms", 3rd ed.
      Chapters 23.1 (MST properties), 23.2 (Kruskal), 23.2 (Prim).
"""

import heapq
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from .core import WeightedGraph


class UnionFind:
    """
    Union-Find (Disjoint Set) data structure with path compression and union by rank.

    Used by Kruskal's algorithm for efficient cycle detection.
    """

    def __init__(self, nodes: Iterable[Hashable]):
        """
        Initialize union-find with given nodes.

        Args:
            nodes: Iterable of nodes.
        """
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}

        for node in nodes:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, x: Hashable) -> Hashable:
        """
        Find root of x with path compression.

        Args:
            x: Node to find root for.

        Returns:
            Root node.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Hashable, y: Hashable) -> bool:
        """
        Union sets containing x and y using union by rank.

        Args:
            x: First node.
            y: Second node.

        Returns:
            True if union was performed (x and y were in different sets),
            False if they were already in the same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


def kruskal_mst(
    nodes: Iterable[Hashable], edges: Iterable[Tuple[Hashable, Hashable, float]]
) -> List[Tuple[Hashable, Hashable, float]]:
    """
    Kruskal's algorithm for minimum spanning tree.

    Returns edges in the MST sorted deterministically (by weight, then nodes).

    Args:
        nodes: Iterable of all nodes in the graph.
        edges: Iterable of (u, v, weight) tuples.

    Returns:
        List of (u, v, weight) edges in the MST, sorted deterministically.
        For disconnected graphs, returns MST forest (one MST per component).

    Complexity: O(E log E) = O(E log V) for sorting and union-find operations.

    Example:
        >>> nodes = ['A', 'B', 'C']
        >>> edges = [('A', 'B', 1.0), ('B', 'C', 2.0), ('A', 'C', 3.0)]
        >>> mst = kruskal_mst(nodes, edges)
        >>> len(mst)
        2
    """
    # Convert to list and sort deterministically
    edge_list = list(edges)
    # Sort by (weight, str(u), str(v)) for deterministic tie-breaking
    edge_list.sort(key=lambda x: (x[2], str(x[0]), str(x[1])))

    uf = UnionFind(nodes)
    mst_edges: List[Tuple[Hashable, Hashable, float]] = []

    for u, v, weight in edge_list:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))

    # Sort result for deterministic output
    mst_edges.sort(key=lambda x: (x[2], str(x[0]), str(x[1])))
    return mst_edges


def prim_mst(
    graph: WeightedGraph, start: Optional[Hashable] = None
) -> List[Tuple[Hashable, Hashable, float]]:
    """
    Prim's algorithm for minimum spanning tree.

    Uses priority queue to greedily add minimum-weight edges connecting
    tree to non-tree vertices.

    Args:
        graph: WeightedGraph (undirected).
        start: Starting node (defaults to first node in sorted order).

    Returns:
        List of (u, v, weight) edges in the MST, sorted deterministically.
        For disconnected graphs, returns MST forest (one MST per component).

    Complexity: O(E log V) using binary heap.

    Example:
        >>> G = WeightedGraph()
        >>> G.add_edge('A', 'B', 1.0)
        >>> G.add_edge('B', 'C', 2.0)
        >>> mst = prim_mst(G, 'A')
        >>> len(mst)
        2
    """
    nodes = graph.nodes()
    if not nodes:
        return []

    if start is None:
        start = nodes[0]
    elif start not in graph.adj:
        raise ValueError(f"Start node {start} not in graph")

    mst_edges: List[Tuple[Hashable, Hashable, float]] = []
    in_mst: set = set()
    # Priority queue: (weight, str(u), str(v), u, v) for deterministic tie-breaking
    pq: List[Tuple[float, str, str, Hashable, Hashable]] = []

    # Start with first node
    in_mst.add(start)

    # Add edges from start to priority queue
    for v, weight in graph.neighbors(start):
        heapq.heappush(pq, (weight, str(start), str(v), start, v))

    while pq and len(in_mst) < len(nodes):
        weight, _, _, u, v = heapq.heappop(pq)

        # Skip if both nodes already in MST
        if u in in_mst and v in in_mst:
            continue

        # Determine which node is new
        if u in in_mst:
            new_node = v
            mst_edge = (u, v, weight)
        else:
            new_node = u
            mst_edge = (v, u, weight)

        if new_node not in in_mst:
            in_mst.add(new_node)
            mst_edges.append(mst_edge)

            # Add edges from new_node to priority queue
            for neighbor, edge_weight in graph.neighbors(new_node):
                if neighbor not in in_mst:
                    heapq.heappush(
                        pq, (edge_weight, str(new_node), str(neighbor), new_node, neighbor)
                    )

    # Sort result for deterministic output
    mst_edges.sort(key=lambda x: (x[2], str(x[0]), str(x[1])))
    return mst_edges

