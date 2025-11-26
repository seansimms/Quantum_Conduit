"""
Graph traversal algorithms: BFS and DFS.

Provides breadth-first and depth-first search with deterministic ordering.
Neighbors are visited in sorted order for reproducible results.

References:
    - Cormen, Leiserson, Rivest, Stein. "Introduction to Algorithms", 3rd ed.
      Chapters 22.2 (BFS) and 22.3 (DFS).
"""

from collections import deque
from typing import Dict, Hashable, List, Optional, Tuple

from .core import Graph


def bfs(
    graph: Graph, source: Hashable
) -> Tuple[List[Hashable], Dict[Hashable, int], Dict[Hashable, Optional[Hashable]]]:
    """
    Breadth-first search from a source node.

    Returns nodes in BFS visitation order, distances from source, and parent
    map for path reconstruction.

    Args:
        graph: Graph to traverse.
        source: Source node to start BFS from.

    Returns:
        Tuple of:
        - order: List of nodes in BFS visitation order
        - distance: Dictionary mapping node -> distance from source (int or inf)
        - parent: Dictionary mapping node -> parent node (None for source/unreached)

    Complexity: O(V + E) where V is vertices and E is edges.

    Example:
        >>> G = Graph()
        >>> G.add_edge('A', 'B')
        >>> G.add_edge('A', 'C')
        >>> order, dist, parent = bfs(G, 'A')
        >>> order
        ['A', 'B', 'C']
        >>> dist['B']
        1
    """
    if source not in graph.adj:
        return [], {}, {}

    order: List[Hashable] = []
    distance: Dict[Hashable, int] = {}
    parent: Dict[Hashable, Optional[Hashable]] = {}

    # Initialize distances to infinity for all nodes
    for node in graph.nodes():
        distance[node] = float("inf")
        parent[node] = None

    distance[source] = 0
    parent[source] = None
    queue = deque([source])

    while queue:
        u = queue.popleft()
        order.append(u)

        # Visit neighbors in sorted order for determinism
        neighbors = graph.neighbors(u)
        for v in neighbors:
            if distance[v] == float("inf"):
                distance[v] = distance[u] + 1
                parent[v] = u
                queue.append(v)

    return order, distance, parent


def dfs_recursive(
    graph: Graph, source: Hashable
) -> Tuple[List[Hashable], List[Hashable], Dict[Hashable, Optional[Hashable]]]:
    """
    Depth-first search (recursive implementation).

    Returns pre-order and post-order visitation lists, plus parent map.

    Args:
        graph: Graph to traverse.
        source: Source node to start DFS from.

    Returns:
        Tuple of:
        - preorder: List of nodes in pre-order (when first discovered)
        - postorder: List of nodes in post-order (when finished exploring)
        - parent: Dictionary mapping node -> parent node (None for source/unreached)

    Complexity: O(V + E) where V is vertices and E is edges.

    Example:
        >>> G = Graph()
        >>> G.add_edge('A', 'B')
        >>> G.add_edge('A', 'C')
        >>> pre, post, parent = dfs_recursive(G, 'A')
    """
    if source not in graph.adj:
        return [], [], {}

    preorder: List[Hashable] = []
    postorder: List[Hashable] = []
    parent: Dict[Hashable, Optional[Hashable]] = {}
    visited: set = set()

    def dfs_visit(u: Hashable) -> None:
        visited.add(u)
        preorder.append(u)
        parent[u] = parent.get(u)  # Keep existing parent if set

        # Visit neighbors in sorted order for determinism
        neighbors = graph.neighbors(u)
        for v in neighbors:
            if v not in visited:
                parent[v] = u
                dfs_visit(v)

        postorder.append(u)

    # Initialize parent for source
    parent[source] = None
    dfs_visit(source)

    return preorder, postorder, parent


def dfs_iterative(
    graph: Graph, source: Hashable
) -> Tuple[List[Hashable], List[Hashable], Dict[Hashable, Optional[Hashable]]]:
    """
    Depth-first search (iterative implementation using stack).

    Returns pre-order and post-order visitation lists, plus parent map.
    Matches recursive DFS behavior.

    Args:
        graph: Graph to traverse.
        source: Source node to start DFS from.

    Returns:
        Tuple of:
        - preorder: List of nodes in pre-order (when first discovered)
        - postorder: List of nodes in post-order (when finished exploring)
        - parent: Dictionary mapping node -> parent node (None for source/unreached)

    Complexity: O(V + E) where V is vertices and E is edges.

    Example:
        >>> G = Graph()
        >>> G.add_edge('A', 'B')
        >>> G.add_edge('A', 'C')
        >>> pre, post, parent = dfs_iterative(G, 'A')
    """
    if source not in graph.adj:
        return [], [], {}

    preorder: List[Hashable] = []
    postorder: List[Hashable] = []
    parent: Dict[Hashable, Optional[Hashable]] = {source: None}
    visited: set = set()
    stack: List[Tuple[Hashable, bool]] = [(source, False)]  # (node, is_finished)

    while stack:
        u, is_finished = stack.pop()

        if is_finished:
            postorder.append(u)
        else:
            if u not in visited:
                visited.add(u)
                preorder.append(u)

                # Push finish marker
                stack.append((u, True))

                # Push neighbors in reverse sorted order (to visit in sorted order)
                neighbors = graph.neighbors(u)
                for v in reversed(neighbors):
                    if v not in visited:
                        parent[v] = u
                        stack.append((v, False))

    return preorder, postorder, parent

