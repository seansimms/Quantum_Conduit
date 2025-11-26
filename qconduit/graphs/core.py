"""
Core graph data structures.

Provides Graph (unweighted) and WeightedGraph classes with adjacency-list
representations. Operations are O(1) amortized for adding nodes/edges.
Neighbors are returned in sorted order for deterministic behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Set, Tuple


@dataclass
class Graph:
    """
    Unweighted graph with adjacency-list representation.

    Supports directed and undirected graphs. Neighbors are returned in
    sorted order for deterministic behavior.

    Attributes:
        directed: If True, graph is directed; otherwise undirected.
        adj: Adjacency list mapping node -> list of neighbors.

    Complexity:
        - add_node: O(1) amortized
        - add_edge: O(1) amortized
        - neighbors: O(deg(v)) where deg(v) is degree of node v
        - nodes: O(V) where V is number of nodes
        - edges: O(E) where E is number of edges
    """

    directed: bool = False
    adj: Dict[Hashable, List[Hashable]] = field(default_factory=dict)

    def __init__(self, directed: bool = False):
        """
        Initialize an empty graph.

        Args:
            directed: If True, graph is directed; otherwise undirected.
        """
        self.directed = directed
        self.adj = {}

    def add_node(self, node: Hashable) -> None:
        """
        Add a node to the graph.

        Args:
            node: Hashable node identifier.
        """
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, u: Hashable, v: Hashable) -> None:
        """
        Add an edge from u to v.

        For undirected graphs, also adds edge from v to u.

        Args:
            u: Source node.
            v: Target node.
        """
        self.add_node(u)
        self.add_node(v)

        if v not in self.adj[u]:
            self.adj[u].append(v)

        if not self.directed and u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self) -> List[Hashable]:
        """
        Return list of all nodes in sorted order.

        Returns:
            Sorted list of nodes.
        """
        return sorted(self.adj.keys(), key=lambda x: str(x))

    def neighbors(self, node: Hashable) -> List[Hashable]:
        """
        Return neighbors of a node in sorted order.

        Args:
            node: Node to get neighbors for.

        Returns:
            Sorted list of neighbors.

        Raises:
            KeyError: If node is not in graph.
        """
        if node not in self.adj:
            raise KeyError(f"Node {node} not in graph")
        return sorted(self.adj[node], key=lambda x: str(x))

    def edges(self) -> List[Tuple[Hashable, Hashable]]:
        """
        Return list of all edges.

        For undirected graphs, each edge appears once (u, v) with u < v
        in string ordering.

        Returns:
            List of (u, v) tuples.
        """
        edges_list = []
        seen: Set[Tuple[Hashable, Hashable]] = set()

        for u in sorted(self.adj.keys(), key=lambda x: str(x)):
            for v in sorted(self.adj[u], key=lambda x: str(x)):
                if self.directed:
                    edges_list.append((u, v))
                else:
                    # For undirected, only include each edge once
                    edge = (u, v) if str(u) < str(v) else (v, u)
                    if edge not in seen:
                        seen.add(edge)
                        edges_list.append(edge)

        return edges_list


@dataclass
class WeightedGraph:
    """
    Weighted graph with adjacency-list representation.

    Supports directed and undirected graphs. Neighbors are returned in
    sorted order for deterministic behavior.

    Attributes:
        directed: If True, graph is directed; otherwise undirected.
        adj: Adjacency list mapping node -> list of (neighbor, weight) tuples.

    Complexity:
        - add_node: O(1) amortized
        - add_edge: O(1) amortized
        - neighbors: O(deg(v)) where deg(v) is degree of node v
        - nodes: O(V) where V is number of nodes
        - edges: O(E) where E is number of edges
    """

    directed: bool = False
    adj: Dict[Hashable, List[Tuple[Hashable, float]]] = field(default_factory=dict)

    def __init__(self, directed: bool = False):
        """
        Initialize an empty weighted graph.

        Args:
            directed: If True, graph is directed; otherwise undirected.
        """
        self.directed = directed
        self.adj = {}

    def add_node(self, node: Hashable) -> None:
        """
        Add a node to the graph.

        Args:
            node: Hashable node identifier.
        """
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, u: Hashable, v: Hashable, weight: float = 1.0) -> None:
        """
        Add a weighted edge from u to v.

        For undirected graphs, also adds edge from v to u with same weight.

        Args:
            u: Source node.
            v: Target node.
            weight: Edge weight (default 1.0).
        """
        self.add_node(u)
        self.add_node(v)

        # Check if edge already exists and update weight
        neighbors_u = [n for n, _ in self.adj[u]]
        if v not in neighbors_u:
            self.adj[u].append((v, weight))
        else:
            # Update existing edge weight
            self.adj[u] = [(n, w if n != v else weight) for n, w in self.adj[u]]

        if not self.directed:
            neighbors_v = [n for n, _ in self.adj[v]]
            if u not in neighbors_v:
                self.adj[v].append((u, weight))
            else:
                # Update existing edge weight
                self.adj[v] = [(n, w if n != u else weight) for n, w in self.adj[v]]

    def neighbors(self, node: Hashable) -> List[Tuple[Hashable, float]]:
        """
        Return neighbors of a node with weights in sorted order.

        Args:
            node: Node to get neighbors for.

        Returns:
            Sorted list of (neighbor, weight) tuples.

        Raises:
            KeyError: If node is not in graph.
        """
        if node not in self.adj:
            raise KeyError(f"Node {node} not in graph")
        return sorted(self.adj[node], key=lambda x: (str(x[0]), x[1]))

    def nodes(self) -> List[Hashable]:
        """
        Return list of all nodes in sorted order.

        Returns:
            Sorted list of nodes.
        """
        return sorted(self.adj.keys(), key=lambda x: str(x))

    def edges(self) -> List[Tuple[Hashable, Hashable, float]]:
        """
        Return list of all edges with weights.

        For undirected graphs, each edge appears once (u, v, w) with u < v
        in string ordering.

        Returns:
            List of (u, v, weight) tuples.
        """
        edges_list = []
        seen: Set[Tuple[Hashable, Hashable]] = set()

        for u in sorted(self.adj.keys(), key=lambda x: str(x)):
            for v, weight in sorted(self.adj[u], key=lambda x: (str(x[0]), x[1])):
                if self.directed:
                    edges_list.append((u, v, weight))
                else:
                    # For undirected, only include each edge once
                    edge = (u, v) if str(u) < str(v) else (v, u)
                    if edge not in seen:
                        seen.add(edge)
                        edges_list.append((u, v, weight))

        return edges_list

