"""
Spectral graph methods: Laplacian matrix and spectral clustering.

Computes graph Laplacian and performs basic spectral clustering using
eigenvalue decomposition and k-means.

References:
    - Golub, G. H., Van Loan, C. F. "Matrix Computations", 4th ed.
    - Von Luxburg, U. "A Tutorial on Spectral Clustering" (2007).
"""

from typing import Dict, Hashable, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .core import WeightedGraph
from .utils import node_index_map


def graph_laplacian_matrix(
    graph: WeightedGraph, nodes: Optional[List[Hashable]] = None
) -> "np.ndarray":
    """
    Compute graph Laplacian matrix L = D - W.

    For weighted graphs, W[i,j] = weight of edge i->j (symmetric for undirected).
    D is diagonal degree matrix where D[i,i] = sum of weights of edges incident to i.

    Args:
        graph: WeightedGraph instance.
        nodes: Optional list of nodes to include (defaults to all nodes in sorted order).

    Returns:
        (n, n) numpy array representing Laplacian matrix in node index order.

    Raises:
        ImportError: If numpy is not available.

    Example:
        >>> G = WeightedGraph()
        >>> G.add_edge('A', 'B', 1.0)
        >>> L = graph_laplacian_matrix(G)
        >>> L.shape
        (2, 2)
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for graph_laplacian_matrix")

    if nodes is None:
        nodes = graph.nodes()
    else:
        # Ensure deterministic ordering
        nodes = sorted(set(nodes), key=lambda x: str(x))

    node_to_idx, idx_to_node = node_index_map(nodes)
    n = len(nodes)

    # Initialize adjacency/weight matrix W
    W = np.zeros((n, n))

    # Build weight matrix
    for u in nodes:
        i = node_to_idx[u]
        for v, weight in graph.neighbors(u):
            if v in node_to_idx:
                j = node_to_idx[v]
                W[i, j] = weight
                # For undirected graphs, ensure symmetry
                if not graph.directed:
                    W[j, i] = weight

    # Compute degree matrix D (diagonal)
    D = np.diag(W.sum(axis=1))

    # Laplacian L = D - W
    L = D - W

    return L


def spectral_clustering(
    graph: WeightedGraph, k: int, normalized: bool = True
) -> Dict[Hashable, int]:
    """
    Perform spectral clustering by computing k smallest eigenvectors of Laplacian.

    Uses k-means on the rows of the eigenvector matrix to assign clusters.

    Args:
        graph: WeightedGraph instance (undirected recommended).
        k: Number of clusters.
        normalized: If True, use normalized Laplacian L_norm = D^(-1/2) L D^(-1/2).

    Returns:
        Dictionary mapping node -> cluster_id (0 to k-1).

    Raises:
        ImportError: If numpy is not available.
        ValueError: If k < 1 or k > number of nodes.

    Complexity: O(n^3) for eigenvalue decomposition, O(n*k*iterations) for k-means.

    Example:
        >>> G = WeightedGraph()
        >>> # Create two disconnected components
        >>> G.add_edge('A', 'B', 1.0)
        >>> G.add_edge('C', 'D', 1.0)
        >>> clusters = spectral_clustering(G, k=2)
        >>> # A and B should be in same cluster, C and D in another
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for spectral_clustering")

    nodes = graph.nodes()
    n = len(nodes)

    if n == 0:
        return {}

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k must be <= number of nodes ({n}), got {k}")

    if n == 1:
        return {nodes[0]: 0}

    # Compute Laplacian
    L = graph_laplacian_matrix(graph, nodes)

    if normalized:
        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        D = np.diag(L.sum(axis=1))
        # Avoid division by zero for isolated nodes
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Get k smallest eigenvectors (excluding first if normalized, as it's constant)
    # For normalized Laplacian, first eigenvalue is 0 with constant eigenvector
    start_idx = 1 if normalized else 0
    k_actual = min(k, n - start_idx)

    if k_actual < k:
        # Not enough eigenvectors, assign all to cluster 0
        return {node: 0 for node in nodes}

    # Extract k smallest eigenvectors (columns of eigenvector matrix)
    eigenvecs_k = eigenvectors[:, start_idx : start_idx + k_actual]

    # Normalize rows to unit length
    row_norms = np.linalg.norm(eigenvecs_k, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)  # Avoid division by zero
    eigenvecs_k_normalized = eigenvecs_k / row_norms

    # Apply k-means clustering to rows
    node_to_idx, idx_to_node = node_index_map(nodes)
    assignments = _kmeans_deterministic(eigenvecs_k_normalized, k_actual)

    # Map back to nodes
    result = {idx_to_node[i]: int(assignments[i]) for i in range(n)}
    return result


def _kmeans_deterministic(data: "np.ndarray", k: int, max_iter: int = 100) -> "np.ndarray":
    """
    Deterministic k-means clustering.

    Uses deterministic initialization: first k points in sorted order by first dimension.

    Args:
        data: (n, d) array of n data points in d dimensions.
        k: Number of clusters.
        max_iter: Maximum iterations.

    Returns:
        (n,) array of cluster assignments (0 to k-1).
    """
    n, d = data.shape

    if k >= n:
        return np.arange(n)

    # Deterministic initialization: sort by first dimension, take first k
    sorted_indices = np.argsort(data[:, 0])
    centroids = data[sorted_indices[:k]].copy()

    assignments = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        # Assign points to nearest centroid
        diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        new_assignments = np.argmin(distances, axis=1)

        # Check convergence
        if np.array_equal(new_assignments, assignments):
            break

        assignments = new_assignments

        # Update centroids
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = data[mask].mean(axis=0)

    return assignments

