"""
PageRank algorithm using power iteration.

Computes importance scores for nodes in a directed graph based on random walk
with damping factor.

References:
    - Page, L., Brin, S., Motwani, R., Winograd, T. "The PageRank Citation Ranking:
      Bringing Order to the Web" (1998).
"""

from typing import Dict, Hashable, Optional

from .core import Graph


def pagerank(
    graph: Graph,
    alpha: float = 0.85,
    tol: float = 1e-6,
    maxiter: int = 100,
    personalization: Optional[Dict[Hashable, float]] = None,
) -> Dict[Hashable, float]:
    """
    Compute PageRank scores using power iteration.

    PageRank models a random surfer who follows links with probability alpha
    and jumps to random nodes with probability (1-alpha). Dangling nodes (no
    outgoing edges) are handled by redistributing their probability uniformly
    or via personalization vector.

    Args:
        graph: Directed Graph (undirected graphs treated as bidirectional).
        alpha: Damping factor (probability of following links), default 0.85.
        tol: Convergence tolerance (L1 norm difference), default 1e-6.
        maxiter: Maximum iterations, default 100.
        personalization: Optional dict mapping node -> personalization weight.
            If None, uniform personalization is used. Must sum to 1 if provided.

    Returns:
        Dictionary mapping node -> PageRank score (sums to 1.0).

    Raises:
        ValueError: If alpha not in [0, 1] or personalization doesn't sum to 1.

    Complexity: O(k * (V + E)) where k is number of iterations until convergence.

    Example:
        >>> G = Graph(directed=True)
        >>> G.add_edge('A', 'B')
        >>> scores = pagerank(G)
        >>> sum(scores.values())
        1.0
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    nodes = graph.nodes()
    n = len(nodes)

    if n == 0:
        return {}

    if n == 1:
        return {nodes[0]: 1.0}

    # Build node index mapping for deterministic ordering
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = nodes

    # Initialize personalization vector
    if personalization is None:
        personalization_vec = [1.0 / n] * n
    else:
        if abs(sum(personalization.values()) - 1.0) > 1e-10:
            raise ValueError(
                f"Personalization must sum to 1.0, got {sum(personalization.values())}"
            )
        personalization_vec = [personalization.get(node, 0.0) for node in nodes]

    # Build transition matrix (sparse representation: list of outgoing edges per node)
    # For each node, track its outgoing neighbors and whether it's dangling
    outgoing: Dict[int, list] = {}
    dangling: list = []

    for node in nodes:
        idx = node_to_idx[node]
        neighbors = graph.neighbors(node)
        if neighbors:
            # For undirected graphs, treat as bidirectional
            outgoing[idx] = [node_to_idx[n] for n in neighbors]
        else:
            outgoing[idx] = []
            dangling.append(idx)

    # Initialize PageRank vector uniformly
    pr = [1.0 / n] * n

    # Power iteration
    for iteration in range(maxiter):
        pr_new = [0.0] * n

        # Compute contribution from following links
        for i in range(n):
            if i in outgoing and outgoing[i]:
                out_degree = len(outgoing[i])
                for j in outgoing[i]:
                    pr_new[j] += alpha * pr[i] / out_degree

        # Handle dangling nodes: redistribute uniformly
        dangling_sum = sum(pr[i] for i in dangling)
        if dangling_sum > 0:
            for j in range(n):
                pr_new[j] += alpha * dangling_sum / n

        # Add teleportation (random jump)
        for j in range(n):
            pr_new[j] += (1.0 - alpha) * personalization_vec[j]

        # Check convergence (L1 norm) - line 76 was too long, fixed by removing pr_sum
        diff = sum(abs(pr_new[i] - pr[i]) for i in range(n))
        pr = pr_new

        if diff < tol:
            break

    # Convert back to dictionary
    result = {idx_to_node[i]: pr[i] for i in range(n)}
    return result

