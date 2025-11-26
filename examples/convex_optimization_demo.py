"""
Example: Convex Optimization in Quantum Conduit

This example demonstrates the convex optimization module (G26) integrated
into the Quantum Conduit package. It shows how to use linear programming,
quadratic programming, and interior-point methods for optimization tasks
that might arise in quantum algorithm design and resource allocation.
"""

import numpy as np

from qconduit import (
    Status,
    active_set_qp,
    is_kkt_optimal,
    kkt_residuals,
    log_barrier_method,
    projected_gradient,
    simplex,
)


def example_linear_programming():
    """Example: Resource allocation with linear constraints."""
    print("=" * 60)
    print("Example 1: Linear Programming - Resource Allocation")
    print("=" * 60)

    # Maximize profit: 3x + 5y
    # Subject to: x + 2y <= 4, 3x + 2y <= 6, x >= 0, y >= 0
    c = np.array([-3.0, -5.0])  # Negative for minimization
    G = np.array([[1.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])

    result = simplex(c, a_mat=None, b_vec=None, g_mat=G, h_vec=h)
    print(f"Status: {result.status}")
    if result.status == Status.OPTIMAL:
        print(f"Optimal solution: x = {result.x}")
        print(f"Optimal value: {result.fun}")
        print(f"Iterations: {result.nit}")
    print()


def example_quadratic_programming():
    """Example: Portfolio optimization (convex QP)."""
    print("=" * 60)
    print("Example 2: Quadratic Programming - Portfolio Optimization")
    print("=" * 60)

    # Minimize risk: 0.5 * x^T H x
    # Subject to: sum(x) = 1 (budget constraint), x >= 0
    n = 3
    H = np.eye(n)  # Identity covariance (simplified)
    g = np.zeros(n)
    A = np.ones((1, n))  # Budget constraint
    b = np.array([1.0])
    lb = np.zeros(n)

    result = active_set_qp(H, g, a_mat=A, b_vec=b, lb=lb)
    print(f"Status: {result.status}")
    if result.status == Status.OPTIMAL and result.x is not None:
        print(f"Optimal portfolio: {result.x}")
        print(f"Risk (objective): {result.fun}")
        print(f"Iterations: {result.nit}")

        # Validate with KKT conditions
        residuals = kkt_residuals(H, g, A, b, None, None, result.x)
        is_optimal = is_kkt_optimal(H, g, A, b, None, None, result.x, tol=1e-6)
        print(f"KKT optimal: {is_optimal}")
        print(f"Primal residual: {residuals['primal_eq']:.2e}")
        print(f"Dual residual: {residuals['dual']:.2e}")
    print()


def example_interior_point():
    """Example: Interior-point method for LP."""
    print("=" * 60)
    print("Example 3: Interior-Point Method (Log-Barrier)")
    print("=" * 60)

    c = np.array([-2.0, -3.0])
    G = np.array([[1.0, 1.0], [2.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([4.0, 6.0, 0.0, 0.0])

    result = log_barrier_method(c, G, h, mu0=10.0, maxiter=50)
    print(f"Status: {result.status}")
    if result.status in {Status.OPTIMAL, Status.MAX_ITER} and result.x is not None:
        print(f"Solution: x = {result.x}")
        print(f"Objective: {result.fun}")
        print(f"Iterations: {result.nit}")
        if result.primal_residual is not None:
            print(f"Primal residual: {result.primal_residual:.2e}")
    print()


def example_projected_gradient():
    """Example: Projected gradient for box-constrained optimization."""
    print("=" * 60)
    print("Example 4: Projected Gradient - Box Constraints")
    print("=" * 60)

    # Minimize ||x - target||^2 subject to 0 <= x <= 1
    target = np.array([0.8, -0.2, 1.2])

    def obj(x: np.ndarray) -> float:
        return 0.5 * np.sum((x - target) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        return x - target

    def proj(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    result = projected_gradient(obj, grad, proj, x0=np.zeros(3), lr=0.5, maxiter=200)
    print(f"Status: {result.status}")
    if result.status == Status.OPTIMAL and result.x is not None:
        print(f"Optimal point: {result.x}")
        print(f"Objective: {result.fun}")
        print(f"Iterations: {result.nit}")
        print(f"Expected (clipped target): {proj(target)}")
    print()


def example_workflow():
    """Example: Multi-step optimization workflow."""
    print("=" * 60)
    print("Example 5: Multi-Step Optimization Workflow")
    print("=" * 60)

    # Step 1: Find feasible region via LP
    print("Step 1: Finding feasible region...")
    c_feas = np.array([0.0, 0.0])  # Feasibility problem
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([2.0, 0.0, 0.0])
    lp_result = simplex(c_feas, None, None, G, h)

    if lp_result.status == Status.OPTIMAL and lp_result.x is not None:
        print(f"  Feasible point found: {lp_result.x}")

        # Step 2: Optimize quadratic objective in feasible region
        print("Step 2: Optimizing quadratic objective...")
        H = np.eye(2)
        g = np.array([-1.0, -1.0])
        qp_result = active_set_qp(H, g, lb=np.zeros(2), ub=lp_result.x + 0.1)

        if qp_result.status == Status.OPTIMAL and qp_result.x is not None:
            print(f"  QP solution: {qp_result.x}")
            print(f"  QP objective: {qp_result.fun}")

            # Step 3: Validate with KKT
            print("Step 3: Validating with KKT conditions...")
            residuals = kkt_residuals(H, g, None, None, None, None, qp_result.x)
            is_opt = is_kkt_optimal(H, g, None, None, None, None, qp_result.x)
            print(f"  KKT optimal: {is_opt}")
            print(f"  Dual residual: {residuals['dual']:.2e}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Quantum Conduit - Convex Optimization Examples (G26)")
    print("=" * 60 + "\n")

    example_linear_programming()
    example_quadratic_programming()
    example_interior_point()
    example_projected_gradient()
    example_workflow()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

