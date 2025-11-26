"""Deterministic line-search routines following Nocedal & Wright."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .core import Array, Gradient, Objective


def backtracking_armijo(
    f: Objective,
    x: Array,
    p: Array,
    grad_fx: Array,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50,
) -> tuple[float, int]:
    """Classic Armijo backtracking line search."""
    if not (0 < c < 1):
        raise ValueError("Armijo constant c must lie in (0, 1)")
    if not (0 < rho < 1):
        raise ValueError("rho must lie in (0, 1)")
    alpha = float(alpha0)
    fx = f(x)
    grad_dot = float(np.dot(grad_fx, p))
    nfev = 0
    for _ in range(max_iter):
        candidate = x + alpha * p
        f_new = f(candidate)
        nfev += 1
        if f_new <= fx + c * alpha * grad_dot:
            return alpha, nfev
        alpha *= rho
    return alpha, nfev


def wolfe_line_search(
    f: Objective,
    grad: Gradient,
    x: Array,
    p: Array,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 40,
) -> tuple[float, int]:
    """Perform a strong Wolfe line search using bracketing and zoom."""
    if not (0 < c1 < c2 < 1):
        raise ValueError("Require 0 < c1 < c2 < 1 for Wolfe conditions.")

    nfev = 0

    def phi(alpha: float) -> float:
        nonlocal nfev
        nfev += 1
        return f(x + alpha * p)

    def phi_prime(alpha: float) -> float:
        return float(np.dot(grad(x + alpha * p), p))

    alpha_prev = 0.0
    phi0 = phi(0.0)
    der0 = phi_prime(0.0)
    if der0 >= 0:
        raise ValueError("Search direction must be a descent direction.")
    alpha = float(alpha0)
    phi_prev = phi0

    for iteration in range(max_iter):
        phi_alpha = phi(alpha)
        if phi_alpha > phi0 + c1 * alpha * der0 or (
            iteration > 0 and phi_alpha >= phi_prev
        ):
            alpha_zoom = _zoom(
                phi,
                phi_prime,
                alpha_prev,
                alpha,
                phi0,
                der0,
                c1,
                c2,
            )
            return alpha_zoom, nfev
        der_alpha = phi_prime(alpha)
        if abs(der_alpha) <= -c2 * der0:
            return alpha, nfev
        if der_alpha >= 0:
            alpha_zoom = _zoom(
                phi,
                phi_prime,
                alpha,
                alpha_prev,
                phi0,
                der0,
                c1,
                c2,
            )
            return alpha_zoom, nfev
        alpha_prev = alpha
        phi_prev = phi_alpha
        alpha *= 2.0
    return alpha, nfev


def _zoom(
    phi: Callable[[float], float],
    phi_prime: Callable[[float], float],
    alo: float,
    ahi: float,
    phi0: float,
    der0: float,
    c1: float,
    c2: float,
) -> float:
    """Zoom stage enforcing strong Wolfe conditions."""
    phi_alo = phi(alo)
    for _ in range(32):
        alpha = 0.5 * (alo + ahi)
        phi_alpha = phi(alpha)
        if phi_alpha > phi0 + c1 * alpha * der0 or phi_alpha >= phi_alo:
            ahi = alpha
        else:
            der_alpha = phi_prime(alpha)
            if abs(der_alpha) <= -c2 * der0:
                return alpha
            if der_alpha * (ahi - alo) > 0:
                ahi = alo
            alo = alpha
            phi_alo = phi_alpha
        if abs(ahi - alo) < 1e-12:
            break
    return alpha


__all__ = ["backtracking_armijo", "wolfe_line_search"]

