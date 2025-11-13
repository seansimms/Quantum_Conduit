"""Fermionic operator primitives for second-quantized fermionic operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

# Type alias for a single ladder operator symbol
FermionOpSymbol = Tuple[int, str]
"""
A single ladder operator symbol: (mode_index, op_type).

- mode_index: int - 0-based index of the spin-orbital (mode).
- op_type: str - "+" for creation a^\\dagger, "-" for annihilation a.
"""


@dataclass(frozen=True)
class FermionTerm:
    """
    Single term in a second-quantized fermionic operator.

    Each term has the form:

        coeff * a_{p1}^{σ1} a_{p2}^{σ2} ... a_{pk}^{σk},

    where σj is either "+" or "-" and the sequence of operators is ordered
    exactly as stored. No automatic anti-commutation or normal-ordering
    is performed by this class.

    Attributes
    ----------
    coeff:
        Complex coefficient multiplying the ladder operator string.
    operators:
        Tuple of (mode_index, op_type) pairs, where op_type is "+" (creation)
        or "-" (annihilation), and mode_index >= 0.
    """

    coeff: complex
    operators: Tuple[FermionOpSymbol, ...]

    def __post_init__(self) -> None:
        """Validate FermionTerm invariants."""
        # Normalize coeff to complex
        object.__setattr__(self, "coeff", complex(self.coeff))

        # Convert operators to tuple if needed
        if not isinstance(self.operators, tuple):
            object.__setattr__(self, "operators", tuple(self.operators))

        # Validate all modes are non-negative
        invalid_modes = [mode for mode, _ in self.operators if mode < 0]
        if invalid_modes:
            raise ValueError(
                f"All mode indices must be >= 0, got invalid modes: {invalid_modes}"
            )

        # Validate all operator types are valid
        invalid_ops = [op for _, op in self.operators if op not in {"+", "-"}]
        if invalid_ops:
            raise ValueError(
                f"All operator types must be '+' (creation) or '-' (annihilation), "
                f"got invalid types: {invalid_ops}"
            )

    def is_vacuum_term(self) -> bool:
        """Return True if this term has no ladder operators (pure scalar)."""
        return len(self.operators) == 0


@dataclass(frozen=True)
class FermionOperator:
    """
    Sum of FermionTerm objects representing a fermionic operator:

        F = sum_k FermionTerm_k.

    This is a lightweight container with basic arithmetic; it does not
    automatically normal-order or simplify via anti-commutation.
    """

    terms: Tuple[FermionTerm, ...]

    def __post_init__(self) -> None:
        """Validate and filter FermionOperator invariants."""
        # Convert input terms into a tuple
        if not isinstance(self.terms, tuple):
            object.__setattr__(self, "terms", tuple(self.terms))

        # Filter out any terms whose coefficient is exactly zero (within small tolerance)
        filtered = []
        for term in self.terms:
            if abs(term.coeff) > 1e-15:
                filtered.append(term)
        object.__setattr__(self, "terms", tuple(filtered))

    @classmethod
    def from_terms(cls, terms: Iterable[FermionTerm]) -> "FermionOperator":
        """
        Create a FermionOperator from an iterable of FermionTerm objects.

        Parameters
        ----------
        terms:
            Iterable of FermionTerm objects.

        Returns
        -------
        FermionOperator
            A new FermionOperator instance.
        """
        return cls(terms=tuple(terms))

    def __add__(self, other: "FermionOperator") -> "FermionOperator":
        """
        Add two FermionOperators by concatenating their terms.

        Parameters
        ----------
        other:
            Another FermionOperator to add.

        Returns
        -------
        FermionOperator
            Sum of the two operators.
        """
        if not isinstance(other, FermionOperator):
            return NotImplemented
        return FermionOperator.from_terms(self.terms + other.terms)

    def __rmul__(self, scalar: complex) -> "FermionOperator":
        """
        Multiply a FermionOperator by a scalar on the left.

        Parameters
        ----------
        scalar:
            Scalar coefficient to multiply by.

        Returns
        -------
        FermionOperator
            Scaled operator.

        Raises
        ------
        TypeError:
            If scalar cannot be converted to complex.
        """
        try:
            c = complex(scalar)
        except (TypeError, ValueError) as exc:
            raise TypeError("FermionOperator can only be multiplied by scalars.") from exc
        return FermionOperator.from_terms(
            [
                FermionTerm(coeff=c * term.coeff, operators=term.operators)
                for term in self.terms
            ]
        )

    def __mul__(self, scalar: complex) -> "FermionOperator":
        """
        Multiply a FermionOperator by a scalar on the right.

        Parameters
        ----------
        scalar:
            Scalar coefficient to multiply by.

        Returns
        -------
        FermionOperator
            Scaled operator.
        """
        return self.__rmul__(scalar)

    def is_zero(self) -> bool:
        """Return True if the operator has no nonzero terms."""
        return len(self.terms) == 0


__all__ = [
    "FermionOpSymbol",
    "FermionTerm",
    "FermionOperator",
]

