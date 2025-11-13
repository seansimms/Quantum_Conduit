"""Pauli operator primitives for representing Pauli-sum Hamiltonians."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import torch

from ..gates.standard import I, X, Y, Z

# Valid Pauli labels
_VALID_PAULI_LABELS = {"I", "X", "Y", "Z"}


def _pauli_label_to_matrix(label: str) -> dict[str, torch.Tensor]:
    """
    Get the 2×2 matrix representation of a Pauli operator.

    This is a standard textbook mapping:
    - I: Identity matrix
    - X: Pauli-X (bit-flip)
    - Y: Pauli-Y
    - Z: Pauli-Z (phase-flip)

    Args:
        label: Single character "I", "X", "Y", or "Z".

    Returns:
        Dictionary mapping labels to their 2×2 complex matrices.

    Raises:
        ValueError: If label is not in {"I", "X", "Y", "Z"}.
    """
    if label not in _VALID_PAULI_LABELS:
        raise ValueError(
            f"Invalid Pauli label '{label}'. Must be one of {_VALID_PAULI_LABELS}"
        )

    # Use default dtype and device for the mapping (will be overridden in to_matrix)
    return {
        "I": I(dtype=torch.complex64, device=torch.device("cpu")),
        "X": X(dtype=torch.complex64, device=torch.device("cpu")),
        "Y": Y(dtype=torch.complex64, device=torch.device("cpu")),
        "Z": Z(dtype=torch.complex64, device=torch.device("cpu")),
    }


@dataclass(frozen=True)
class PauliTerm:
    """
    A single Pauli term: a coefficient times a tensor product of Pauli operators.

    This represents a term of the form c * P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}, where
    each P_i is one of I, X, Y, Z, and c is a real coefficient.

    This is a standard, textbook representation of Pauli operators used in
    quantum computing. There is nothing proprietary or novel here; it is a
    utility container for I/X/Y/Z tensor products.

    Args:
        coeff: Real scalar coefficient.
        paulis: Sequence of Pauli labels, one per qubit. Each label must be
            "I", "X", "Y", or "Z". Length must be >= 1.

    Attributes:
        coeff: Real coefficient.
        paulis: Tuple of Pauli labels.

    Example:
        >>> term = PauliTerm(1.0, ("Z", "I"))  # Z ⊗ I on 2 qubits
        >>> term.n_qubits()
        2
        >>> term.is_identity()
        False
    """

    coeff: float
    paulis: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate PauliTerm invariants."""
        # Convert coeff to float
        object.__setattr__(self, "coeff", float(self.coeff))

        # Convert paulis to tuple if needed
        if not isinstance(self.paulis, tuple):
            object.__setattr__(self, "paulis", tuple(self.paulis))

        # Validate length
        if len(self.paulis) < 1:
            raise ValueError(
                f"paulis must have length >= 1, got {len(self.paulis)}"
            )

        # Validate all labels
        invalid_labels = [p for p in self.paulis if p not in _VALID_PAULI_LABELS]
        if invalid_labels:
            raise ValueError(
                f"Invalid Pauli labels: {invalid_labels}. "
                f"All labels must be in {_VALID_PAULI_LABELS}"
            )

    def n_qubits(self) -> int:
        """Return the number of qubits this term acts on."""
        return len(self.paulis)

    def is_identity(self) -> bool:
        """
        Check if this term is proportional to the identity operator.

        Returns:
            True if all Pauli labels are "I", False otherwise.
        """
        return all(p == "I" for p in self.paulis)


@dataclass
class PauliSum:
    """
    A Pauli-sum Hamiltonian: a sum of PauliTerm objects.

    This represents a Hamiltonian of the form H = ∑ᵢ cᵢ Pᵢ, where each Pᵢ
    is a tensor product of Pauli operators (I, X, Y, Z) and cᵢ is a real
    coefficient.

    This is a standard, textbook representation of Pauli-sum Hamiltonians
    used in quantum computing. There is nothing proprietary or novel here;
    it is a utility container for linear combinations of Pauli operators.

    All terms must act on the same number of qubits.

    Args:
        terms: List of PauliTerm objects. All terms must have the same n_qubits.

    Attributes:
        terms: List of PauliTerm objects.

    Example:
        >>> term1 = PauliTerm(1.0, ("Z",))
        >>> term2 = PauliTerm(0.5, ("X",))
        >>> hamiltonian = PauliSum.from_terms([term1, term2])
        >>> hamiltonian.n_qubits()
        1
    """

    terms: List[PauliTerm] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate PauliSum invariants."""
        if len(self.terms) > 0:
            n_qubits = self.terms[0].n_qubits()
            for i, term in enumerate(self.terms):
                if term.n_qubits() != n_qubits:
                    raise ValueError(
                        f"All terms must have the same n_qubits. "
                        f"Term 0 has {n_qubits} qubits, but term {i} has {term.n_qubits()} qubits."
                    )

    def n_qubits(self) -> int:
        """
        Return the number of qubits this Hamiltonian acts on.

        Returns:
            Number of qubits, or 0 if there are no terms.
        """
        if len(self.terms) == 0:
            return 0
        return self.terms[0].n_qubits()

    def __len__(self) -> int:
        """Return the number of terms in this Pauli-sum."""
        return len(self.terms)

    def add_term(self, term: PauliTerm) -> None:
        """
        Add a term to this Pauli-sum.

        The term must have the same n_qubits as existing terms (or this can
        be the first term).

        Args:
            term: PauliTerm to add.

        Raises:
            ValueError: If term.n_qubits() does not match existing terms.
        """
        if len(self.terms) == 0:
            self.terms.append(term)
        else:
            n_qubits = self.n_qubits()
            if term.n_qubits() != n_qubits:
                raise ValueError(
                    f"Cannot add term with {term.n_qubits()} qubits to "
                    f"PauliSum with {n_qubits} qubits."
                )
            self.terms.append(term)

    @classmethod
    def from_terms(cls, terms: Iterable[PauliTerm]) -> "PauliSum":
        """
        Create a PauliSum from an iterable of PauliTerm objects.

        Args:
            terms: Iterable of PauliTerm objects. All must have the same n_qubits.

        Returns:
            A new PauliSum instance.

        Raises:
            ValueError: If terms have inconsistent n_qubits.
        """
        terms_list = list(terms)
        return cls(terms=terms_list)

    def simplify(self, tol: float = 1e-12) -> "PauliSum":
        """
        Simplify this Pauli-sum by combining terms with identical Pauli sequences.

        Terms with identical paulis sequences are combined by summing their
        coefficients. Terms with |coeff| < tol are dropped.

        This is straightforward linear combination; nothing novel.

        Args:
            tol: Tolerance for dropping small terms. Terms with |coeff| < tol
                are removed.

        Returns:
            A new simplified PauliSum.
        """
        if len(self.terms) == 0:
            return PauliSum()

        # Group terms by their paulis sequence
        coeff_map: dict[Tuple[str, ...], float] = {}
        for term in self.terms:
            key = term.paulis
            if key in coeff_map:
                coeff_map[key] += term.coeff
            else:
                coeff_map[key] = term.coeff

        # Create new terms, dropping those below tolerance
        new_terms: List[PauliTerm] = []
        for paulis, coeff in coeff_map.items():
            if abs(coeff) >= tol:
                new_terms.append(PauliTerm(coeff=coeff, paulis=paulis))

        return PauliSum(terms=new_terms)

    def to_matrix(
        self, dtype: torch.dtype = torch.complex64, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute the full 2ⁿ×2ⁿ matrix representation of this Pauli-sum.

        This method constructs the matrix via Kronecker products of the
        individual Pauli matrices. This is standard linear algebra.

        WARNING: This method is only intended for small systems (n ≤ 5) and
        testing purposes. For larger systems, use expectation evaluation
        methods instead.

        Args:
            dtype: Complex dtype for the matrix. Defaults to torch.complex64.
            device: PyTorch device. Defaults to torch.device("cpu").

        Returns:
            A (2**n_qubits, 2**n_qubits) complex tensor representing the
            Hamiltonian matrix.

        Raises:
            ValueError: If n_qubits > 5 (safety check for large matrices).
            ValueError: If there are no terms.
        """
        if device is None:
            device = torch.device("cpu")

        n_qubits = self.n_qubits()
        if n_qubits == 0:
            raise ValueError("Cannot compute matrix for empty PauliSum")

        if n_qubits > 5:
            raise ValueError(
                f"to_matrix() is only intended for small systems (n ≤ 5). "
                f"Got n_qubits = {n_qubits}. Use expectation evaluation methods instead."
            )

        dim = 2**n_qubits
        matrix = torch.zeros((dim, dim), dtype=dtype, device=device)

        # Get Pauli matrices with correct dtype and device
        pauli_matrices = {
            "I": I(dtype=dtype, device=device),
            "X": X(dtype=dtype, device=device),
            "Y": Y(dtype=dtype, device=device),
            "Z": Z(dtype=dtype, device=device),
        }

        # For each term, compute the Kronecker product and add to matrix
        # Note: For little-endian indexing (qubit 0 is LSB), we need to reverse
        # the order of the Kronecker product. If paulis = (P_0, P_1, ..., P_{n-1})
        # where P_i acts on qubit i (LSB = 0), then the matrix is:
        # P_{n-1} ⊗ P_{n-2} ⊗ ... ⊗ P_0
        for term in self.terms:
            # Build tensor product in reverse order for little-endian
            term_matrix = pauli_matrices[term.paulis[n_qubits - 1]]
            for i in range(n_qubits - 2, -1, -1):
                pauli_i = pauli_matrices[term.paulis[i]]
                # Kronecker product: A ⊗ B
                term_matrix = torch.kron(term_matrix, pauli_i)

            # Add to total matrix with coefficient
            matrix = matrix + term.coeff * term_matrix

        return matrix

