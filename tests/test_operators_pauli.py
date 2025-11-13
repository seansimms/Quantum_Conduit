"""Tests for Pauli operator primitives."""

import pytest
import torch

from qconduit.operators.pauli import PauliTerm, PauliSum
from qconduit.gates.standard import I, X, Y, Z


class TestPauliTerm:
    """Tests for PauliTerm class."""

    def test_pauli_term_construction(self):
        """Test constructing valid PauliTerm objects."""
        term1 = PauliTerm(1.0, ("Z",))
        assert term1.coeff == 1.0
        assert term1.paulis == ("Z",)
        assert term1.n_qubits() == 1

        term2 = PauliTerm(0.5, ("X", "Y", "Z"))
        assert term2.coeff == 0.5
        assert term2.paulis == ("X", "Y", "Z")
        assert term2.n_qubits() == 3

    def test_pauli_term_coeff_conversion(self):
        """Test that coeff is converted to float."""
        term = PauliTerm(1, ("Z",))  # int
        assert isinstance(term.coeff, float)
        assert term.coeff == 1.0

    def test_pauli_term_paulis_to_tuple(self):
        """Test that paulis is converted to tuple."""
        term = PauliTerm(1.0, ["Z", "I"])  # list
        assert isinstance(term.paulis, tuple)
        assert term.paulis == ("Z", "I")

    def test_pauli_term_invalid_label(self):
        """Test that invalid Pauli labels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli labels"):
            PauliTerm(1.0, ("A",))

        with pytest.raises(ValueError, match="Invalid Pauli labels"):
            PauliTerm(1.0, ("Z", "X", "W"))

    def test_pauli_term_empty_paulis(self):
        """Test that empty paulis raises ValueError."""
        with pytest.raises(ValueError, match="must have length >= 1"):
            PauliTerm(1.0, ())

    def test_pauli_term_n_qubits(self):
        """Test n_qubits() method."""
        term1 = PauliTerm(1.0, ("Z",))
        assert term1.n_qubits() == 1

        term2 = PauliTerm(1.0, ("I", "X", "Y", "Z"))
        assert term2.n_qubits() == 4

    def test_pauli_term_is_identity(self):
        """Test is_identity() method."""
        term1 = PauliTerm(1.0, ("I", "I", "I"))
        assert term1.is_identity() is True

        term2 = PauliTerm(1.0, ("I", "Z", "I"))
        assert term2.is_identity() is False

        term3 = PauliTerm(1.0, ("Z",))
        assert term3.is_identity() is False


class TestPauliSum:
    """Tests for PauliSum class."""

    def test_pauli_sum_construction(self):
        """Test constructing PauliSum objects."""
        term1 = PauliTerm(1.0, ("Z",))
        term2 = PauliTerm(0.5, ("X",))
        hamiltonian = PauliSum.from_terms([term1, term2])
        assert len(hamiltonian) == 2
        assert hamiltonian.n_qubits() == 1

    def test_pauli_sum_empty(self):
        """Test empty PauliSum."""
        hamiltonian = PauliSum()
        assert len(hamiltonian) == 0
        assert hamiltonian.n_qubits() == 0

    def test_pauli_sum_inconsistent_n_qubits(self):
        """Test that inconsistent n_qubits raises ValueError."""
        term1 = PauliTerm(1.0, ("Z",))
        term2 = PauliTerm(0.5, ("X", "Y"))
        with pytest.raises(ValueError, match="must have the same n_qubits"):
            PauliSum.from_terms([term1, term2])

    def test_pauli_sum_add_term(self):
        """Test add_term() method."""
        hamiltonian = PauliSum()
        term1 = PauliTerm(1.0, ("Z",))
        hamiltonian.add_term(term1)
        assert len(hamiltonian) == 1
        assert hamiltonian.n_qubits() == 1

        term2 = PauliTerm(0.5, ("X",))
        hamiltonian.add_term(term2)
        assert len(hamiltonian) == 2

    def test_pauli_sum_add_term_inconsistent_n_qubits(self):
        """Test that adding term with inconsistent n_qubits raises ValueError."""
        hamiltonian = PauliSum()
        term1 = PauliTerm(1.0, ("Z",))
        hamiltonian.add_term(term1)

        term2 = PauliTerm(0.5, ("X", "Y"))
        with pytest.raises(ValueError, match="Cannot add term"):
            hamiltonian.add_term(term2)

    def test_pauli_sum_simplify(self):
        """Test simplify() method combines identical terms."""
        term1 = PauliTerm(1.0, ("Z", "I"))
        term2 = PauliTerm(0.5, ("Z", "I"))
        term3 = PauliTerm(0.3, ("I", "X"))
        hamiltonian = PauliSum.from_terms([term1, term2, term3])

        simplified = hamiltonian.simplify()
        assert len(simplified) == 2

        # Find the combined ZI term
        zi_term = None
        ix_term = None
        for term in simplified.terms:
            if term.paulis == ("Z", "I"):
                zi_term = term
            elif term.paulis == ("I", "X"):
                ix_term = term

        assert zi_term is not None
        assert abs(zi_term.coeff - 1.5) < 1e-10
        assert ix_term is not None
        assert abs(ix_term.coeff - 0.3) < 1e-10

    def test_pauli_sum_simplify_drop_small(self):
        """Test simplify() drops terms below tolerance."""
        term1 = PauliTerm(1.0, ("Z",))
        term2 = PauliTerm(1e-15, ("X",))  # Very small
        hamiltonian = PauliSum.from_terms([term1, term2])

        simplified = hamiltonian.simplify(tol=1e-12)
        assert len(simplified) == 1
        assert simplified.terms[0].paulis == ("Z",)

    def test_pauli_sum_to_matrix_1_qubit(self):
        """Test to_matrix() for 1-qubit systems."""
        # Single Z term
        term = PauliTerm(1.0, ("Z",))
        hamiltonian = PauliSum.from_terms([term])
        matrix = hamiltonian.to_matrix()

        # Compare with known Z matrix
        z_matrix = Z(dtype=torch.complex64, device=torch.device("cpu"))
        assert torch.allclose(matrix, z_matrix, atol=1e-6)

        # Single X term
        term_x = PauliTerm(1.0, ("X",))
        hamiltonian_x = PauliSum.from_terms([term_x])
        matrix_x = hamiltonian_x.to_matrix()
        x_matrix = X(dtype=torch.complex64, device=torch.device("cpu"))
        assert torch.allclose(matrix_x, x_matrix, atol=1e-6)

        # Single Y term
        term_y = PauliTerm(1.0, ("Y",))
        hamiltonian_y = PauliSum.from_terms([term_y])
        matrix_y = hamiltonian_y.to_matrix()
        y_matrix = Y(dtype=torch.complex64, device=torch.device("cpu"))
        assert torch.allclose(matrix_y, y_matrix, atol=1e-6)

        # Linear combination: 0.5 * Z + 0.5 * X
        term_z = PauliTerm(0.5, ("Z",))
        term_x2 = PauliTerm(0.5, ("X",))
        hamiltonian_comb = PauliSum.from_terms([term_z, term_x2])
        matrix_comb = hamiltonian_comb.to_matrix()
        expected = 0.5 * z_matrix + 0.5 * x_matrix
        assert torch.allclose(matrix_comb, expected, atol=1e-6)

    def test_pauli_sum_to_matrix_2_qubit(self):
        """Test to_matrix() for 2-qubit systems."""
        # Z ⊗ I (Z on qubit 0, I on qubit 1) with little-endian indexing
        # This corresponds to I ⊗ Z in standard tensor product order
        term_zi = PauliTerm(1.0, ("Z", "I"))
        hamiltonian_zi = PauliSum.from_terms([term_zi])
        matrix_zi = hamiltonian_zi.to_matrix()

        # Expected: I ⊗ Z using Kronecker product (reversed for little-endian)
        z_mat = Z(dtype=torch.complex64, device=torch.device("cpu"))
        i_mat = I(dtype=torch.complex64, device=torch.device("cpu"))
        expected_zi = torch.kron(i_mat, z_mat)
        assert torch.allclose(matrix_zi, expected_zi, atol=1e-6)

        # I ⊗ Z (I on qubit 0, Z on qubit 1) with little-endian indexing
        # This corresponds to Z ⊗ I in standard tensor product order
        term_iz = PauliTerm(1.0, ("I", "Z"))
        hamiltonian_iz = PauliSum.from_terms([term_iz])
        matrix_iz = hamiltonian_iz.to_matrix()
        expected_iz = torch.kron(z_mat, i_mat)
        assert torch.allclose(matrix_iz, expected_iz, atol=1e-6)

    def test_pauli_sum_to_matrix_empty(self):
        """Test to_matrix() raises for empty PauliSum."""
        hamiltonian = PauliSum()
        with pytest.raises(ValueError, match="Cannot compute matrix for empty"):
            hamiltonian.to_matrix()

    def test_pauli_sum_to_matrix_large_n(self):
        """Test to_matrix() raises for large n_qubits."""
        term = PauliTerm(1.0, ("Z",) * 6)  # 6 qubits
        hamiltonian = PauliSum.from_terms([term])
        with pytest.raises(ValueError, match="only intended for small systems"):
            hamiltonian.to_matrix()


class TestPauliSumMatrixConversion:
    """Comprehensive tests for PauliSum matrix conversion."""

    def test_pauli_sum_to_matrix_z_on_one_qubit(self):
        """Test H = Z on 1 qubit matches diag(1, -1)."""
        term = PauliTerm(1.0, ("Z",))
        H = PauliSum.from_terms([term])
        matrix = H.to_matrix()

        expected = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_pauli_sum_to_matrix_zz_on_two_qubits(self):
        """Test H = Z⊗Z matches diag(1, -1, -1, 1)."""
        term = PauliTerm(1.0, ("Z", "Z"))
        H = PauliSum.from_terms([term])
        matrix = H.to_matrix()

        expected = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.complex64,
        )
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_pauli_sum_commuting_hamiltonians(self):
        """Test that commuting Hamiltonians have [H1, H2] = 0."""
        # H1 = Z on qubit 0
        term1 = PauliTerm(1.0, ("Z", "I"))
        H1 = PauliSum.from_terms([term1])

        # H2 = Z on qubit 1
        term2 = PauliTerm(1.0, ("I", "Z"))
        H2 = PauliSum.from_terms([term2])

        # Compute commutator [H1, H2] = H1 @ H2 - H2 @ H1
        M1 = H1.to_matrix()
        M2 = H2.to_matrix()
        commutator = M1 @ M2 - M2 @ M1

        # Should be zero (commuting)
        assert torch.allclose(commutator, torch.zeros_like(commutator), atol=1e-6)

    def test_pauli_sum_non_commuting_hamiltonians(self):
        """Test that non-commuting Hamiltonians have [H1, H2] ≠ 0."""
        # H1 = X on qubit 0
        term1 = PauliTerm(1.0, ("X", "I"))
        H1 = PauliSum.from_terms([term1])

        # H2 = Z on qubit 0
        term2 = PauliTerm(1.0, ("Z", "I"))
        H2 = PauliSum.from_terms([term2])

        # Compute commutator
        M1 = H1.to_matrix()
        M2 = H2.to_matrix()
        commutator = M1 @ M2 - M2 @ M1

        # Should not be zero (non-commuting)
        assert not torch.allclose(commutator, torch.zeros_like(commutator), atol=1e-6)

    def test_pauli_sum_commuting_zz_terms(self):
        """Test that sums of Z or ZZ terms commute."""
        # H1 = Z₀ + Z₁
        term1 = PauliTerm(1.0, ("Z", "I"))
        term2 = PauliTerm(1.0, ("I", "Z"))
        H1 = PauliSum.from_terms([term1, term2])

        # H2 = Z₀Z₁
        term3 = PauliTerm(1.0, ("Z", "Z"))
        H2 = PauliSum.from_terms([term3])

        # Compute commutator
        M1 = H1.to_matrix()
        M2 = H2.to_matrix()
        commutator = M1 @ M2 - M2 @ M1

        # Should be zero (all Z terms commute)
        assert torch.allclose(commutator, torch.zeros_like(commutator), atol=1e-6)

