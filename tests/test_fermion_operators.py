"""Tests for fermionic operator primitives."""

from __future__ import annotations

import pytest

from qconduit.fermion import (
    FermionOperator,
    FermionTerm,
)


class TestFermionTerm:
    """Tests for FermionTerm class."""

    def test_fermion_term_construction(self):
        """Test constructing valid FermionTerm objects."""
        term = FermionTerm(coeff=1.0, operators=((0, "+"), (1, "-")))
        assert term.coeff == 1.0
        assert term.operators == ((0, "+"), (1, "-"))
        assert not term.is_vacuum_term()

    def test_fermion_term_vacuum(self):
        """Test vacuum term (no operators)."""
        vac = FermionTerm(coeff=2.0, operators=())
        assert vac.is_vacuum_term()
        assert vac.coeff == 2.0
        assert len(vac.operators) == 0

    def test_fermion_term_coeff_conversion(self):
        """Test that coeff is converted to complex."""
        term = FermionTerm(coeff=1, operators=((0, "+"),))  # int
        assert isinstance(term.coeff, complex)
        assert term.coeff == 1.0

        term2 = FermionTerm(coeff=1.5 + 2.0j, operators=((0, "+"),))
        assert term2.coeff == 1.5 + 2.0j

    def test_fermion_term_operators_to_tuple(self):
        """Test that operators is converted to tuple."""
        term = FermionTerm(coeff=1.0, operators=[(0, "+"), (1, "-")])  # list
        assert isinstance(term.operators, tuple)
        assert term.operators == ((0, "+"), (1, "-"))

    def test_fermion_term_invalid_mode(self):
        """Test that invalid mode indices raise ValueError."""
        with pytest.raises(ValueError, match="mode indices must be >= 0"):
            FermionTerm(coeff=1.0, operators=((-1, "+"),))

        with pytest.raises(ValueError, match="mode indices must be >= 0"):
            FermionTerm(coeff=1.0, operators=((0, "+"), (-2, "-")))

    def test_fermion_term_invalid_op_type(self):
        """Test that invalid operator types raise ValueError."""
        with pytest.raises(ValueError, match="operator types must be"):
            FermionTerm(coeff=1.0, operators=((0, "x"),))

        with pytest.raises(ValueError, match="operator types must be"):
            FermionTerm(coeff=1.0, operators=((0, "+"), (1, "dagger")))

    def test_fermion_term_is_vacuum_term(self):
        """Test is_vacuum_term() method."""
        term1 = FermionTerm(coeff=1.0, operators=())
        assert term1.is_vacuum_term() is True

        term2 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        assert term2.is_vacuum_term() is False


class TestFermionOperator:
    """Tests for FermionOperator class."""

    def test_fermion_operator_construction(self):
        """Test constructing FermionOperator objects."""
        t1 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        t2 = FermionTerm(coeff=2.0, operators=((1, "-"),))
        op = FermionOperator(terms=(t1, t2))
        assert len(op.terms) == 2

    def test_fermion_operator_empty(self):
        """Test empty FermionOperator."""
        op = FermionOperator(terms=())
        assert len(op.terms) == 0
        assert op.is_zero()

    def test_fermion_operator_zero_filtering(self):
        """Test that zero-coefficient terms are filtered out."""
        t1 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        t2 = FermionTerm(coeff=0.0, operators=((1, "-"),))
        t3 = FermionTerm(coeff=1e-16, operators=((2, "+"),))  # Below threshold
        op = FermionOperator(terms=(t1, t2, t3))
        assert len(op.terms) == 1
        assert op.terms[0] == t1

    def test_fermion_operator_from_terms(self):
        """Test from_terms class method."""
        t1 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        t2 = FermionTerm(coeff=2.0, operators=((1, "-"),))
        op = FermionOperator.from_terms([t1, t2])
        assert len(op.terms) == 2

    def test_fermion_operator_is_zero(self):
        """Test is_zero() method."""
        zero_op = FermionOperator(terms=())
        assert zero_op.is_zero()

        t1 = FermionTerm(coeff=0.0, operators=((0, "+"),))
        op = FermionOperator(terms=(t1,))
        assert op.is_zero()  # Should be filtered out

        t2 = FermionTerm(coeff=1.0, operators=((0, "+"),))
        op2 = FermionOperator(terms=(t2,))
        assert not op2.is_zero()

    def test_fermion_operator_addition(self):
        """Test addition of FermionOperators."""
        op1 = FermionOperator(terms=(FermionTerm(1.0, ((0, "+"),)),))
        op2 = FermionOperator(terms=(FermionTerm(2.0, ((1, "-"),)),))
        op_sum = op1 + op2
        assert len(op_sum.terms) == 2

        # Test that addition is immutable
        assert len(op1.terms) == 1
        assert len(op2.terms) == 1

    def test_fermion_operator_scalar_multiplication(self):
        """Test scalar multiplication."""
        op1 = FermionOperator(terms=(FermionTerm(1.0, ((0, "+"),)),))
        op_scaled = 2.0 * op1
        assert pytest.approx(op_scaled.terms[0].coeff) == 2.0
        assert op_scaled.terms[0].operators == op1.terms[0].operators

        # Test complex scalar
        op_scaled_complex = (1.0 + 1.0j) * op1
        assert op_scaled_complex.terms[0].coeff == 1.0 + 1.0j

        # Test right multiplication
        op_scaled_right = op1 * 3.0
        assert pytest.approx(op_scaled_right.terms[0].coeff) == 3.0

    def test_fermion_operator_invalid_scalar(self):
        """Test that invalid scalar types raise TypeError."""
        op1 = FermionOperator(terms=(FermionTerm(1.0, ((0, "+"),)),))
        with pytest.raises(TypeError, match="can only be multiplied by scalars"):
            _ = "x" * op1

    def test_fermion_operator_addition_not_implemented(self):
        """Test that addition with non-FermionOperator returns NotImplemented."""
        op1 = FermionOperator(terms=(FermionTerm(1.0, ((0, "+"),)),))
        result = op1.__add__("not an operator")
        assert result is NotImplemented


