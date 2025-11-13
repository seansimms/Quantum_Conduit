"""Tests for adiabatic PauliSum interpolation."""

from __future__ import annotations

import pytest

from qconduit.adiabatic import interpolate_paulisum
from qconduit.operators import PauliSum, PauliTerm


def test_interpolate_paulisum_simple():
    """Test interpolation between two simple 1-qubit Hamiltonians."""
    # H0 = Z with coeff 1.0
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])

    # H1 = Z with coeff 3.0
    H1 = PauliSum([PauliTerm(coeff=3.0, paulis=("Z",))])

    # H_mid = interpolate at s=0.25
    H_mid = interpolate_paulisum(H0, H1, s=0.25)

    # Expected coefficient: (1-0.25)*1 + 0.25*3 = 0.75 + 0.75 = 1.5
    assert len(H_mid.terms) == 1
    assert H_mid.terms[0].paulis == ("Z",)
    assert H_mid.terms[0].coeff == pytest.approx(1.5, abs=1e-10)


def test_interpolate_paulisum_endpoints():
    """Test that s=0 gives H0 and s=1 gives H1."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=3.0, paulis=("Z",))])

    # s=0 should equal H0
    H_s0 = interpolate_paulisum(H0, H1, s=0.0)
    assert len(H_s0.terms) == 1
    assert H_s0.terms[0].coeff == pytest.approx(1.0, abs=1e-10)
    assert H_s0.terms[0].paulis == ("Z",)

    # s=1 should equal H1
    H_s1 = interpolate_paulisum(H0, H1, s=1.0)
    assert len(H_s1.terms) == 1
    assert H_s1.terms[0].coeff == pytest.approx(3.0, abs=1e-10)
    assert H_s1.terms[0].paulis == ("Z",)


def test_interpolate_paulisum_multi_term():
    """Test interpolation with multiple terms."""
    H0 = PauliSum(
        [
            PauliTerm(coeff=1.0, paulis=("Z",)),
            PauliTerm(coeff=0.5, paulis=("X",)),
        ]
    )

    H1 = PauliSum(
        [
            PauliTerm(coeff=2.0, paulis=("Z",)),
            PauliTerm(coeff=1.5, paulis=("X",)),
        ]
    )

    H_mid = interpolate_paulisum(H0, H1, s=0.5)

    assert len(H_mid.terms) == 2

    # Find Z and X terms
    z_term = next(t for t in H_mid.terms if t.paulis == ("Z",))
    x_term = next(t for t in H_mid.terms if t.paulis == ("X",))

    # Expected: (1-0.5)*1 + 0.5*2 = 1.5 for Z
    assert z_term.coeff == pytest.approx(1.5, abs=1e-10)

    # Expected: (1-0.5)*0.5 + 0.5*1.5 = 1.0 for X
    assert x_term.coeff == pytest.approx(1.0, abs=1e-10)


def test_interpolate_paulisum_two_qubits():
    """Test interpolation with 2-qubit Hamiltonians."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z", "I"))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z", "I"))])

    H_mid = interpolate_paulisum(H0, H1, s=0.3)

    assert len(H_mid.terms) == 1
    assert H_mid.terms[0].paulis == ("Z", "I")
    # Expected: 0.7*1 + 0.3*2 = 1.3
    assert H_mid.terms[0].coeff == pytest.approx(1.3, abs=1e-10)


def test_interpolate_paulisum_invalid_s():
    """Test that interpolation raises ValueError for invalid s."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z",))])

    with pytest.raises(ValueError, match="Interpolation parameter s must be in \\[0,1\\]"):
        interpolate_paulisum(H0, H1, s=-0.1)

    with pytest.raises(ValueError, match="Interpolation parameter s must be in \\[0,1\\]"):
        interpolate_paulisum(H0, H1, s=1.1)


def test_interpolate_paulisum_different_terms():
    """Test that interpolation works with different number of terms."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum(
        [
            PauliTerm(coeff=2.0, paulis=("Z",)),
            PauliTerm(coeff=1.0, paulis=("X",)),
        ]
    )

    # Should work: combines all terms
    H_mid = interpolate_paulisum(H0, H1, s=0.5)

    # Should have both Z and X terms
    assert len(H_mid.terms) == 2

    # Find Z and X terms
    z_term = next(t for t in H_mid.terms if t.paulis == ("Z",))
    x_term = next(t for t in H_mid.terms if t.paulis == ("X",))

    # Z: 0.5*1 + 0.5*2 = 1.5
    assert z_term.coeff == pytest.approx(1.5, abs=1e-10)
    # X: 0.5*0 + 0.5*1 = 0.5
    assert x_term.coeff == pytest.approx(0.5, abs=1e-10)


def test_interpolate_paulisum_different_paulis():
    """Test that interpolation works with completely different Pauli strings."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("X",))])

    # Should work: combines terms with different Pauli strings
    H_mid = interpolate_paulisum(H0, H1, s=0.5)

    # Should have both Z and X terms
    assert len(H_mid.terms) == 2

    # Find Z and X terms
    z_term = next(t for t in H_mid.terms if t.paulis == ("Z",))
    x_term = next(t for t in H_mid.terms if t.paulis == ("X",))

    # Z: 0.5*1 + 0.5*0 = 0.5
    assert z_term.coeff == pytest.approx(0.5, abs=1e-10)
    # X: 0.5*0 + 0.5*2 = 1.0
    assert x_term.coeff == pytest.approx(1.0, abs=1e-10)


def test_interpolate_paulisum_mismatched_n_qubits():
    """Test that interpolation raises ValueError for mismatched n_qubits."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z", "I"))])

    with pytest.raises(ValueError, match="must act on the same number of qubits"):
        interpolate_paulisum(H0, H1, s=0.5)


def test_interpolate_paulisum_empty():
    """Test interpolation with empty PauliSum."""
    H0 = PauliSum([])
    H1 = PauliSum([])

    H_mid = interpolate_paulisum(H0, H1, s=0.5)
    assert len(H_mid.terms) == 0


def test_interpolate_paulisum_clamping():
    """Test that s values slightly outside [0,1] are clamped."""
    H0 = PauliSum([PauliTerm(coeff=1.0, paulis=("Z",))])
    H1 = PauliSum([PauliTerm(coeff=2.0, paulis=("Z",))])

    # s slightly below 0 should be clamped to 0
    H_neg = interpolate_paulisum(H0, H1, s=-1e-12)
    assert H_neg.terms[0].coeff == pytest.approx(1.0, abs=1e-10)

    # s slightly above 1 should be clamped to 1
    H_pos = interpolate_paulisum(H0, H1, s=1.0 + 1e-12)
    assert H_pos.terms[0].coeff == pytest.approx(2.0, abs=1e-10)

