"""Tests for probabilistic utilities."""

import numpy as np
import pytest

from qconduit.probabilistic.utils import (
    effective_sample_size,
    ensure_1d,
    ensure_2d,
    log_normal_pdf,
    logsumexp,
    normal_pdf,
    normalize_log_weights,
    systematic_resample,
)


def test_logsumexp_basic():
    """Test logsumexp with simple cases."""
    # Single value
    assert logsumexp(np.array([1.0])) == pytest.approx(1.0)

    # Two values
    result = logsumexp(np.array([1.0, 2.0]))
    expected = np.log(np.exp(1.0) + np.exp(2.0))
    assert result == pytest.approx(expected)

    # Negative values (should be stable)
    result = logsumexp(np.array([-10.0, -11.0, -12.0]))
    expected = np.log(np.exp(-10.0) + np.exp(-11.0) + np.exp(-12.0))
    assert result == pytest.approx(expected, rel=1e-10)

    # With axis
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = logsumexp(a, axis=0)
    expected0 = np.log(np.exp(1.0) + np.exp(3.0))
    expected1 = np.log(np.exp(2.0) + np.exp(4.0))
    assert result[0] == pytest.approx(expected0)
    assert result[1] == pytest.approx(expected1)


def test_logsumexp_against_naive():
    """Compare logsumexp to naive implementation for small arrays."""
    a = np.array([1.0, 2.0, 3.0])
    result = logsumexp(a)
    naive = np.log(np.sum(np.exp(a)))
    assert result == pytest.approx(naive)

    # Large negative values (naive would underflow)
    a = np.array([-100.0, -101.0])
    result = logsumexp(a)
    # Should still work
    assert np.isfinite(result)


def test_normal_pdf_1d():
    """Test 1D normal PDF."""
    x = np.array([0.0])
    mean = np.array([0.0])
    cov = np.array([[1.0]])
    pdf_val = normal_pdf(x, mean, cov)
    expected = 1.0 / np.sqrt(2 * np.pi)
    assert pdf_val == pytest.approx(expected, rel=1e-6)


def test_normal_pdf_2d():
    """Test 2D normal PDF."""
    x = np.array([0.0, 0.0])
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    pdf_val = normal_pdf(x, mean, cov)
    expected = 1.0 / (2 * np.pi)
    assert pdf_val == pytest.approx(expected, rel=1e-6)


def test_log_normal_pdf_1d():
    """Test 1D log normal PDF."""
    x = np.array([0.0])
    mean = np.array([0.0])
    cov = np.array([[1.0]])
    log_pdf_val = log_normal_pdf(x, mean, cov)
    expected = -0.5 * np.log(2 * np.pi)
    assert log_pdf_val == pytest.approx(expected, rel=1e-6)


def test_log_normal_pdf_2d():
    """Test 2D log normal PDF."""
    x = np.array([1.0, 1.0])
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    log_pdf_val = log_normal_pdf(x, mean, cov)
    # Should be -log(2*pi) - 0.5 * (1^2 + 1^2)
    expected = -np.log(2 * np.pi) - 1.0
    assert log_pdf_val == pytest.approx(expected, rel=1e-6)


def test_systematic_resample_deterministic():
    """Test systematic resampling is deterministic with seed."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    weights = np.array([0.5, 0.3, 0.2])

    indices1 = systematic_resample(weights, rng1)
    indices2 = systematic_resample(weights, rng2)

    np.testing.assert_array_equal(indices1, indices2)
    assert len(indices1) == len(weights)


def test_systematic_resample_properties():
    """Test systematic resampling properties."""
    rng = np.random.default_rng(42)
    weights = np.array([0.6, 0.3, 0.1])
    indices = systematic_resample(weights, rng)

    # Should have correct length
    assert len(indices) == len(weights)
    # All indices should be valid
    assert np.all(indices >= 0)
    assert np.all(indices < len(weights))


def test_effective_sample_size():
    """Test effective sample size computation."""
    # Uniform weights: ESS = N
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    ess = effective_sample_size(weights)
    assert ess == pytest.approx(4.0, rel=1e-6)

    # Degenerate: one weight = 1, others = 0
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    ess = effective_sample_size(weights)
    assert ess == pytest.approx(1.0, rel=1e-6)

    # General case
    weights = np.array([0.5, 0.3, 0.2])
    ess = effective_sample_size(weights)
    expected = 1.0 / (0.5**2 + 0.3**2 + 0.2**2)
    assert ess == pytest.approx(expected, rel=1e-6)
    assert 1.0 <= ess <= len(weights)


def test_normalize_log_weights():
    """Test log-weight normalization."""
    log_w = np.array([-1.0, -2.0, -3.0])
    w, log_z = normalize_log_weights(log_w)

    # Weights should sum to 1
    assert np.sum(w) == pytest.approx(1.0, rel=1e-10)
    # All weights should be positive
    assert np.all(w > 0)
    # Log evidence should be logsumexp
    expected_log_z = logsumexp(log_w)
    assert log_z == pytest.approx(expected_log_z, rel=1e-10)


def test_normalize_log_weights_extreme():
    """Test normalization with extreme log-weights."""
    # Very negative (should still work)
    log_w = np.array([-100.0, -101.0, -102.0])
    w, log_z = normalize_log_weights(log_w)
    assert np.sum(w) == pytest.approx(1.0, rel=1e-10)
    assert np.all(w > 0)
    assert np.isfinite(log_z)


def test_ensure_1d():
    """Test ensure_1d function."""
    # Scalar -> 1D
    x = np.array(5.0)
    result = ensure_1d(x)
    assert result.shape == (1,)

    # Already 1D
    x = np.array([1, 2, 3])
    result = ensure_1d(x)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, x)

    # 2D -> error
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        ensure_1d(x)


def test_ensure_2d():
    """Test ensure_2d function."""
    # Scalar -> 2D
    x = np.array(5.0)
    result = ensure_2d(x)
    assert result.shape == (1, 1)

    # 1D -> 2D (column vector)
    x = np.array([1, 2, 3])
    result = ensure_2d(x)
    assert result.shape == (3, 1)

    # Already 2D
    x = np.array([[1, 2], [3, 4]])
    result = ensure_2d(x)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, x)

