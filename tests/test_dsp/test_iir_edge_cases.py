"""Comprehensive edge case tests for DSP/IIR functions to improve coverage and robustness."""

import numpy as np
import pytest

from qconduit.dsp.iir import butterworth, cheby1, sosfilt


class TestButterworthEdgeCases:
    """Edge case tests for butterworth() function."""

    def test_butterworth_zero_order(self):
        """Test butterworth with zero order raises error."""
        with pytest.raises(ValueError, match="Order must be positive"):
            butterworth(0, 0.1, fs=1.0)

    def test_butterworth_negative_order(self):
        """Test butterworth with negative order raises error."""
        with pytest.raises(ValueError, match="Order must be positive"):
            butterworth(-1, 0.1, fs=1.0)

    def test_butterworth_very_large_order(self):
        """Test butterworth with very large order raises error."""
        with pytest.raises(ValueError, match="Order too large"):
            butterworth(100, 0.1, fs=1.0)

    def test_butterworth_zero_cutoff(self):
        """Test butterworth with zero cutoff raises error."""
        with pytest.raises(ValueError, match="Cutoff frequency must be positive"):
            butterworth(4, 0.0, fs=1.0)

    def test_butterworth_negative_cutoff(self):
        """Test butterworth with negative cutoff raises error."""
        with pytest.raises(ValueError, match="Cutoff frequency must be positive"):
            butterworth(4, -0.1, fs=1.0)

    def test_butterworth_nan_cutoff(self):
        """Test butterworth with NaN cutoff raises error."""
        with pytest.raises(ValueError, match="Cutoff frequency must be finite"):
            butterworth(4, np.nan, fs=1.0)

    def test_butterworth_inf_cutoff(self):
        """Test butterworth with Inf cutoff raises error."""
        with pytest.raises(ValueError, match="Cutoff frequency must be finite"):
            butterworth(4, np.inf, fs=1.0)

    def test_butterworth_cutoff_above_nyquist(self):
        """Test butterworth with cutoff above Nyquist raises error."""
        with pytest.raises(ValueError, match="must be < Nyquist frequency"):
            butterworth(4, 0.6, fs=1.0)  # 0.6 > fs/2 = 0.5

    def test_butterworth_zero_fs(self):
        """Test butterworth with zero sampling frequency raises error."""
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            butterworth(4, 0.1, fs=0.0)

    def test_butterworth_negative_fs(self):
        """Test butterworth with negative sampling frequency raises error."""
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            butterworth(4, 0.1, fs=-1.0)

    def test_butterworth_nan_fs(self):
        """Test butterworth with NaN sampling frequency raises error."""
        with pytest.raises(ValueError, match="Sampling frequency must be finite"):
            butterworth(4, 0.1, fs=np.nan)

    def test_butterworth_invalid_btype(self):
        """Test butterworth with invalid btype raises error."""
        with pytest.raises(ValueError, match="Invalid btype"):
            butterworth(4, 0.1, fs=1.0, btype="invalid")

    def test_butterworth_bandpass_wrong_cutoff(self):
        """Test butterworth bandpass with single cutoff raises error."""
        with pytest.raises(ValueError, match="requires tuple cutoff"):
            butterworth(4, 0.1, fs=1.0, btype="bandpass")

    def test_butterworth_lowpass_tuple_cutoff(self):
        """Test butterworth lowpass with tuple cutoff raises error."""
        with pytest.raises(ValueError, match="Tuple cutoff requires"):
            butterworth(4, (0.1, 0.2), fs=1.0, btype="lowpass")

    def test_butterworth_bandpass_invalid_tuple(self):
        """Test butterworth bandpass with invalid tuple raises error."""
        with pytest.raises(ValueError, match="Cutoff tuple must have 2 elements"):
            butterworth(4, (0.1,), fs=1.0, btype="bandpass")

    def test_butterworth_bandpass_wrong_order(self):
        """Test butterworth bandpass with cutoff_low >= cutoff_high raises error."""
        with pytest.raises(ValueError, match="cutoff_low must be < cutoff_high"):
            butterworth(4, (0.2, 0.1), fs=1.0, btype="bandpass")

    def test_butterworth_bandpass_above_nyquist(self):
        """Test butterworth bandpass with cutoff_high above Nyquist raises error."""
        with pytest.raises(ValueError, match="must be < Nyquist frequency"):
            butterworth(4, (0.1, 0.6), fs=1.0, btype="bandpass")

    def test_butterworth_near_zero_cutoff(self):
        """Test butterworth with near-zero cutoff (should use minimum)."""
        # Should not raise, but use minimum cutoff
        # The cutoff gets adjusted to 1e-6 * fs internally
        b, a = butterworth(2, 1e-10, fs=1.0)
        b_arr = np.asarray(b)
        a_arr = np.asarray(a)
        assert b_arr.size > 0
        assert a_arr.size > 0
        # Coefficients should be finite
        assert np.all(np.isfinite(b_arr))
        assert np.all(np.isfinite(a_arr))

    def test_butterworth_small_order(self):
        """Test butterworth with order=1 (minimum valid)."""
        b, a = butterworth(1, 0.1, fs=1.0)
        b_arr = np.asarray(b)
        a_arr = np.asarray(a)
        assert b_arr.size > 0
        assert a_arr.size > 0
        # Coefficients should be finite
        assert np.all(np.isfinite(b_arr))
        assert np.all(np.isfinite(a_arr))


class TestCheby1EdgeCases:
    """Edge case tests for cheby1() function."""

    def test_cheby1_zero_order(self):
        """Test cheby1 with zero order raises error."""
        with pytest.raises(ValueError, match="Order must be positive"):
            cheby1(0, 0.5, 0.1, fs=1.0)

    def test_cheby1_negative_rp(self):
        """Test cheby1 with negative ripple raises error."""
        with pytest.raises(ValueError, match="Passband ripple must be positive"):
            cheby1(4, -0.5, 0.1, fs=1.0)

    def test_cheby1_zero_rp(self):
        """Test cheby1 with zero ripple raises error."""
        with pytest.raises(ValueError, match="Passband ripple must be positive"):
            cheby1(4, 0.0, 0.1, fs=1.0)

    def test_cheby1_nan_rp(self):
        """Test cheby1 with NaN ripple raises error."""
        with pytest.raises(ValueError, match="Passband ripple must be finite"):
            cheby1(4, np.nan, 0.1, fs=1.0)

    def test_cheby1_very_large_rp(self):
        """Test cheby1 with very large ripple raises error."""
        with pytest.raises(ValueError, match="Passband ripple too large"):
            cheby1(4, 200.0, 0.1, fs=1.0)

    def test_cheby1_invalid_btype(self):
        """Test cheby1 with invalid btype raises error."""
        with pytest.raises(ValueError, match="Invalid btype"):
            cheby1(4, 0.5, 0.1, fs=1.0, btype="invalid")

    def test_cheby1_cutoff_validation(self):
        """Test cheby1 cutoff validation (same as butterworth)."""
        with pytest.raises(ValueError, match="Cutoff frequency must be positive"):
            cheby1(4, 0.5, -0.1, fs=1.0)

    def test_cheby1_bandpass_validation(self):
        """Test cheby1 bandpass validation."""
        with pytest.raises(ValueError, match="cutoff_low must be < cutoff_high"):
            cheby1(4, 0.5, (0.2, 0.1), fs=1.0, btype="bandpass")


class TestSosfiltEdgeCases:
    """Edge case tests for sosfilt() function."""

    def test_sosfilt_empty_sos(self):
        """Test sosfilt with empty SOS array."""
        x = np.array([1.0, 2.0, 3.0])
        sos = np.array([]).reshape(0, 6)
        result = sosfilt(sos, x)
        np.testing.assert_array_equal(result, x)

    def test_sosfilt_empty_input(self):
        """Test sosfilt with empty input signal."""
        sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        x = np.array([])
        result = sosfilt(sos, x)
        assert len(result) == 0

    def test_sosfilt_single_sample(self):
        """Test sosfilt with single sample."""
        sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        x = np.array([1.0])
        result = sosfilt(sos, x)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_sosfilt_constant_signal(self):
        """Test sosfilt with constant signal."""
        sos = np.array([[1.0, 0.0, 0.0, 1.0, -0.5, 0.0]])  # Simple IIR
        x = np.ones(100) * 5.0
        result = sosfilt(sos, x)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_sosfilt_ill_conditioned_sos(self):
        """Test sosfilt with ill-conditioned SOS (very small a0)."""
        # Create SOS with very small a0 (should normalize)
        sos = np.array([[1.0, 0.0, 0.0, 1e-15, 0.0, 0.0]])
        x = np.array([1.0, 2.0, 3.0])
        result = sosfilt(sos, x)
        # Should handle gracefully
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_sosfilt_multiple_sections(self):
        """Test sosfilt with multiple SOS sections."""
        # Create cascade of two sections
        sos = np.array([
            [1.0, 0.0, 0.0, 1.0, -0.5, 0.0],
            [1.0, 0.0, 0.0, 1.0, -0.3, 0.0],
        ])
        x = np.random.randn(100)
        result = sosfilt(sos, x)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_sosfilt_nan_input(self):
        """Test sosfilt with NaN in input raises error (validation)."""
        sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        x = np.array([1.0, np.nan, 3.0])
        # check_1d_array should catch NaN
        with pytest.raises(ValueError, match="Input contains NaN"):
            sosfilt(sos, x)

    def test_sosfilt_inf_input(self):
        """Test sosfilt with Inf in input raises error (validation)."""
        sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        x = np.array([1.0, np.inf, 3.0])
        # check_1d_array should catch Inf
        with pytest.raises(ValueError, match="Input contains Inf"):
            sosfilt(sos, x)

