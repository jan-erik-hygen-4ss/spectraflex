"""Tests for spectraflex.spectrum module.

Tests cover:
- JONSWAP spectrum shape, integration to Hs, gamma parameter effects
- Pierson-Moskowitz spectrum shape and integration
- from_array() for user-defined spectra
- Edge cases and validation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectraflex.spectrum import jonswap, pierson_moskowitz, from_array


class TestJonswap:
    """Tests for the JONSWAP spectrum function."""

    def test_jonswap_returns_correct_shape(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should return array matching input frequency shape."""
        hs, tp = 3.0, 10.0
        S = jonswap(hs=hs, tp=tp, f=frequency_array)

        assert S.shape == frequency_array.shape
        assert S.dtype == np.float64

    def test_jonswap_integrates_to_hs(self, frequency_array: np.ndarray) -> None:
        """Integral of spectrum should give m0 = (Hs/4)^2."""
        hs, tp = 4.0, 12.0
        S = jonswap(hs=hs, tp=tp, f=frequency_array)

        # m0 = integral of S(f) df
        m0 = np.trapezoid(S, frequency_array)
        hs_recovered = 4.0 * np.sqrt(m0)

        # Allow 2% tolerance due to finite frequency range truncation
        assert_allclose(hs_recovered, hs, rtol=0.02)

    def test_jonswap_peak_at_correct_frequency(self, frequency_array: np.ndarray) -> None:
        """JONSWAP peak should occur at fp = 1/Tp."""
        hs, tp = 3.5, 8.0
        fp_expected = 1.0 / tp
        S = jonswap(hs=hs, tp=tp, f=frequency_array)

        peak_idx = np.argmax(S)
        fp_actual = frequency_array[peak_idx]

        # Peak should be within one frequency bin of expected
        df = frequency_array[1] - frequency_array[0]
        assert abs(fp_actual - fp_expected) < 2 * df

    def test_jonswap_gamma_1_approaches_pm(self, frequency_array: np.ndarray) -> None:
        """JONSWAP with gamma=1 should equal Pierson-Moskowitz."""
        hs, tp = 3.0, 10.0

        S_jonswap = jonswap(hs=hs, tp=tp, gamma=1.0, f=frequency_array)
        S_pm = pierson_moskowitz(hs=hs, tp=tp, f=frequency_array)

        # Should be very close (within 1% relative tolerance)
        assert_allclose(S_jonswap, S_pm, rtol=0.01)

    def test_jonswap_higher_gamma_sharper_peak(self, frequency_array: np.ndarray) -> None:
        """Higher gamma should produce a sharper spectral peak."""
        hs, tp = 3.0, 10.0

        S_gamma_1 = jonswap(hs=hs, tp=tp, gamma=1.0, f=frequency_array)
        S_gamma_3 = jonswap(hs=hs, tp=tp, gamma=3.3, f=frequency_array)
        S_gamma_5 = jonswap(hs=hs, tp=tp, gamma=5.0, f=frequency_array)

        # Peak values should increase with gamma
        assert np.max(S_gamma_3) > np.max(S_gamma_1)
        assert np.max(S_gamma_5) > np.max(S_gamma_3)

        # All should still integrate to approximately the same m0
        m0_1 = np.trapezoid(S_gamma_1, frequency_array)
        m0_3 = np.trapezoid(S_gamma_3, frequency_array)
        m0_5 = np.trapezoid(S_gamma_5, frequency_array)

        assert_allclose(m0_1, m0_3, rtol=0.05)
        assert_allclose(m0_3, m0_5, rtol=0.05)

    def test_jonswap_default_gamma(self, frequency_array: np.ndarray) -> None:
        """Default gamma should be 3.3 (standard JONSWAP)."""
        hs, tp = 3.0, 10.0

        S_default = jonswap(hs=hs, tp=tp, f=frequency_array)
        S_explicit = jonswap(hs=hs, tp=tp, gamma=3.3, f=frequency_array)

        assert_allclose(S_default, S_explicit)

    def test_jonswap_all_positive_values(self, frequency_array: np.ndarray) -> None:
        """Spectrum values should all be non-negative."""
        S = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        assert np.all(S >= 0)

    def test_jonswap_zero_at_zero_frequency(self) -> None:
        """Spectrum should be zero at f=0 (avoid division by zero)."""
        f = np.array([0.0, 0.01, 0.1, 0.5])
        S = jonswap(hs=3.0, tp=10.0, f=f)

        assert S[0] == 0.0
        assert np.all(S[1:] > 0)

    def test_jonswap_various_sea_states(self) -> None:
        """Test JONSWAP for different sea state combinations."""
        f = np.linspace(0.02, 0.5, 200)

        test_cases = [
            (0.5, 4.0),   # Calm sea
            (2.0, 8.0),   # Moderate sea
            (5.0, 12.0),  # Rough sea
            (10.0, 15.0), # Very rough sea
        ]

        for hs, tp in test_cases:
            S = jonswap(hs=hs, tp=tp, f=f)
            m0 = np.trapezoid(S, f)
            hs_recovered = 4.0 * np.sqrt(m0)

            # Each should approximately recover Hs
            assert_allclose(hs_recovered, hs, rtol=0.05, err_msg=f"Failed for Hs={hs}, Tp={tp}")

    def test_jonswap_sigma_parameters(self, frequency_array: np.ndarray) -> None:
        """Test that custom sigma_a and sigma_b parameters work."""
        hs, tp = 3.0, 10.0

        # Default sigma values
        S_default = jonswap(hs=hs, tp=tp, f=frequency_array)

        # Custom sigma values (standard JONSWAP uses 0.07/0.09)
        S_custom = jonswap(hs=hs, tp=tp, f=frequency_array, sigma_a=0.07, sigma_b=0.09)

        assert_allclose(S_default, S_custom)


class TestPiersonMoskowitz:
    """Tests for the Pierson-Moskowitz spectrum function."""

    def test_pm_returns_correct_shape(self, frequency_array: np.ndarray) -> None:
        """PM should return array matching input frequency shape."""
        S = pierson_moskowitz(hs=3.0, tp=10.0, f=frequency_array)

        assert S.shape == frequency_array.shape
        assert S.dtype == np.float64

    def test_pm_integrates_to_hs(self, frequency_array: np.ndarray) -> None:
        """Integral of PM spectrum should give m0 = (Hs/4)^2."""
        hs, tp = 4.0, 12.0
        S = pierson_moskowitz(hs=hs, tp=tp, f=frequency_array)

        m0 = np.trapezoid(S, frequency_array)
        hs_recovered = 4.0 * np.sqrt(m0)

        assert_allclose(hs_recovered, hs, rtol=0.02)

    def test_pm_peak_at_correct_frequency(self, frequency_array: np.ndarray) -> None:
        """PM peak should occur near fp = 1/Tp."""
        hs, tp = 3.5, 8.0
        fp_expected = 1.0 / tp
        S = pierson_moskowitz(hs=hs, tp=tp, f=frequency_array)

        peak_idx = np.argmax(S)
        fp_actual = frequency_array[peak_idx]

        df = frequency_array[1] - frequency_array[0]
        assert abs(fp_actual - fp_expected) < 2 * df

    def test_pm_all_positive_values(self, frequency_array: np.ndarray) -> None:
        """Spectrum values should all be non-negative."""
        S = pierson_moskowitz(hs=3.0, tp=10.0, f=frequency_array)
        assert np.all(S >= 0)

    def test_pm_zero_at_zero_frequency(self) -> None:
        """Spectrum should be zero at f=0."""
        f = np.array([0.0, 0.01, 0.1, 0.5])
        S = pierson_moskowitz(hs=3.0, tp=10.0, f=f)

        assert S[0] == 0.0
        assert np.all(S[1:] > 0)

    def test_pm_fully_developed_sea_relation(self) -> None:
        """For fully developed sea, Tp ≈ 5.0 * sqrt(Hs) should hold."""
        # This is an approximate relationship for fully developed seas
        f = np.linspace(0.02, 0.5, 200)
        hs = 4.0
        tp = 5.0 * np.sqrt(hs)  # ~10 seconds

        S = pierson_moskowitz(hs=hs, tp=tp, f=f)
        m0 = np.trapezoid(S, f)
        hs_recovered = 4.0 * np.sqrt(m0)

        assert_allclose(hs_recovered, hs, rtol=0.05)

    def test_pm_shape_characteristic(self, frequency_array: np.ndarray) -> None:
        """PM spectrum should decay as f^-5 at high frequencies."""
        hs, tp = 3.0, 10.0
        S = pierson_moskowitz(hs=hs, tp=tp, f=frequency_array)

        # Look at high frequency tail (last quarter of spectrum)
        high_f_start = len(frequency_array) * 3 // 4
        high_f = frequency_array[high_f_start:]
        high_S = S[high_f_start:]

        # Fit log-log slope: log(S) = n*log(f) + const
        # For PM, n should be approximately -5
        log_f = np.log(high_f)
        log_S = np.log(high_S)

        # Linear fit
        slope, _ = np.polyfit(log_f, log_S, 1)

        # Slope should be close to -5 (allow some tolerance due to PM formulation)
        assert slope < -4.0, f"High frequency decay slope {slope} not steep enough"


class TestFromArray:
    """Tests for the from_array function for user-defined spectra."""

    def test_from_array_returns_correct_values(self) -> None:
        """from_array should return the exact input values."""
        f = np.array([0.05, 0.1, 0.15, 0.2])
        S = np.array([0.1, 1.0, 0.5, 0.2])

        result = from_array(f=f, S=S)

        assert_allclose(result, S)

    def test_from_array_validates_shapes(self) -> None:
        """from_array should raise error if f and S have different shapes."""
        f = np.array([0.05, 0.1, 0.15])
        S = np.array([0.1, 1.0])  # Wrong length

        with pytest.raises(ValueError, match="shape|length"):
            from_array(f=f, S=S)

    def test_from_array_validates_positive_frequencies(self) -> None:
        """from_array should raise error for negative frequencies."""
        f = np.array([-0.05, 0.1, 0.15])
        S = np.array([0.1, 1.0, 0.5])

        with pytest.raises(ValueError, match="positive|negative"):
            from_array(f=f, S=S)

    def test_from_array_validates_positive_spectrum(self) -> None:
        """from_array should raise error for negative spectrum values."""
        f = np.array([0.05, 0.1, 0.15])
        S = np.array([0.1, -1.0, 0.5])  # Negative value

        with pytest.raises(ValueError, match="positive|negative"):
            from_array(f=f, S=S)

    def test_from_array_allows_zero_values(self) -> None:
        """from_array should allow zero values in spectrum."""
        f = np.array([0.05, 0.1, 0.15])
        S = np.array([0.0, 1.0, 0.0])

        result = from_array(f=f, S=S)
        assert_allclose(result, S)


class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_jonswap_invalid_hs(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should raise error for non-positive Hs."""
        with pytest.raises(ValueError):
            jonswap(hs=0.0, tp=10.0, f=frequency_array)

        with pytest.raises(ValueError):
            jonswap(hs=-1.0, tp=10.0, f=frequency_array)

    def test_jonswap_invalid_tp(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should raise error for non-positive Tp."""
        with pytest.raises(ValueError):
            jonswap(hs=3.0, tp=0.0, f=frequency_array)

        with pytest.raises(ValueError):
            jonswap(hs=3.0, tp=-1.0, f=frequency_array)

    def test_jonswap_invalid_gamma(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should raise error for gamma < 1."""
        with pytest.raises(ValueError):
            jonswap(hs=3.0, tp=10.0, gamma=0.5, f=frequency_array)

    def test_pm_invalid_hs(self, frequency_array: np.ndarray) -> None:
        """PM should raise error for non-positive Hs."""
        with pytest.raises(ValueError):
            pierson_moskowitz(hs=0.0, tp=10.0, f=frequency_array)

    def test_pm_invalid_tp(self, frequency_array: np.ndarray) -> None:
        """PM should raise error for non-positive Tp."""
        with pytest.raises(ValueError):
            pierson_moskowitz(hs=3.0, tp=0.0, f=frequency_array)

    def test_empty_frequency_array(self) -> None:
        """Functions should handle empty frequency array gracefully."""
        f = np.array([])

        S_jonswap = jonswap(hs=3.0, tp=10.0, f=f)
        S_pm = pierson_moskowitz(hs=3.0, tp=10.0, f=f)

        assert len(S_jonswap) == 0
        assert len(S_pm) == 0

    def test_single_frequency(self) -> None:
        """Functions should work with single frequency value."""
        f = np.array([0.1])

        S_jonswap = jonswap(hs=3.0, tp=10.0, f=f)
        S_pm = pierson_moskowitz(hs=3.0, tp=10.0, f=f)

        assert len(S_jonswap) == 1
        assert len(S_pm) == 1
        assert S_jonswap[0] > 0
        assert S_pm[0] > 0

    def test_very_high_frequencies(self) -> None:
        """Spectrum should decay to near-zero at very high frequencies."""
        f = np.linspace(1.0, 5.0, 100)  # Much higher than typical wave frequencies

        S = jonswap(hs=3.0, tp=10.0, f=f)

        # All values should be very small
        assert np.all(S < 1e-6)


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_jonswap_very_small_hs(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should handle very small Hs without underflow."""
        S = jonswap(hs=0.01, tp=5.0, f=frequency_array)

        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)

    def test_jonswap_very_large_hs(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should handle very large Hs without overflow."""
        S = jonswap(hs=20.0, tp=15.0, f=frequency_array)

        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)

    def test_jonswap_very_small_tp(self) -> None:
        """JONSWAP should handle very small Tp (high frequency peak)."""
        f = np.linspace(0.1, 2.0, 200)  # Higher frequency range
        S = jonswap(hs=1.0, tp=2.0, f=f)

        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)

    def test_jonswap_very_large_tp(self, frequency_array: np.ndarray) -> None:
        """JONSWAP should handle very large Tp (low frequency peak)."""
        S = jonswap(hs=5.0, tp=25.0, f=frequency_array)

        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)
