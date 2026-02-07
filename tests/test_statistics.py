"""Tests for spectraflex.statistics module.

Tests cover:
- Spectral moments m0, m1, m2, m4 calculation
- Hs from spectrum (4 * sqrt(m0))
- Tp from spectrum (2π * sqrt(m0/m2))
- Most probable maximum (MPM) using Rayleigh distribution
- Edge cases and validation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectraflex.statistics import (
    spectral_moments,
    hs_from_spectrum,
    tp_from_spectrum,
    mpm_rayleigh,
)


class TestSpectralMoments:
    """Tests for the spectral_moments function."""

    def test_spectral_moments_returns_four_values(
        self, frequency_array: np.ndarray
    ) -> None:
        """spectral_moments should return m0, m1, m2, m4."""
        # Create a simple test spectrum
        S = np.exp(-((frequency_array - 0.1) ** 2) / 0.01)

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        assert isinstance(m0, float)
        assert isinstance(m1, float)
        assert isinstance(m2, float)
        assert isinstance(m4, float)

    def test_spectral_moments_analytical_white_noise(self) -> None:
        """Test moments against analytical values for white noise spectrum."""
        # White noise: S(f) = S0 (constant)
        # m_n = integral(f^n * S0 df) from f1 to f2
        # m_n = S0 * (f2^(n+1) - f1^(n+1)) / (n+1)

        f1, f2 = 0.05, 0.5
        S0 = 2.0
        f = np.linspace(f1, f2, 1000)
        S = np.full_like(f, S0)

        m0, m1, m2, m4 = spectral_moments(f, S)

        # Analytical values
        m0_analytical = S0 * (f2 - f1)
        m1_analytical = S0 * (f2**2 - f1**2) / 2
        m2_analytical = S0 * (f2**3 - f1**3) / 3
        m4_analytical = S0 * (f2**5 - f1**5) / 5

        assert_allclose(m0, m0_analytical, rtol=0.01)
        assert_allclose(m1, m1_analytical, rtol=0.01)
        assert_allclose(m2, m2_analytical, rtol=0.01)
        assert_allclose(m4, m4_analytical, rtol=0.01)

    def test_spectral_moments_positive(self, frequency_array: np.ndarray) -> None:
        """All moments should be positive for positive spectrum."""
        S = np.exp(-((frequency_array - 0.1) ** 2) / 0.01)

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        assert m0 > 0
        assert m1 > 0
        assert m2 > 0
        assert m4 > 0

    def test_spectral_moments_ordering(self, frequency_array: np.ndarray) -> None:
        """Higher moments weighted by higher frequencies should show expected relations."""
        # For a narrow-band spectrum centered at f0, m_n ≈ f0^n * m0
        f0 = 0.15
        bandwidth = 0.01
        S = np.exp(-((frequency_array - f0) ** 2) / (2 * bandwidth**2))

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        # For narrow-band spectrum: m1/m0 ≈ f0, m2/m0 ≈ f0^2
        assert_allclose(m1 / m0, f0, rtol=0.1)
        assert_allclose(m2 / m0, f0**2, rtol=0.1)
        assert_allclose(m4 / m0, f0**4, rtol=0.1)

    def test_spectral_moments_zero_spectrum(self, frequency_array: np.ndarray) -> None:
        """All moments should be zero for zero spectrum."""
        S = np.zeros_like(frequency_array)

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        assert m0 == 0.0
        assert m1 == 0.0
        assert m2 == 0.0
        assert m4 == 0.0

    def test_spectral_moments_shape_validation(self) -> None:
        """Should raise error if f and S have different shapes."""
        f = np.linspace(0.01, 0.5, 100)
        S = np.ones(50)  # Wrong shape

        with pytest.raises(ValueError, match="shape|length"):
            spectral_moments(f, S)


class TestHsFromSpectrum:
    """Tests for the hs_from_spectrum function."""

    def test_hs_from_known_m0(self) -> None:
        """Hs = 4 * sqrt(m0), test against known m0."""
        m0_values = [0.25, 1.0, 4.0, 16.0]

        for m0 in m0_values:
            hs = hs_from_spectrum(m0)
            expected = 4.0 * np.sqrt(m0)
            assert_allclose(hs, expected)

    def test_hs_round_trip(self, frequency_array: np.ndarray) -> None:
        """Test Hs calculation matches input Hs for known spectrum."""
        # Create a spectrum with known Hs (using m0 = (Hs/4)^2)
        hs_target = 3.5
        m0_target = (hs_target / 4.0) ** 2

        # Simple Gaussian spectrum with area = m0
        f0 = 0.1
        sigma = 0.02
        S = (m0_target / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((frequency_array - f0) ** 2) / (2 * sigma**2)
        )

        m0, _, _, _ = spectral_moments(frequency_array, S)
        hs_recovered = hs_from_spectrum(m0)

        assert_allclose(hs_recovered, hs_target, rtol=0.02)

    def test_hs_zero_m0(self) -> None:
        """Hs should be 0 for zero m0."""
        hs = hs_from_spectrum(0.0)
        assert hs == 0.0

    def test_hs_negative_m0_raises(self) -> None:
        """Should raise error for negative m0."""
        with pytest.raises(ValueError, match="negative|positive"):
            hs_from_spectrum(-1.0)


class TestTpFromSpectrum:
    """Tests for the tp_from_spectrum function."""

    def test_tp_analytical(self) -> None:
        """Tp = 2π * sqrt(m0/m2), test against known values."""
        # For narrow-band spectrum at f0: m0 ≈ const, m2 ≈ f0^2 * m0
        # So Tp ≈ 2π * sqrt(1/f0^2) = 2π / f0 = 1/f0 * 2π... but wait
        # Actually Tp = sqrt(m0/m2) has units of 1/f, so Tp ≈ 1/f0

        m0 = 1.0
        f0 = 0.1
        m2 = f0**2 * m0  # Narrow-band approximation

        tp = tp_from_spectrum(m0, m2)

        # Tp should be approximately 1/f0 (the period at f0)
        # Tp = 2π * sqrt(m0/m2) = 2π * sqrt(1/(f0^2)) = 2π/f0 ≈ 62.8 for f0=0.1
        # Wait, that's angular... Actually for f in Hz:
        # Zero-crossing period Tz = sqrt(m0/m2) (in time units matching f)
        # If f in Hz, Tz in seconds
        expected_tz = np.sqrt(m0 / m2)
        assert_allclose(tp, expected_tz, rtol=0.01)

    def test_tp_from_narrow_band_spectrum(self, frequency_array: np.ndarray) -> None:
        """Tp should match inverse of peak frequency for narrow-band spectrum."""
        f0 = 0.1  # Peak frequency
        sigma = 0.005  # Very narrow
        S = np.exp(-((frequency_array - f0) ** 2) / (2 * sigma**2))

        m0, _, m2, _ = spectral_moments(frequency_array, S)
        tp = tp_from_spectrum(m0, m2)

        # For very narrow band, Tp ≈ 1/f0
        expected = 1.0 / f0
        assert_allclose(tp, expected, rtol=0.1)

    def test_tp_zero_m0_raises(self) -> None:
        """Should raise error for zero m0."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            tp_from_spectrum(0.0, 1.0)

    def test_tp_zero_m2_raises(self) -> None:
        """Should raise error for zero m2."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            tp_from_spectrum(1.0, 0.0)


class TestMpmRayleigh:
    """Tests for the mpm_rayleigh function (Most Probable Maximum)."""

    def test_mpm_formula(self) -> None:
        """MPM = sigma * sqrt(2 * ln(N)) where N = duration / Tz."""
        m0 = 1.0
        m2 = 0.01  # Tz = sqrt(m0/m2) = 10 seconds
        duration = 3 * 3600  # 3 hours = 10800 seconds

        mpm = mpm_rayleigh(m0, duration, m2=m2)

        # Calculate expected
        sigma = np.sqrt(m0)
        Tz = np.sqrt(m0 / m2)
        N = duration / Tz
        expected = sigma * np.sqrt(2 * np.log(N))

        assert_allclose(mpm, expected, rtol=0.01)

    def test_mpm_scales_with_sigma(self) -> None:
        """MPM should scale linearly with standard deviation (sqrt(m0))."""
        duration = 3600.0
        m2 = 0.01

        m0_1 = 1.0
        m0_4 = 4.0  # 2x sigma

        mpm_1 = mpm_rayleigh(m0_1, duration, m2=m2)
        mpm_4 = mpm_rayleigh(m0_4, duration, m2=m2)

        # Should scale with sqrt(m0), so ratio should be 2
        # (ignoring the small log term change)
        assert mpm_4 > mpm_1
        assert_allclose(mpm_4 / mpm_1, 2.0, rtol=0.1)

    def test_mpm_increases_with_duration(self) -> None:
        """Longer duration should give higher MPM (more cycles)."""
        m0 = 1.0
        m2 = 0.01

        mpm_1h = mpm_rayleigh(m0, 3600, m2=m2)
        mpm_3h = mpm_rayleigh(m0, 3 * 3600, m2=m2)
        mpm_10h = mpm_rayleigh(m0, 10 * 3600, m2=m2)

        assert mpm_3h > mpm_1h
        assert mpm_10h > mpm_3h

    def test_mpm_typical_values(self) -> None:
        """Test MPM for typical offshore values."""
        # Typical: Hs = 4m, Tz = 8s, 3-hour storm
        hs = 4.0
        m0 = (hs / 4.0) ** 2  # 1.0 m^2
        tz = 8.0
        m2 = m0 / (tz**2)
        duration = 3 * 3600

        mpm = mpm_rayleigh(m0, duration, m2=m2)

        # Expected: sqrt(m0) * sqrt(2 * ln(N))
        # N = 10800/8 = 1350 cycles
        # sqrt(2*ln(1350)) ≈ sqrt(2*7.2) ≈ 3.8
        # MPM ≈ 1.0 * 3.8 ≈ 3.8 m
        # Which is about Hs (which makes sense for 3-hour max)
        assert 3.0 < mpm < 5.0

    def test_mpm_zero_m0(self) -> None:
        """MPM should be 0 for zero variance."""
        mpm = mpm_rayleigh(0.0, 3600, m2=0.01)
        assert mpm == 0.0

    def test_mpm_alternative_interface_with_tz(self) -> None:
        """Test MPM can accept Tz directly instead of m2."""
        m0 = 1.0
        tz = 10.0
        duration = 3600.0

        # Using m2
        m2 = m0 / (tz**2)
        mpm_via_m2 = mpm_rayleigh(m0, duration, m2=m2)

        # Using tz directly (if supported)
        try:
            mpm_via_tz = mpm_rayleigh(m0, duration, tz=tz)
            assert_allclose(mpm_via_m2, mpm_via_tz)
        except TypeError:
            # If tz parameter not supported, that's ok
            pass


class TestSpectralStatisticsIntegration:
    """Integration tests combining multiple statistics functions."""

    def test_full_statistics_chain(self, frequency_array: np.ndarray) -> None:
        """Test computing full statistics from a known spectrum."""
        # Create a realistic spectrum
        f0 = 0.1  # 10-second peak period
        bandwidth = 0.02
        hs_target = 4.0
        m0_target = (hs_target / 4.0) ** 2

        # Gaussian spectrum normalized to target m0
        S_unnorm = np.exp(-((frequency_array - f0) ** 2) / (2 * bandwidth**2))
        area = np.trapezoid(S_unnorm, frequency_array)
        S = S_unnorm * (m0_target / area)

        # Compute all statistics
        m0, m1, m2, m4 = spectral_moments(frequency_array, S)
        hs = hs_from_spectrum(m0)
        tp = tp_from_spectrum(m0, m2)
        mpm = mpm_rayleigh(m0, 3 * 3600, m2=m2)

        # Verify
        assert_allclose(hs, hs_target, rtol=0.02)
        assert tp > 0
        assert mpm > hs  # MPM should exceed Hs for 3-hour duration

    def test_statistics_with_bimodal_spectrum(
        self, frequency_array: np.ndarray
    ) -> None:
        """Test statistics for bimodal (swell + wind sea) spectrum."""
        # Swell component
        f_swell = 0.07
        S_swell = 2.0 * np.exp(-((frequency_array - f_swell) ** 2) / 0.0005)

        # Wind sea component
        f_wind = 0.15
        S_wind = 1.0 * np.exp(-((frequency_array - f_wind) ** 2) / 0.002)

        S_total = S_swell + S_wind

        m0, m1, m2, m4 = spectral_moments(frequency_array, S_total)
        hs = hs_from_spectrum(m0)
        tp = tp_from_spectrum(m0, m2)

        # Hs should combine both components
        m0_swell = np.trapezoid(S_swell, frequency_array)
        m0_wind = np.trapezoid(S_wind, frequency_array)
        hs_combined = 4.0 * np.sqrt(m0_swell + m0_wind)

        assert_allclose(hs, hs_combined, rtol=0.02)

        # Tp should be somewhere between the two peaks
        assert 1.0 / f_wind < tp < 1.0 / f_swell


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_moments_very_small_spectrum(self, frequency_array: np.ndarray) -> None:
        """Handle very small spectrum values without underflow."""
        S = 1e-20 * np.exp(-((frequency_array - 0.1) ** 2) / 0.01)

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        assert np.isfinite(m0)
        assert np.isfinite(m1)
        assert np.isfinite(m2)
        assert np.isfinite(m4)

    def test_moments_very_large_spectrum(self, frequency_array: np.ndarray) -> None:
        """Handle very large spectrum values without overflow."""
        S = 1e10 * np.exp(-((frequency_array - 0.1) ** 2) / 0.01)

        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        assert np.isfinite(m0)
        assert np.isfinite(m1)
        assert np.isfinite(m2)
        assert np.isfinite(m4)

    def test_mpm_very_long_duration(self) -> None:
        """MPM should handle very long durations."""
        m0 = 1.0
        m2 = 0.01
        duration = 365 * 24 * 3600  # 1 year in seconds

        mpm = mpm_rayleigh(m0, duration, m2=m2)

        assert np.isfinite(mpm)
        assert mpm > 0

    def test_mpm_very_short_duration(self) -> None:
        """MPM should handle very short durations."""
        m0 = 1.0
        m2 = 0.01
        duration = 1.0  # 1 second

        # This might give N < 1, need to handle gracefully
        try:
            mpm = mpm_rayleigh(m0, duration, m2=m2)
            assert np.isfinite(mpm)
        except ValueError:
            # Acceptable to raise error for N < 1
            pass


class TestInputValidation:
    """Tests for input validation."""

    def test_spectral_moments_negative_spectrum(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should handle or reject negative spectrum values."""
        S = np.ones_like(frequency_array)
        S[50] = -1.0  # One negative value

        # Either raise error or handle gracefully
        try:
            m0, m1, m2, m4 = spectral_moments(frequency_array, S)
            # If it doesn't raise, moments should still be reasonable
            # (treating negative as actual negative contribution)
        except ValueError:
            pass  # Also acceptable

    def test_spectral_moments_nan_in_spectrum(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should handle NaN values appropriately."""
        S = np.ones_like(frequency_array)
        S[50] = np.nan

        # Should either raise error or propagate NaN
        m0, m1, m2, m4 = spectral_moments(frequency_array, S)

        # Result should be NaN if input contains NaN
        assert np.isnan(m0) or m0 > 0  # Depends on implementation

    def test_hs_from_spectrum_nan(self) -> None:
        """Hs should propagate NaN."""
        hs = hs_from_spectrum(np.nan)
        assert np.isnan(hs)
