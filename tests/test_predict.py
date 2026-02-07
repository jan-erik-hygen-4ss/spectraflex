"""Tests for spectraflex.predict module.

Tests cover:
- response_spectrum(): S_yy = |H|² · S_xx
- statistics(): spectral moments, Hs, Tp, MPM from response spectrum
- synthesize_timeseries(): spectral synthesis with random phases
- All tests use synthetic data with known transfer functions
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from spectraflex.predict import response_spectrum, statistics, synthesize_timeseries
from spectraflex.spectrum import jonswap, pierson_moskowitz
from spectraflex.transfer_function import create as create_tf


def create_simple_transfer_function(
    frequency: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray | None = None,
    variable_names: list[str] | None = None,
) -> xr.Dataset:
    """Helper to create a simple transfer function dataset for testing."""
    if phase is None:
        phase = np.zeros_like(magnitude)
    if variable_names is None:
        if magnitude.ndim == 1:
            variable_names = ["response"]
            magnitude = magnitude[:, np.newaxis]
            phase = phase[:, np.newaxis]
        else:
            variable_names = [f"var_{i}" for i in range(magnitude.shape[1])]

    coherence = np.ones_like(magnitude)

    return create_tf(
        frequency=frequency,
        magnitude=magnitude,
        phase=phase,
        coherence=coherence,
        variable_names=variable_names,
    )


class TestResponseSpectrum:
    """Tests for the response_spectrum function."""

    def test_response_spectrum_shape(self, frequency_array: np.ndarray) -> None:
        """Response spectrum should have correct shape."""
        # Create transfer function with 2 variables
        H_mag = np.column_stack([
            np.ones_like(frequency_array),
            2.0 * np.ones_like(frequency_array),
        ])
        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["var1", "var2"]
        )

        # Input spectrum
        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        # Compute response
        S_yy = response_spectrum(tf, S_xx)

        assert isinstance(S_yy, xr.Dataset)
        assert "Syy" in S_yy.data_vars
        assert S_yy.Syy.shape == (len(frequency_array), 2)

    def test_response_spectrum_unity_transfer_function(
        self, frequency_array: np.ndarray
    ) -> None:
        """With |H(f)| = 1, response spectrum should equal input spectrum."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        S_yy = response_spectrum(tf, S_xx)

        assert_allclose(S_yy.Syy.values[:, 0], S_xx, rtol=1e-10)

    def test_response_spectrum_constant_gain(
        self, frequency_array: np.ndarray
    ) -> None:
        """With |H(f)| = k, response spectrum should be k² × input spectrum."""
        gain = 2.5
        H_mag = gain * np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        S_yy = response_spectrum(tf, S_xx)

        expected = (gain**2) * S_xx
        assert_allclose(S_yy.Syy.values[:, 0], expected, rtol=1e-10)

    def test_response_spectrum_frequency_dependent_gain(
        self, frequency_array: np.ndarray
    ) -> None:
        """Test with frequency-dependent transfer function."""
        # Linear gain: |H(f)| = f
        H_mag = frequency_array[:, np.newaxis]
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = pierson_moskowitz(hs=4.0, tp=12.0, f=frequency_array)
        S_yy = response_spectrum(tf, S_xx)

        expected = (frequency_array**2) * S_xx
        assert_allclose(S_yy.Syy.values[:, 0], expected, rtol=1e-10)

    def test_response_spectrum_resonance(self, frequency_array: np.ndarray) -> None:
        """Test with resonant transfer function (single peak)."""
        f0 = 0.1  # Resonance frequency
        bandwidth = 0.02

        # Create resonance peak in |H(f)|
        H_mag = 5.0 * np.exp(-((frequency_array - f0) ** 2) / (2 * bandwidth**2))
        H_mag = H_mag[:, np.newaxis]
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = np.ones_like(frequency_array)  # White noise input
        S_yy = response_spectrum(tf, S_xx)

        # Response should have peak at resonance frequency
        peak_idx = np.argmax(S_yy.Syy.values[:, 0])
        peak_freq = frequency_array[peak_idx]

        assert abs(peak_freq - f0) < 2 * (frequency_array[1] - frequency_array[0])

    def test_response_spectrum_multiple_variables(
        self, frequency_array: np.ndarray
    ) -> None:
        """Test with multiple response variables."""
        gains = [1.0, 2.0, 0.5]
        H_mag = np.column_stack([g * np.ones_like(frequency_array) for g in gains])
        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["UFJ", "LFJ", "WH_BM"]
        )

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        S_yy = response_spectrum(tf, S_xx)

        for i, g in enumerate(gains):
            expected = (g**2) * S_xx
            assert_allclose(S_yy.Syy.values[:, i], expected, rtol=1e-10)

    def test_response_spectrum_preserves_zero_input(
        self, frequency_array: np.ndarray
    ) -> None:
        """Zero input spectrum should give zero response."""
        H_mag = 5.0 * np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = np.zeros_like(frequency_array)
        S_yy = response_spectrum(tf, S_xx)

        assert_allclose(S_yy.Syy.values[:, 0], 0.0)

    def test_response_spectrum_phase_ignored(
        self, frequency_array: np.ndarray
    ) -> None:
        """Phase should not affect power spectrum (only |H|² matters)."""
        H_mag = 2.0 * np.ones_like(frequency_array)

        # Create TF with zero phase
        tf_zero_phase = create_simple_transfer_function(
            frequency_array,
            H_mag[:, np.newaxis],
            phase=np.zeros((len(frequency_array), 1)),
        )

        # Create TF with random phase
        rng = np.random.default_rng(42)
        random_phase = rng.uniform(0, 2 * np.pi, (len(frequency_array), 1))
        tf_random_phase = create_simple_transfer_function(
            frequency_array, H_mag[:, np.newaxis], phase=random_phase
        )

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        S_yy_zero = response_spectrum(tf_zero_phase, S_xx)
        S_yy_random = response_spectrum(tf_random_phase, S_xx)

        assert_allclose(S_yy_zero.Syy.values, S_yy_random.Syy.values, rtol=1e-10)


class TestStatistics:
    """Tests for the statistics function."""

    def test_statistics_returns_dict(self, frequency_array: np.ndarray) -> None:
        """statistics should return a dict with per-variable stats."""
        H_mag = np.ones((len(frequency_array), 2))
        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["var1", "var2"]
        )

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        assert isinstance(stats, dict)
        assert "var1" in stats
        assert "var2" in stats

    def test_statistics_contents(self, frequency_array: np.ndarray) -> None:
        """Each variable should have m0, hs, tp, mpm_3h stats."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        var_stats = stats["response"]
        assert "m0" in var_stats
        assert "hs" in var_stats
        assert "tp" in var_stats or "tz" in var_stats
        assert "mpm_3h" in var_stats or "mpm" in var_stats

    def test_statistics_unity_tf_matches_input(
        self, frequency_array: np.ndarray
    ) -> None:
        """With |H| = 1, response Hs should match input Hs."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        hs_input = 4.0
        S_xx = jonswap(hs=hs_input, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        hs_response = stats["response"]["hs"]
        assert_allclose(hs_response, hs_input, rtol=0.02)

    def test_statistics_gain_scaling(self, frequency_array: np.ndarray) -> None:
        """Response Hs should scale linearly with |H|."""
        gain = 2.0
        H_mag = gain * np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        hs_input = 3.0
        S_xx = jonswap(hs=hs_input, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        # Hs_response = 4*sqrt(m0_response) = 4*sqrt(|H|^2 * m0_input)
        # = |H| * 4*sqrt(m0_input) = |H| * Hs_input
        hs_expected = gain * hs_input
        assert_allclose(stats["response"]["hs"], hs_expected, rtol=0.02)

    def test_statistics_mpm_exceeds_hs(self, frequency_array: np.ndarray) -> None:
        """3-hour MPM should exceed Hs for typical narrow-band spectrum."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=4.0, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        hs = stats["response"]["hs"]
        mpm = stats["response"].get("mpm_3h", stats["response"].get("mpm"))

        assert mpm > hs

    def test_statistics_multiple_variables(
        self, frequency_array: np.ndarray
    ) -> None:
        """Test statistics for multiple response variables."""
        gains = [1.0, 2.0, 3.0]
        H_mag = np.column_stack([g * np.ones_like(frequency_array) for g in gains])
        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["A", "B", "C"]
        )

        hs_input = 3.0
        S_xx = jonswap(hs=hs_input, tp=10.0, f=frequency_array)
        stats = statistics(tf, S_xx)

        for i, (name, g) in enumerate(zip(["A", "B", "C"], gains)):
            expected_hs = g * hs_input
            assert_allclose(stats[name]["hs"], expected_hs, rtol=0.02)


class TestSynthesizeTimeseries:
    """Tests for the synthesize_timeseries function."""

    def test_timeseries_shape(self, frequency_array: np.ndarray) -> None:
        """Synthesized time series should have correct shape."""
        H_mag = np.ones((len(frequency_array), 2))
        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["var1", "var2"]
        )

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        duration = 100.0
        dt = 0.1

        ts = synthesize_timeseries(tf, S_xx, duration=duration, dt=dt, seed=42)

        assert isinstance(ts, xr.Dataset)
        expected_samples = int(duration / dt)
        assert ts.dims["time"] == expected_samples or abs(ts.dims["time"] - expected_samples) <= 1
        assert ts.dims["variable"] == 2

    def test_timeseries_has_time_coord(self, frequency_array: np.ndarray) -> None:
        """Time series should have time coordinate in seconds."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        ts = synthesize_timeseries(tf, S_xx, duration=50.0, dt=0.1, seed=42)

        assert "time" in ts.coords
        assert ts.time.values[0] >= 0
        assert_allclose(np.diff(ts.time.values), 0.1, rtol=0.01)

    def test_timeseries_reproducible_with_seed(
        self, frequency_array: np.ndarray
    ) -> None:
        """Same seed should give identical time series."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts1 = synthesize_timeseries(tf, S_xx, duration=50.0, dt=0.1, seed=42)
        ts2 = synthesize_timeseries(tf, S_xx, duration=50.0, dt=0.1, seed=42)

        assert_allclose(ts1.response.values, ts2.response.values)

    def test_timeseries_different_with_different_seed(
        self, frequency_array: np.ndarray
    ) -> None:
        """Different seeds should give different time series."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts1 = synthesize_timeseries(tf, S_xx, duration=50.0, dt=0.1, seed=42)
        ts2 = synthesize_timeseries(tf, S_xx, duration=50.0, dt=0.1, seed=123)

        # Should not be identical
        assert not np.allclose(ts1.response.values, ts2.response.values)

    def test_timeseries_statistics_match_spectrum(
        self, frequency_array: np.ndarray
    ) -> None:
        """Time series statistics should approximately match spectral prediction."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        hs_input = 4.0
        S_xx = jonswap(hs=hs_input, tp=10.0, f=frequency_array)

        # Generate long time series for better statistics
        duration = 1800.0  # 30 minutes
        dt = 0.1
        ts = synthesize_timeseries(tf, S_xx, duration=duration, dt=dt, seed=42)

        # Compute Hs from time series
        # Hs = 4 * std for zero-mean Gaussian
        data = ts.response.values.flatten()
        data = data - np.mean(data)  # Remove any DC offset
        std = np.std(data)
        hs_from_ts = 4.0 * std

        # Should be close to input Hs (allow 10% tolerance due to randomness)
        assert_allclose(hs_from_ts, hs_input, rtol=0.15)

    def test_timeseries_zero_mean(self, frequency_array: np.ndarray) -> None:
        """Synthesized time series should have approximately zero mean."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)
        ts = synthesize_timeseries(tf, S_xx, duration=500.0, dt=0.1, seed=42)

        mean = np.mean(ts.response.values)
        std = np.std(ts.response.values)

        # Mean should be small compared to std
        assert abs(mean) < 0.1 * std

    def test_timeseries_applies_transfer_function(
        self, frequency_array: np.ndarray
    ) -> None:
        """Transfer function gain should affect time series amplitude."""
        gain = 3.0
        H_mag_unity = np.ones((len(frequency_array), 1))
        H_mag_scaled = gain * np.ones((len(frequency_array), 1))

        tf_unity = create_simple_transfer_function(frequency_array, H_mag_unity)
        tf_scaled = create_simple_transfer_function(frequency_array, H_mag_scaled)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts_unity = synthesize_timeseries(tf_unity, S_xx, duration=500.0, dt=0.1, seed=42)
        ts_scaled = synthesize_timeseries(tf_scaled, S_xx, duration=500.0, dt=0.1, seed=42)

        std_unity = np.std(ts_unity.response.values)
        std_scaled = np.std(ts_scaled.response.values)

        # Std should scale with |H|
        assert_allclose(std_scaled / std_unity, gain, rtol=0.1)

    def test_timeseries_phase_affects_waveform(
        self, frequency_array: np.ndarray
    ) -> None:
        """Phase in transfer function should affect time series shape."""
        H_mag = np.ones_like(frequency_array)

        # Zero phase
        tf_zero = create_simple_transfer_function(
            frequency_array,
            H_mag[:, np.newaxis],
            phase=np.zeros((len(frequency_array), 1)),
        )

        # Linear phase (delay)
        delay = 0.5  # seconds
        linear_phase = -2 * np.pi * frequency_array * delay
        tf_delayed = create_simple_transfer_function(
            frequency_array, H_mag[:, np.newaxis], phase=linear_phase[:, np.newaxis]
        )

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts_zero = synthesize_timeseries(tf_zero, S_xx, duration=100.0, dt=0.1, seed=42)
        ts_delayed = synthesize_timeseries(tf_delayed, S_xx, duration=100.0, dt=0.1, seed=42)

        # Waveforms should be different (shifted)
        assert not np.allclose(ts_zero.response.values, ts_delayed.response.values)

        # But should have same statistics
        assert_allclose(
            np.std(ts_zero.response.values),
            np.std(ts_delayed.response.values),
            rtol=0.01,
        )


class TestPredictIntegration:
    """Integration tests combining predict functions."""

    def test_full_prediction_workflow(self, frequency_array: np.ndarray) -> None:
        """Test complete prediction workflow: TF + spectrum → response → stats → timeseries."""
        # Create a realistic transfer function (simple resonance)
        f0 = 0.1  # Resonance at 10s period
        Q = 5.0  # Quality factor
        H_mag = 1.0 / np.sqrt((1 - (frequency_array / f0) ** 2) ** 2 + (frequency_array / (Q * f0)) ** 2)
        H_mag = np.clip(H_mag, 0, 10)  # Limit peak amplitude
        H_mag = H_mag[:, np.newaxis]

        tf = create_simple_transfer_function(
            frequency_array, H_mag, variable_names=["UFJ_Angle"]
        )

        # Input sea state
        S_xx = jonswap(hs=4.0, tp=10.0, f=frequency_array)

        # Compute response spectrum
        S_yy = response_spectrum(tf, S_xx)
        assert np.all(S_yy.Syy.values >= 0)

        # Get statistics
        stats = statistics(tf, S_xx)
        assert stats["UFJ_Angle"]["hs"] > 0
        assert stats["UFJ_Angle"]["m0"] > 0

        # Synthesize time series
        ts = synthesize_timeseries(tf, S_xx, duration=600.0, dt=0.1, seed=42)
        assert len(ts.time) > 0

    def test_prediction_consistency(self, frequency_array: np.ndarray) -> None:
        """Statistics from response_spectrum should match statistics function."""
        gain = 2.0
        H_mag = gain * np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=4.0, tp=10.0, f=frequency_array)

        # Method 1: Get stats directly
        stats = statistics(tf, S_xx)
        hs_direct = stats["response"]["hs"]

        # Method 2: Compute spectrum, then integrate
        S_yy = response_spectrum(tf, S_xx)
        m0 = np.trapezoid(S_yy.Syy.values[:, 0], frequency_array)
        hs_from_spectrum = 4.0 * np.sqrt(m0)

        assert_allclose(hs_direct, hs_from_spectrum, rtol=0.01)


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_response_spectrum_frequency_mismatch(self) -> None:
        """Should handle or raise error for mismatched frequencies."""
        f_tf = np.linspace(0.01, 0.5, 100)
        f_spectrum = np.linspace(0.02, 0.6, 150)  # Different range

        H_mag = np.ones((len(f_tf), 1))
        tf = create_simple_transfer_function(f_tf, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=f_spectrum)

        # Should either interpolate or raise clear error
        try:
            S_yy = response_spectrum(tf, S_xx)
            # If it works, result should be reasonable
            assert np.all(np.isfinite(S_yy.Syy.values))
        except (ValueError, KeyError):
            pass  # Also acceptable

    def test_synthesize_very_short_duration(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should handle very short duration."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts = synthesize_timeseries(tf, S_xx, duration=1.0, dt=0.1, seed=42)
        assert len(ts.time) >= 1

    def test_synthesize_very_fine_dt(self, frequency_array: np.ndarray) -> None:
        """Should handle very fine time resolution."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = jonswap(hs=3.0, tp=10.0, f=frequency_array)

        ts = synthesize_timeseries(tf, S_xx, duration=10.0, dt=0.01, seed=42)
        assert len(ts.time) >= 900  # Should be ~1000 samples

    def test_zero_spectrum_input(self, frequency_array: np.ndarray) -> None:
        """Zero input spectrum should give zero response."""
        H_mag = np.ones((len(frequency_array), 1))
        tf = create_simple_transfer_function(frequency_array, H_mag)

        S_xx = np.zeros_like(frequency_array)

        S_yy = response_spectrum(tf, S_xx)
        assert_allclose(S_yy.Syy.values, 0.0)

        stats = statistics(tf, S_xx)
        assert stats["response"]["m0"] == 0.0 or np.isclose(stats["response"]["m0"], 0.0)
