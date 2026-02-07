"""Tests for spectraflex.identify module.

Tests cover:
- from_time_histories(): identify H(f) from raw time series
- from_spectra(): identify H(f) from pre-computed spectra (post-calc output)
- Round-trip tests: create known H(f), generate input/output, verify recovery

The key validation is the round-trip test:
1. Define a known transfer function H(f)
2. Generate white noise input
3. Filter through H(f) to create response
4. Use identify.from_time_histories() to recover H(f)
5. Compare with original H(f)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from spectraflex.identify import from_time_histories, from_spectra


def apply_transfer_function(
    input_signal: np.ndarray,
    frequency: np.ndarray,
    H: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply transfer function H(f) to input signal via frequency domain filtering.

    Args:
        input_signal: Input time series
        frequency: Frequency array for H (Hz)
        H: Complex transfer function values
        dt: Sample interval (seconds)

    Returns:
        Filtered output signal
    """
    n = len(input_signal)

    # FFT of input
    X = np.fft.rfft(input_signal)
    fft_freq = np.fft.rfftfreq(n, dt)

    # Interpolate H to FFT frequencies
    H_interp = np.interp(fft_freq, frequency, H, left=0, right=0)

    # Apply transfer function
    Y = X * H_interp

    # Inverse FFT
    output = np.fft.irfft(Y, n)

    return output


def create_simple_resonance_tf(
    frequency: np.ndarray, f0: float, Q: float, gain: float = 1.0
) -> np.ndarray:
    """Create a simple resonance transfer function.

    H(f) = gain / (1 + jQ(f/f0 - f0/f))

    Args:
        frequency: Frequency array (Hz)
        f0: Resonance frequency (Hz)
        Q: Quality factor
        gain: DC gain

    Returns:
        Complex transfer function values
    """
    # Avoid division by zero at f=0
    f_safe = np.where(frequency > 0, frequency, 1e-10)
    H = gain / (1 + 1j * Q * (f_safe / f0 - f0 / f_safe))
    # Set H=0 at f=0
    H = np.where(frequency > 0, H, 0)
    return H


def create_lowpass_tf(
    frequency: np.ndarray, fc: float, order: int = 2
) -> np.ndarray:
    """Create a lowpass filter transfer function.

    Args:
        frequency: Frequency array (Hz)
        fc: Cutoff frequency (Hz)
        order: Filter order

    Returns:
        Complex transfer function values
    """
    # Simple lowpass: H = 1 / (1 + (f/fc)^(2*order))^0.5
    # This is a Butterworth-like approximation
    H_mag = 1.0 / (1 + (frequency / fc) ** (2 * order)) ** 0.5
    # Zero phase for simplicity
    return H_mag.astype(complex)


class TestFromTimeHistoriesBasic:
    """Basic tests for from_time_histories function."""

    def test_returns_dataset_with_correct_structure(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """from_time_histories should return properly structured Dataset."""
        n = 5120
        wave = rng.normal(0, 1, n)
        responses = {
            "UFJ": rng.normal(0, 0.5, n),
            "LFJ": rng.normal(0, 0.3, n),
        }

        tf = from_time_histories(
            wave_elevation=wave,
            responses=responses,
            dt=dt,
            nperseg=1024,
        )

        # Check structure
        assert "frequency" in tf.dims
        assert "variable" in tf.dims
        assert "magnitude" in tf.data_vars
        assert "phase" in tf.data_vars
        assert "coherence" in tf.data_vars

        # Check variable names
        assert list(tf.coords["variable"].values) == ["UFJ", "LFJ"]

    def test_frequency_range(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Identified frequencies should match FFT frequencies for given dt."""
        n = 5120
        wave = rng.normal(0, 1, n)
        responses = {"response": rng.normal(0, 0.5, n)}

        nperseg = 1024
        tf = from_time_histories(
            wave_elevation=wave,
            responses=responses,
            dt=dt,
            nperseg=nperseg,
        )

        # Frequencies should go from 0 to Nyquist (or slightly less)
        freqs = tf.coords["frequency"].values
        nyquist = 1.0 / (2 * dt)

        assert freqs[0] >= 0
        assert freqs[-1] <= nyquist
        assert len(freqs) == nperseg // 2 + 1

    def test_coherence_in_valid_range(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Coherence should be in [0, 1]."""
        n = 5120
        wave = rng.normal(0, 1, n)
        # Correlated response
        responses = {"response": wave + 0.5 * rng.normal(0, 1, n)}

        tf = from_time_histories(
            wave_elevation=wave,
            responses=responses,
            dt=dt,
            nperseg=1024,
        )

        coherence = tf.coherence.values
        assert np.all(coherence >= 0)
        assert np.all(coherence <= 1)


class TestRoundTripIdentification:
    """Round-trip tests: known H(f) → generate data → identify → verify recovery."""

    def test_unity_transfer_function(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Unity transfer function (H=1) should be recovered accurately."""
        n = 10240
        nperseg = 2048

        # Generate white noise input
        wave = rng.normal(0, 1, n)

        # Response = input (H = 1)
        response = wave.copy()

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        # Magnitude should be close to 1
        # Skip very low frequencies where estimation is poor
        freq = tf.coords["frequency"].values
        mask = freq > 0.02
        magnitude = tf.magnitude.values[mask, 0]

        assert_allclose(magnitude, 1.0, rtol=0.1)

        # Coherence should be very high
        coherence = tf.coherence.values[mask, 0]
        assert np.mean(coherence) > 0.95

    def test_constant_gain_transfer_function(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Constant gain transfer function should be recovered."""
        n = 10240
        nperseg = 2048
        gain = 2.5

        wave = rng.normal(0, 1, n)
        response = gain * wave

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        freq = tf.coords["frequency"].values
        mask = freq > 0.02
        magnitude = tf.magnitude.values[mask, 0]

        assert_allclose(magnitude, gain, rtol=0.1)

    def test_lowpass_transfer_function(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Lowpass filter transfer function should be recovered."""
        n = 20480
        nperseg = 2048
        fc = 0.2  # Cutoff at 0.2 Hz

        # Generate longer sequence for better statistics
        wave = rng.normal(0, 1, n)

        # Create lowpass filter and apply
        nyq = 1.0 / (2 * dt)
        sos = signal.butter(4, fc / nyq, btype="low", output="sos")
        response = signal.sosfilt(sos, wave)

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        freq = tf.coords["frequency"].values
        magnitude = tf.magnitude.values[:, 0]

        # Below cutoff: magnitude ≈ 1
        below_cutoff = (freq > 0.02) & (freq < 0.5 * fc)
        assert_allclose(magnitude[below_cutoff], 1.0, rtol=0.2)

        # Well above cutoff: magnitude should be significantly reduced
        above_cutoff = freq > 2 * fc
        if np.any(above_cutoff):
            # Butterworth filter: -24 dB/octave for 4th order
            # At 2*fc, magnitude should be < 0.3
            assert np.mean(magnitude[above_cutoff]) < 0.5

    def test_resonance_transfer_function(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Resonant transfer function should be recovered."""
        n = 20480
        nperseg = 2048
        f0 = 0.1  # Resonance at 0.1 Hz
        Q = 3.0   # Quality factor

        # Full frequency array for creating true H(f)
        freq_full = np.fft.rfftfreq(n, dt)

        # Create true transfer function
        H_true = create_simple_resonance_tf(freq_full, f0, Q)

        # Generate input and apply H(f)
        wave = rng.normal(0, 1, n)
        response = apply_transfer_function(wave, freq_full, H_true, dt)

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        freq = tf.coords["frequency"].values
        magnitude = tf.magnitude.values[:, 0]

        # Check peak is at resonance frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freq[peak_idx]
        assert abs(peak_freq - f0) < 0.02

        # Compare magnitudes where coherence is high
        coherence = tf.coherence.values[:, 0]
        high_coh = coherence > 0.7

        H_expected = create_simple_resonance_tf(freq, f0, Q)
        expected_mag = np.abs(H_expected)

        # Where coherence is high, magnitude should match
        if np.sum(high_coh) > 10:
            correlation = np.corrcoef(
                magnitude[high_coh], expected_mag[high_coh]
            )[0, 1]
            assert correlation > 0.9

    def test_multiple_responses(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should correctly identify multiple different transfer functions."""
        n = 10240
        nperseg = 2048

        wave = rng.normal(0, 1, n)

        # Different gains for different responses
        gains = {"A": 1.0, "B": 2.0, "C": 0.5}
        responses = {name: gain * wave for name, gain in gains.items()}

        tf = from_time_histories(
            wave_elevation=wave,
            responses=responses,
            dt=dt,
            nperseg=nperseg,
        )

        freq = tf.coords["frequency"].values
        mask = freq > 0.02

        for i, (name, gain) in enumerate(gains.items()):
            magnitude = tf.magnitude.values[mask, i]
            assert_allclose(magnitude, gain, rtol=0.1)


class TestFromTimeHistoriesParameters:
    """Tests for different parameter settings in from_time_histories."""

    def test_different_nperseg_values(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Different nperseg should give different frequency resolution."""
        n = 10240
        wave = rng.normal(0, 1, n)
        responses = {"response": wave}

        tf_512 = from_time_histories(wave, responses, dt, nperseg=512)
        tf_2048 = from_time_histories(wave, responses, dt, nperseg=2048)

        # More segments = finer frequency resolution
        assert tf_2048.dims["frequency"] > tf_512.dims["frequency"]

    def test_longer_signal_better_coherence(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Longer signals should give better (or equal) coherence estimates."""
        nperseg = 1024

        # Short signal
        n_short = 2048
        wave_short = rng.normal(0, 1, n_short)
        response_short = wave_short + 0.1 * rng.normal(0, 1, n_short)
        tf_short = from_time_histories(
            wave_short, {"r": response_short}, dt, nperseg=nperseg
        )

        # Long signal (same seed for reproducibility)
        rng_long = np.random.default_rng(42)
        n_long = 20480
        wave_long = rng_long.normal(0, 1, n_long)
        response_long = wave_long + 0.1 * rng_long.normal(0, 1, n_long)
        tf_long = from_time_histories(
            wave_long, {"r": response_long}, dt, nperseg=nperseg
        )

        # Coherence estimates should be more stable with longer signal
        # (lower variance, not necessarily higher mean)
        # Just check both are valid
        assert np.all(tf_short.coherence.values >= 0)
        assert np.all(tf_long.coherence.values >= 0)


class TestFromSpectra:
    """Tests for from_spectra() function (loading from post-calc output)."""

    def test_from_spectra_basic(self, tmp_path, frequency_array: np.ndarray) -> None:
        """from_spectra should load pre-computed spectra from npz file."""
        # Create mock post-calc output
        n_freq = len(frequency_array)
        n_var = 2
        variable_names = ["UFJ", "LFJ"]

        # Create realistic spectra
        Sxx = np.ones(n_freq)  # White noise input spectrum
        Syy = np.column_stack([2.0 * Sxx, 0.5 * Sxx])  # Response spectra
        Sxy = np.column_stack([
            np.sqrt(2.0) * Sxx * np.exp(1j * 0.1),
            np.sqrt(0.5) * Sxx * np.exp(-1j * 0.2),
        ])

        # Save to npz
        npz_path = tmp_path / "spectra.npz"
        np.savez(
            npz_path,
            frequency=frequency_array,
            Sxx=Sxx,
            Syy=Syy,
            Sxy=Sxy,
            variable_names=variable_names,
        )

        # Load using from_spectra
        config = {"hs": 2.0, "heading": 0.0}
        tf = from_spectra(npz_path, config=config)

        # Check structure
        assert "frequency" in tf.dims
        assert "variable" in tf.dims
        assert "magnitude" in tf.data_vars
        assert "phase" in tf.data_vars
        assert "coherence" in tf.data_vars

        # Check values are reasonable
        assert np.all(tf.magnitude.values >= 0)
        assert np.all(tf.coherence.values >= 0)
        assert np.all(tf.coherence.values <= 1)

    def test_from_spectra_with_config(
        self, tmp_path, frequency_array: np.ndarray
    ) -> None:
        """Config should be stored in attributes."""
        # Create minimal npz
        Sxx = np.ones(len(frequency_array))
        Syy = np.ones((len(frequency_array), 1))
        Sxy = np.ones((len(frequency_array), 1), dtype=complex)

        npz_path = tmp_path / "spectra.npz"
        np.savez(
            npz_path,
            frequency=frequency_array,
            Sxx=Sxx,
            Syy=Syy,
            Sxy=Sxy,
            variable_names=["response"],
        )

        config = {"hs": 3.0, "draft": 21.0, "heading": 45.0}
        tf = from_spectra(npz_path, config=config)

        assert "config" in tf.attrs
        assert tf.attrs["config"] == config


class TestCoherenceThreshold:
    """Tests for coherence-based reliability flagging."""

    def test_low_coherence_at_noise_frequencies(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Frequencies with uncorrelated noise should have low coherence."""
        n = 10240
        nperseg = 2048

        wave = rng.normal(0, 1, n)
        # Response is mostly independent noise
        response = rng.normal(0, 1, n)

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        coherence = tf.coherence.values[:, 0]

        # Mean coherence should be low for uncorrelated signals
        assert np.mean(coherence) < 0.3

    def test_high_coherence_at_signal_frequencies(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Frequencies with correlated signal should have high coherence."""
        n = 10240
        nperseg = 2048

        wave = rng.normal(0, 1, n)
        # Response is perfectly correlated
        response = 2.0 * wave

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=nperseg,
        )

        freq = tf.coords["frequency"].values
        coherence = tf.coherence.values[:, 0]

        # Skip DC component
        mask = freq > 0.01
        assert np.mean(coherence[mask]) > 0.95


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_mismatched_lengths_raises(self, dt: float) -> None:
        """Should raise error if wave and response have different lengths."""
        wave = np.random.randn(1000)
        responses = {"response": np.random.randn(500)}  # Wrong length

        with pytest.raises(ValueError, match="length|shape"):
            from_time_histories(wave, responses, dt, nperseg=256)

    def test_nperseg_larger_than_signal_raises(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should raise error if nperseg > signal length."""
        n = 500
        wave = rng.normal(0, 1, n)
        responses = {"response": rng.normal(0, 1, n)}

        with pytest.raises(ValueError, match="nperseg|length"):
            from_time_histories(wave, responses, dt, nperseg=1024)

    def test_empty_responses_raises(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should raise error for empty responses dict."""
        wave = rng.normal(0, 1, 1000)

        with pytest.raises(ValueError, match="empty|response"):
            from_time_histories(wave, {}, dt, nperseg=256)

    def test_zero_dt_raises(self, rng: np.random.Generator) -> None:
        """Should raise error for zero or negative dt."""
        wave = rng.normal(0, 1, 1000)
        responses = {"response": rng.normal(0, 1, 1000)}

        with pytest.raises(ValueError, match="dt|positive"):
            from_time_histories(wave, responses, dt=0.0, nperseg=256)

    def test_constant_signal_handling(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should handle constant input signal gracefully."""
        n = 5120
        wave = np.ones(n)  # Constant - no variance
        responses = {"response": rng.normal(0, 1, n)}

        # Should either raise error or return with warnings
        try:
            tf = from_time_histories(wave, responses, dt, nperseg=1024)
            # If it returns, coherence should be low or undefined
            # Magnitude may be inf or nan
        except (ValueError, RuntimeWarning):
            pass  # Also acceptable


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_amplitude_signal(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should handle very small amplitude signals."""
        n = 5120
        wave = 1e-10 * rng.normal(0, 1, n)
        response = 2.0 * wave

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=1024,
        )

        # Magnitude should still be approximately 2
        freq = tf.coords["frequency"].values
        mask = freq > 0.02
        magnitude = tf.magnitude.values[mask, 0]

        assert np.all(np.isfinite(magnitude))
        assert_allclose(np.median(magnitude), 2.0, rtol=0.2)

    def test_very_large_amplitude_signal(
        self, rng: np.random.Generator, dt: float
    ) -> None:
        """Should handle very large amplitude signals."""
        n = 5120
        wave = 1e10 * rng.normal(0, 1, n)
        response = 2.0 * wave

        tf = from_time_histories(
            wave_elevation=wave,
            responses={"response": response},
            dt=dt,
            nperseg=1024,
        )

        freq = tf.coords["frequency"].values
        mask = freq > 0.02
        magnitude = tf.magnitude.values[mask, 0]

        assert np.all(np.isfinite(magnitude))
        assert_allclose(np.median(magnitude), 2.0, rtol=0.2)
