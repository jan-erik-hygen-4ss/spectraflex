#!/usr/bin/env python
"""Basic transfer function identification example.

This example demonstrates how to identify a transfer function from
synthetic time history data. This is useful for:
1. Understanding the identification process
2. Testing without OrcaFlex
3. Validating against known transfer functions

The example creates a known transfer function (a simple resonance),
generates input/output signals, and verifies the identification
recovers the original H(f).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spectraflex import identify


def create_resonance_transfer_function(
    f: np.ndarray,
    f0: float = 0.1,
    q: float = 5.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Create a simple resonance transfer function.

    H(f) = gain / (1 + j*Q*(f/f0 - f0/f))

    Parameters
    ----------
    f : np.ndarray
        Frequency array [Hz].
    f0 : float
        Resonance frequency [Hz].
    q : float
        Quality factor (higher = sharper peak).
    gain : float
        DC gain.

    Returns
    -------
    np.ndarray
        Complex transfer function H(f).
    """
    return gain / (1.0 + 1j * q * (f / f0 - f0 / f))


def filter_signal_frequency_domain(
    signal: np.ndarray,
    h: np.ndarray,
    f: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Filter a signal through H(f) using frequency-domain multiplication.

    Parameters
    ----------
    signal : np.ndarray
        Input time series.
    h : np.ndarray
        Complex transfer function.
    f : np.ndarray
        Frequencies corresponding to h.
    dt : float
        Sample interval.

    Returns
    -------
    np.ndarray
        Filtered output time series.
    """
    n = len(signal)
    fft_freq = np.fft.rfftfreq(n, dt)

    # Interpolate H to FFT frequencies
    h_interp = np.interp(fft_freq, f, h)

    # Apply transfer function in frequency domain
    signal_fft = np.fft.rfft(signal)
    output_fft = signal_fft * h_interp
    output = np.fft.irfft(output_fft, n=n)

    return output


def main() -> None:
    """Run the basic identification example."""
    print("=" * 60)
    print("Spectraflex - Basic Transfer Function Identification")
    print("=" * 60)

    # Parameters
    dt = 0.1  # Sample interval [s]
    duration = 1024.0  # Duration [s]
    n_samples = int(duration / dt)
    f0 = 0.1  # Resonance frequency [Hz]
    q = 5.0  # Quality factor

    print("\nParameters:")
    print(f"  Sample interval: {dt} s")
    print(f"  Duration: {duration} s")
    print(f"  Samples: {n_samples}")
    print(f"  Resonance frequency: {f0} Hz")
    print(f"  Quality factor: {q}")

    # Create frequency array for the known transfer function
    f = np.linspace(0.01, 0.5, 256)
    h_true = create_resonance_transfer_function(f, f0=f0, q=q)

    print("\nCreated known transfer function:")
    print(
        f"  Peak magnitude: {np.abs(h_true).max():.2f} at f={f[np.argmax(np.abs(h_true))]:.3f} Hz"
    )

    # Generate white noise input
    rng = np.random.default_rng(42)
    wave = rng.normal(0, 1, n_samples)
    print("\nGenerated white noise input:")
    print(f"  Std: {wave.std():.3f}")

    # Filter through transfer function to create response
    response = filter_signal_frequency_domain(wave, h_true, f, dt)
    print("\nFiltered to create response:")
    print(f"  Response std: {response.std():.3f}")
    print(f"  Amplification: {response.std() / wave.std():.2f}x")

    # Identify transfer function from time histories
    print("\nIdentifying transfer function...")
    tf = identify.from_time_histories(
        wave_elevation=wave,
        responses={"response": response},
        dt=dt,
        nperseg=1024,
    )

    # Extract identified values
    tf_freq = tf.coords["frequency"].values
    tf_mag = tf["magnitude"].values[:, 0]
    tf_phase = tf["phase"].values[:, 0]
    tf_coh = tf["coherence"].values[:, 0]

    print(f"  Identified {len(tf_freq)} frequency points")
    print(f"  Frequency range: {tf_freq.min():.3f} - {tf_freq.max():.3f} Hz")

    # Interpolate true H to identified frequencies for comparison
    h_true_interp = np.interp(tf_freq, f, h_true)
    mag_true = np.abs(h_true_interp)
    phase_true = np.angle(h_true_interp)

    # Compute errors (only where coherence is good)
    good_coh = tf_coh > 0.8
    if good_coh.sum() > 0:
        mag_error = np.abs(tf_mag[good_coh] - mag_true[good_coh]) / mag_true[good_coh]
        phase_error = np.abs(tf_phase[good_coh] - phase_true[good_coh])

        print(f"\nComparison (coherence > 0.8, {good_coh.sum()} points):")
        print(f"  Magnitude MAPE: {100 * mag_error.mean():.1f}%")
        print(f"  Phase MAE: {np.degrees(phase_error.mean()):.1f} deg")
        print(f"  Mean coherence: {tf_coh[good_coh].mean():.3f}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Magnitude
    ax = axes[0]
    ax.semilogy(tf_freq, tf_mag, "b-", label="Identified", linewidth=1.5)
    ax.semilogy(tf_freq, mag_true, "r--", label="True", linewidth=1.5)
    ax.set_ylabel("|H(f)|")
    ax.legend()
    ax.set_title("Transfer Function Identification")
    ax.grid(True, alpha=0.3)

    # Phase
    ax = axes[1]
    ax.plot(tf_freq, np.degrees(tf_phase), "b-", label="Identified", linewidth=1.5)
    ax.plot(tf_freq, np.degrees(phase_true), "r--", label="True", linewidth=1.5)
    ax.set_ylabel("Phase [deg]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Coherence
    ax = axes[2]
    ax.plot(tf_freq, tf_coh, "g-", linewidth=1.5)
    ax.axhline(0.8, color="k", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("basic_identification_result.png", dpi=150)
    print("\nSaved plot to: basic_identification_result.png")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
