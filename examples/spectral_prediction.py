#!/usr/bin/env python
"""Spectral response prediction example.

This example demonstrates how to use a transfer function to predict
response statistics for different sea states. This is the core use case
for spectraflex: run white noise once, then predict responses for many
sea states in milliseconds.

The example:
1. Creates a synthetic transfer function
2. Defines several sea states (JONSWAP spectra)
3. Computes response spectra and statistics for each
4. Compares results across sea states
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spectraflex import predict, spectrum, transfer_function


def create_sample_transfer_function() -> tuple:
    """Create a sample transfer function for demonstration.

    Returns a transfer function representing a simple single-degree-of-freedom
    system with resonance at 0.08 Hz (12.5s period) - typical for a floating
    structure's heave response.

    Returns
    -------
    tuple
        (xr.Dataset, frequency array, complex H(f))
    """
    # Frequency array
    f = spectrum.frequency_array(f_min=0.02, f_max=0.3, n_freq=256)

    # Create resonance transfer function
    f0 = 0.08  # Natural frequency [Hz]
    zeta = 0.1  # Damping ratio
    gain = 2.0  # Static gain

    # Standard SDOF transfer function
    omega = 2 * np.pi * f
    omega0 = 2 * np.pi * f0

    h = gain / (1 - (omega / omega0) ** 2 + 2j * zeta * (omega / omega0))

    # Create TransferFunction dataset
    tf = transfer_function.from_complex(
        frequency=f,
        h_complex=h,
        coherence=np.ones_like(f) * 0.95,
        variable_names=["Heave"],
        config={"type": "synthetic", "f0": f0, "zeta": zeta},
    )

    return tf, f, h


def main() -> None:
    """Run the spectral prediction example."""
    print("=" * 60)
    print("Spectraflex - Spectral Response Prediction")
    print("=" * 60)

    # Create transfer function
    tf, f, h_complex = create_sample_transfer_function()

    print("\nTransfer Function:")
    print(f"  Resonance frequency: {tf.attrs['config']['f0']} Hz")
    print(f"  Resonance period: {1 / tf.attrs['config']['f0']:.1f} s")
    print(f"  Damping ratio: {tf.attrs['config']['zeta']}")
    print(f"  Peak |H|: {np.abs(h_complex).max():.1f}")

    # Define sea states to analyze
    sea_states = [
        {"hs": 1.0, "tp": 6.0, "gamma": 3.3, "label": "Mild"},
        {"hs": 2.5, "tp": 8.0, "gamma": 3.3, "label": "Moderate"},
        {"hs": 4.0, "tp": 10.0, "gamma": 2.5, "label": "Rough"},
        {"hs": 6.0, "tp": 12.0, "gamma": 2.0, "label": "Severe"},
        {"hs": 8.0, "tp": 14.0, "gamma": 1.5, "label": "Extreme"},
    ]

    # Duration for MPM calculations (3 hours)
    duration = 3 * 3600.0

    print(f"\nAnalyzing {len(sea_states)} sea states:")
    print("-" * 60)

    results = []
    wave_spectra = []
    response_spectra = []

    for sea in sea_states:
        # Create wave spectrum
        wave = spectrum.jonswap(hs=sea["hs"], tp=sea["tp"], f=f, gamma=sea["gamma"])
        wave_spectra.append(wave)

        # Compute response spectrum
        resp = predict.response_spectrum(tf, wave)
        response_spectra.append(resp)

        # Compute statistics
        stats = predict.statistics(tf, wave, duration=duration)
        resp_stats = stats["Heave"]

        results.append(
            {
                "label": sea["label"],
                "wave_hs": sea["hs"],
                "wave_tp": sea["tp"],
                "resp_hs": resp_stats["hs"],
                "resp_tp": resp_stats["tp"],
                "resp_mpm": resp_stats["mpm"],
                "amplification": resp_stats["hs"] / sea["hs"],
            }
        )

        print(
            f"  {sea['label']:10s}: Hs={sea['hs']:.1f}m, Tp={sea['tp']:.1f}s -> "
            f"Response Hs={resp_stats['hs']:.2f}, MPM={resp_stats['mpm']:.2f}"
        )

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(
        f"{'Sea State':<12} {'Wave Hs':<10} {'Wave Tp':<10} {'Resp Hs':<10} {'MPM':<10} {'Amp':<8}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['label']:<12} {r['wave_hs']:<10.1f} {r['wave_tp']:<10.1f} "
            f"{r['resp_hs']:<10.2f} {r['resp_mpm']:<10.2f} {r['amplification']:<8.2f}"
        )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Transfer function
    ax = axes[0, 0]
    ax.semilogy(f, np.abs(h_complex), "b-", linewidth=2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|H(f)|")
    ax.set_title("Transfer Function")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.08, color="r", linestyle="--", alpha=0.5, label="f₀ = 0.08 Hz")
    ax.legend()

    # Wave spectra
    ax = axes[0, 1]
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(sea_states)))
    for i, (sea, wave) in enumerate(zip(sea_states, wave_spectra)):
        ax.plot(f, wave.values, color=colors[i], linewidth=1.5, label=sea["label"])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("S(f) [m²/Hz]")
    ax.set_title("Wave Spectra (JONSWAP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Response spectra
    ax = axes[1, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(sea_states)))
    for i, (sea, resp) in enumerate(zip(sea_states, response_spectra)):
        syy = resp["Syy"].values[:, 0]
        ax.plot(f, syy, color=colors[i], linewidth=1.5, label=sea["label"])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("S_response(f)")
    ax.set_title("Response Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amplification vs sea state
    ax = axes[1, 1]
    labels = [r["label"] for r in results]
    wave_hs = [r["wave_hs"] for r in results]
    resp_hs = [r["resp_hs"] for r in results]
    resp_mpm = [r["resp_mpm"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, wave_hs, width, label="Wave Hs", color="steelblue")
    ax.bar(x + width / 2, resp_hs, width, label="Response Hs", color="coral")
    ax.plot(x, resp_mpm, "ko-", label="Response MPM", markersize=8)

    ax.set_xlabel("Sea State")
    ax.set_ylabel("Height [m]")
    ax.set_title("Wave vs Response Statistics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("spectral_prediction_result.png", dpi=150)
    print("\nSaved plot to: spectral_prediction_result.png")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
