#!/usr/bin/env python
"""Transfer function library workflow example.

This example demonstrates the complete workflow for building and using
a TransferFunctionLibrary:

1. Create transfer functions for multiple configurations
2. Add them to a library
3. Save/load the library
4. Look up transfer functions for prediction
5. Compare predictions across configurations

This represents the typical workflow for a project like a global riser
analysis where transfer functions are identified for multiple operating
configurations (draft, heading, current).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectraflex import (
    TransferFunctionLibrary,
    predict,
    spectrum,
    transfer_function,
)


def create_config_dependent_tf(
    f: np.ndarray,
    heading: float,
    current_speed: float,
) -> np.ndarray:
    """Create a transfer function that varies with configuration.

    This simulates how a real system's response varies with heading
    and current. For demonstration, we modify the resonance frequency
    and damping based on the configuration.

    Parameters
    ----------
    f : np.ndarray
        Frequency array [Hz].
    heading : float
        Wave heading [deg].
    current_speed : float
        Current speed [m/s].

    Returns
    -------
    np.ndarray
        Complex transfer function.
    """
    # Base parameters
    f0_base = 0.08  # Base natural frequency
    zeta_base = 0.10  # Base damping
    gain_base = 2.0  # Base gain

    # Heading effect: beam seas (90 deg) have higher response
    heading_factor = 1.0 + 0.3 * np.sin(np.radians(heading)) ** 2
    gain = gain_base * heading_factor

    # Current effect: higher current increases damping
    zeta = zeta_base + 0.05 * current_speed

    # Slight frequency shift with current
    f0 = f0_base * (1.0 - 0.02 * current_speed)

    # SDOF transfer function
    omega = 2 * np.pi * f
    omega0 = 2 * np.pi * f0

    h = gain / (1 - (omega / omega0) ** 2 + 2j * zeta * (omega / omega0))

    return h


def main() -> None:
    """Run the library workflow example."""
    print("=" * 60)
    print("Spectraflex - Transfer Function Library Workflow")
    print("=" * 60)

    # Define configuration space
    headings = [0.0, 45.0, 90.0, 135.0, 180.0]
    current_speeds = [0.0, 0.5, 1.0]

    # Frequency array
    f = spectrum.frequency_array(f_min=0.02, f_max=0.3, n_freq=256)

    print("\nConfiguration space:")
    print(f"  Headings: {headings} deg")
    print(f"  Current speeds: {current_speeds} m/s")
    print(f"  Total configurations: {len(headings) * len(current_speeds)}")

    # Create library
    print("\nBuilding library...")
    lib = TransferFunctionLibrary()

    for heading in headings:
        for current in current_speeds:
            # Create transfer function for this config
            h = create_config_dependent_tf(f, heading, current)

            tf = transfer_function.from_complex(
                frequency=f,
                h_complex=h,
                coherence=np.ones_like(f) * 0.95,
                variable_names=["Response"],
                config={"heading": heading, "current_speed": current},
            )

            lib.add(tf)

    print(f"  Added {len(lib)} configurations")
    print(f"  Config keys: {lib.config_keys}")

    # Save and reload library
    print("\nTesting save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path = Path(tmpdir) / "example_library.nc"
        lib.save(lib_path)
        print(f"  Saved to: {lib_path}")
        print(f"  File size: {lib_path.stat().st_size / 1024:.1f} KB")

        lib_loaded = TransferFunctionLibrary.load(lib_path)
        print(f"  Loaded {len(lib_loaded)} configurations")

    # Demonstrate lookup methods
    print("\nLookup demonstrations:")

    # Exact match
    _ = lib.select(heading=90.0, current_speed=0.5)
    print("  Exact match (heading=90, current=0.5): found")

    # Nearest neighbor (for values not in library)
    tf_nearest = lib.lookup(heading=75.0, current_speed=0.7, method="nearest")
    print(f"  Nearest to (heading=75, current=0.7): {tf_nearest.attrs['config']}")

    # Interpolated
    _ = lib.lookup(heading=75.0, current_speed=0.7, method="linear")
    print("  Interpolated (heading=75, current=0.7): created")

    # Filter library
    lib_filtered = lib.filter(current_speed=0.5)
    print(f"  Filtered (current=0.5): {len(lib_filtered)} configurations")

    # Prediction comparison across configurations
    print("\n" + "=" * 60)
    print("Prediction Comparison")
    print("=" * 60)

    # Define a sea state
    hs = 4.0
    tp = 10.0
    wave = spectrum.jonswap(hs=hs, tp=tp, f=f)
    duration = 3 * 3600.0

    print(f"\nSea state: Hs={hs}m, Tp={tp}s")
    print(f"Duration: {duration / 3600:.0f} hours")
    print("\nResults by heading (current=0.5 m/s):")
    print("-" * 50)

    heading_results = []
    for heading in headings:
        tf = lib.select(heading=heading, current_speed=0.5)
        stats = predict.statistics(tf, wave, duration=duration)
        resp_stats = stats["Response"]

        heading_results.append(
            {
                "heading": heading,
                "hs": resp_stats["hs"],
                "mpm": resp_stats["mpm"],
            }
        )

        print(
            f"  Heading {heading:5.0f}°: Hs={resp_stats['hs']:.2f}, MPM={resp_stats['mpm']:.2f}"
        )

    print("\nResults by current speed (heading=90°):")
    print("-" * 50)

    current_results = []
    for current in current_speeds:
        tf = lib.select(heading=90.0, current_speed=current)
        stats = predict.statistics(tf, wave, duration=duration)
        resp_stats = stats["Response"]

        current_results.append(
            {
                "current": current,
                "hs": resp_stats["hs"],
                "mpm": resp_stats["mpm"],
            }
        )

        print(
            f"  Current {current:.1f} m/s: Hs={resp_stats['hs']:.2f}, MPM={resp_stats['mpm']:.2f}"
        )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Transfer functions by heading
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(headings)))
    for i, heading in enumerate(headings):
        tf = lib.select(heading=heading, current_speed=0.5)
        mag = tf["magnitude"].values[:, 0]
        ax.semilogy(f, mag, color=colors[i], linewidth=1.5, label=f"{heading}°")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|H(f)|")
    ax.set_title("Transfer Functions by Heading (current=0.5 m/s)")
    ax.legend(title="Heading")
    ax.grid(True, alpha=0.3)

    # Transfer functions by current
    ax = axes[0, 1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(current_speeds)))
    for i, current in enumerate(current_speeds):
        tf = lib.select(heading=90.0, current_speed=current)
        mag = tf["magnitude"].values[:, 0]
        ax.semilogy(f, mag, color=colors[i], linewidth=2, label=f"{current} m/s")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|H(f)|")
    ax.set_title("Transfer Functions by Current (heading=90°)")
    ax.legend(title="Current")
    ax.grid(True, alpha=0.3)

    # Response vs heading
    ax = axes[1, 0]
    headings_arr = [r["heading"] for r in heading_results]
    hs_arr = [r["hs"] for r in heading_results]
    mpm_arr = [r["mpm"] for r in heading_results]

    ax.plot(headings_arr, hs_arr, "bo-", markersize=8, linewidth=2, label="Response Hs")
    ax.plot(
        headings_arr, mpm_arr, "rs-", markersize=8, linewidth=2, label="Response MPM"
    )
    ax.axhline(hs, color="b", linestyle="--", alpha=0.5, label=f"Wave Hs={hs}m")
    ax.set_xlabel("Heading [deg]")
    ax.set_ylabel("Response [m]")
    ax.set_title("Response vs Heading (current=0.5 m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(headings)

    # Response vs current
    ax = axes[1, 1]
    currents_arr = [r["current"] for r in current_results]
    hs_arr = [r["hs"] for r in current_results]
    mpm_arr = [r["mpm"] for r in current_results]

    ax.plot(currents_arr, hs_arr, "bo-", markersize=8, linewidth=2, label="Response Hs")
    ax.plot(
        currents_arr, mpm_arr, "rs-", markersize=8, linewidth=2, label="Response MPM"
    )
    ax.axhline(hs, color="b", linestyle="--", alpha=0.5, label=f"Wave Hs={hs}m")
    ax.set_xlabel("Current Speed [m/s]")
    ax.set_ylabel("Response [m]")
    ax.set_title("Response vs Current (heading=90°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("library_workflow_result.png", dpi=150)
    print("\nSaved plot to: library_workflow_result.png")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
