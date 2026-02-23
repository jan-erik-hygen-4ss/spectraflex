#!/usr/bin/env python
"""Compare spectraflex spectral fatigue with OrcaFlex FatigueAnalysis.

This example demonstrates how to:
1. Identify a stress transfer function from a white noise .sim file
2. Run spectraflex's spectral fatigue on that transfer function + wave spectra
3. Run OrcaFlex's native FatigueAnalysis on a model .sim file
4. Compare the results side-by-side

The two approaches use different inputs:
- **spectraflex**: needs a completed white noise .sim (time-domain results)
  to identify the transfer function H(f), then combines it with wave spectra.
- **OrcaFlex FatigueAnalysis**: needs a model .sim file (frequency-domain
  RAOs), which does not need to have been run in time domain.

These can be the same .sim file if it's a completed white noise simulation,
or different files if your white noise sim and model file are separate.

The example has three modes:
- **Synthetic mode** (no OrcFxAPI required): demonstrates the spectraflex
  fatigue workflow using a synthetic transfer function.
- **spectraflex-only** (requires OrcFxAPI + white noise .sim): identifies TF
  from the .sim and runs spectraflex fatigue.
- **Full comparison** (requires OrcFxAPI + both .sim files): runs both
  analyses and prints a comparison table.

Usage
-----
Synthetic mode (always works):
    python examples/fatigue_comparison.py

spectraflex-only (white noise .sim, no OrcaFlex comparison):
    python examples/fatigue_comparison.py \\
        --wn-sim path/to/white_noise.sim \\
        --line Riser \\
        --variable "Direct Tensile Stress" \\
        --arclength 50.0

Full comparison (white noise .sim + model .sim for OrcaFlex):
    python examples/fatigue_comparison.py \\
        --wn-sim path/to/white_noise.sim \\
        --sim path/to/model.sim \\
        --line Riser \\
        --variable "Direct Tensile Stress" \\
        --arclength 50.0

If --sim and --wn-sim point to the same file (completed white noise sim):
    python examples/fatigue_comparison.py \\
        --wn-sim path/to/white_noise.sim \\
        --sim path/to/white_noise.sim \\
        --line Riser
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from spectraflex import fatigue, identify, spectrum, transfer_function
from spectraflex.fatigue import SNCurve
from spectraflex.orcaflex.fatigue import (
    OrcaFlexFatigueConfig,
    SpectralLoadCase,
    compare_results,
    run_spectral_fatigue,
    sn_curve_to_orcaflex,
)


# =============================================================================
# Shared setup
# =============================================================================

# Sea states to analyze: (Hs [m], Tz [s], Tp [s], exposure [hours])
SEA_STATES = [
    {"hs": 1.5, "tz": 5.5, "tp": 7.0, "gamma": 3.3, "hours": 3000},
    {"hs": 2.5, "tz": 7.0, "tp": 9.0, "gamma": 3.3, "hours": 2000},
    {"hs": 4.0, "tz": 8.5, "tp": 11.0, "gamma": 2.5, "hours": 1000},
    {"hs": 6.0, "tz": 10.0, "tp": 13.0, "gamma": 2.0, "hours": 200},
]


def create_synthetic_transfer_function() -> tuple[xr.Dataset, np.ndarray]:
    """Create a synthetic stress transfer function [MPa/m].

    Simulates a riser with resonance at ~0.08 Hz (12.5s period) and a
    base stress response of 15 MPa per meter wave height.

    Returns
    -------
    tuple
        (xr.Dataset, frequency array)
    """
    f = spectrum.frequency_array(f_min=0.02, f_max=0.3, n_freq=256)

    # SDOF resonance model
    f0 = 0.08  # Natural frequency [Hz]
    zeta = 0.08  # Damping ratio
    gain = 15.0  # Stress per wave height [MPa/m]

    omega = 2 * np.pi * f
    omega0 = 2 * np.pi * f0
    h = gain / (1 - (omega / omega0) ** 2 + 2j * zeta * (omega / omega0))

    tf = transfer_function.from_complex(
        frequency=f,
        h_complex=h,
        coherence=np.ones_like(f) * 0.95,
        variable_names=["stress"],
        config={"type": "synthetic_stress", "f0": f0, "zeta": zeta},
    )

    return tf, f


def extract_transfer_function_from_sim(
    sim_path: str | Path,
    line_name: str,
    variable: str,
    arclength: float,
) -> tuple[xr.Dataset, np.ndarray]:
    """Extract stress transfer function from a completed white noise .sim file.

    Uses cross-spectral identification to estimate H(f) from the wave
    elevation and stress time histories in the simulation.

    Parameters
    ----------
    sim_path : str or Path
        Path to completed .sim file (must have time-domain results).
    line_name : str
        OrcaFlex line object name (e.g., "Riser").
    variable : str
        Stress variable to extract (e.g., "Direct Tensile Stress").
    arclength : float
        Arc length position on the line [m].

    Returns
    -------
    tuple
        (xr.Dataset, frequency array)
    """
    results = [
        {
            "object": line_name,
            "variable": variable,
            "arclength": arclength,
            "label": "stress",
        },
    ]

    tf = identify.from_sim(sim_path, results=results)
    f = tf.coords["frequency"].values

    print(f"  Identified TF from: {sim_path}")
    print(f"  Variable: {line_name} / {variable} at arc length {arclength} m")
    print(f"  Frequency range: {f[0]:.4f} - {f[-1]:.4f} Hz ({len(f)} points)")

    return tf, f


# =============================================================================
# Spectraflex fatigue
# =============================================================================


def run_spectraflex_fatigue(
    tf: xr.Dataset,
    f: np.ndarray,
    sn_curve: SNCurve,
    method: str = "dirlik",
) -> dict:
    """Run spectraflex spectral fatigue for all sea states.

    Parameters
    ----------
    tf : xr.Dataset
        Stress transfer function [MPa/m] (or stress unit / wave height unit).
    f : np.ndarray
        Frequency array [Hz] matching the transfer function.
    sn_curve : SNCurve
        S-N curve for damage calculation.
    method : str
        "dirlik" or "narrow_band".

    Returns
    -------
    dict
        Fatigue results including per-sea-state and total damage.
    """
    total_damage = 0.0
    per_case = []

    for sea in SEA_STATES:
        wave_spec = spectrum.jonswap(hs=sea["hs"], tp=sea["tp"], f=f, gamma=sea["gamma"])
        exposure_seconds = sea["hours"] * 3600.0

        result = fatigue.damage_from_transfer_function(
            tf=tf,
            wave_spectrum=wave_spec,
            sn_curve=sn_curve,
            exposure_time=exposure_seconds,
            method=method,
        )

        total_damage += result["damage"]
        per_case.append(
            {
                "hs": sea["hs"],
                "tp": sea["tp"],
                "hours": sea["hours"],
                "damage": result["damage"],
                "stress_rms": result["stress_rms"],
                "bandwidth": result["bandwidth"],
            }
        )

    total_exposure_hours = sum(sea["hours"] for sea in SEA_STATES)
    total_exposure_years = total_exposure_hours / 8766.0  # hours per year

    return {
        "per_case": per_case,
        "total_damage": total_damage,
        "fatigue_life_years": total_exposure_years / total_damage if total_damage > 0 else np.inf,
        "method": method,
    }


# =============================================================================
# OrcaFlex fatigue (requires OrcFxAPI + .sim files)
# =============================================================================


def run_orcaflex_fatigue(
    sim_file: str | Path,
    line_name: str,
    sn_curve: SNCurve,
    arclengths: list[tuple[float, float]],
) -> dict:
    """Run OrcaFlex native spectral fatigue for all sea states.

    Parameters
    ----------
    sim_file : str or Path
        Path to model .sim file (used for frequency-domain RAOs).
    line_name : str
        OrcaFlex line object name.
    sn_curve : SNCurve
        S-N curve for damage calculation.
    arclengths : list of tuple
        Arc length ranges to analyze.

    Returns
    -------
    dict
        OrcaFlex fatigue results.
    """
    load_cases = [
        SpectralLoadCase(
            sim_file=sim_file,
            line_name=line_name,
            exposure_time=sea["hours"] * 3600.0,
            hs=sea["hs"],
            tz=sea["tz"],
            tp=sea["tp"],
            gamma=sea["gamma"],
        )
        for sea in SEA_STATES
    ]

    config = OrcaFlexFatigueConfig(
        load_cases=load_cases,
        sn_curve=sn_curve,
        arclengths=arclengths,
    )

    ofx_result = run_spectral_fatigue(config)

    return {
        "max_damage": ofx_result.max_damage,
        "max_arclength": ofx_result.max_damage_arclength,
        "max_theta": ofx_result.max_damage_theta,
        "result": ofx_result,
    }


# =============================================================================
# Display
# =============================================================================


def print_spectraflex_results(sfx: dict) -> None:
    """Print spectraflex fatigue results."""
    print(f"\n{'='*70}")
    print(f"Spectraflex Spectral Fatigue ({sfx['method'].title()})")
    print(f"{'='*70}")
    print(
        f"{'Hs [m]':<10} {'Tp [s]':<10} {'Hours':<10} "
        f"{'Stress RMS':<12} {'BW':<8} {'Damage':<12}"
    )
    print("-" * 70)

    for c in sfx["per_case"]:
        print(
            f"{c['hs']:<10.1f} {c['tp']:<10.1f} {c['hours']:<10.0f} "
            f"{c['stress_rms']:<12.2f} {c['bandwidth']:<8.3f} {c['damage']:<12.6f}"
        )

    print("-" * 70)
    print(f"{'Total damage:':<42} {sfx['total_damage']:<12.6f}")
    print(f"{'Fatigue life:':<42} {sfx['fatigue_life_years']:.1f} years")


def print_sn_curve_mapping(sn_curve: SNCurve) -> None:
    """Print the S-N curve to OrcaFlex property mapping."""
    props = sn_curve_to_orcaflex(sn_curve)

    print(f"\n{'='*70}")
    print(f"S-N Curve Mapping: {sn_curve.name} (MPa -> kPa)")
    print(f"{'='*70}")
    print(f"{'spectraflex (MPa)':<25} {'OrcaFlex Property (kPa)':<25} {'Value':<15}")
    print("-" * 70)
    print(f"{'m1=' + str(sn_curve.m1):<25} {'SNCurvem1':<25} {props['SNCurvem1']}")
    print(
        f"{'log_a1=' + str(sn_curve.log_a1):<25} {'SNCurveLogA1':<25} "
        f"{props['SNCurveLogA1']:.3f}"
    )
    print(f"{'n_transition':<25} {'SNCurveRegionBoundary':<25} {props['SNCurveRegionBoundary']:.0f}")
    print(f"{'m2=' + str(sn_curve.m2):<25} {'SNCurvem2':<25} {props['SNCurvem2']}")
    print(f"{'log_a2 (read-only)':<25} {'SNCurveLogA2':<25} {'(auto-calculated)'}")
    print()
    print("Note: log_a converted from MPa to kPa: log_a_kPa = log_a_MPa + m * 3")
    print("      OrcaFlex properties must be set in order shown above.")


def print_comparison(sfx_damage: float, ofx_damage: float) -> None:
    """Print comparison of spectraflex vs OrcaFlex results."""
    comp = compare_results(ofx_damage=ofx_damage, sfx_damage=sfx_damage)

    print(f"\n{'='*70}")
    print("Comparison: spectraflex vs OrcaFlex")
    print(f"{'='*70}")
    print(f"{'spectraflex damage:':<35} {comp.sfx_damage:.6f}")
    print(f"{'OrcaFlex damage:':<35} {comp.ofx_damage:.6f}")
    print(f"{'Ratio (OrcaFlex / spectraflex):':<35} {comp.ratio:.4f}")
    print(f"{'Absolute difference:':<35} {comp.abs_diff:+.6f}")
    print(f"{'Relative difference:':<35} {comp.rel_diff:+.1%}")

    if abs(comp.rel_diff) < 0.05:
        print("\nResult: Excellent agreement (< 5% difference)")
    elif abs(comp.rel_diff) < 0.15:
        print("\nResult: Good agreement (< 15% difference)")
    else:
        print("\nResult: Significant difference — investigate S-N curve or spectrum assumptions")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the fatigue comparison example."""
    parser = argparse.ArgumentParser(
        description="Compare spectraflex vs OrcaFlex spectral fatigue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  Synthetic mode (no OrcFxAPI):
    python examples/fatigue_comparison.py

  spectraflex-only (identify TF from white noise sim):
    python examples/fatigue_comparison.py --wn-sim wn.sim --line Riser

  Full comparison (white noise sim for TF + model sim for OrcaFlex):
    python examples/fatigue_comparison.py --wn-sim wn.sim --sim model.sim --line Riser

  Same file for both (if your white noise .sim can serve as the model):
    python examples/fatigue_comparison.py --wn-sim wn.sim --sim wn.sim --line Riser
""",
    )
    parser.add_argument(
        "--wn-sim", type=str, default=None,
        help="Path to completed white noise .sim file (for TF identification)",
    )
    parser.add_argument(
        "--sim", type=str, default=None,
        help="Path to model .sim file (for OrcaFlex FatigueAnalysis comparison)",
    )
    parser.add_argument(
        "--line", type=str, default="Riser",
        help="OrcaFlex line object name (default: Riser)",
    )
    parser.add_argument(
        "--variable", type=str, default="Direct Tensile Stress",
        help="Stress variable to extract for TF identification (default: Direct Tensile Stress)",
    )
    parser.add_argument(
        "--arclength", type=float, default=50.0,
        help="Arc length for TF extraction [m] (default: 50.0)",
    )
    parser.add_argument(
        "--arc-from", type=float, default=None,
        help="Arc length range start for OrcaFlex [m] (default: arclength - 1)",
    )
    parser.add_argument(
        "--arc-to", type=float, default=None,
        help="Arc length range end for OrcaFlex [m] (default: arclength + 1)",
    )
    parser.add_argument(
        "--method", type=str, default="dirlik",
        choices=["dirlik", "narrow_band"],
        help="Spectraflex cycle counting method (default: dirlik)",
    )
    args = parser.parse_args()

    # S-N curve
    sn_curve = SNCurve.dnv_d()

    print("=" * 70)
    print("Spectral Fatigue Comparison: spectraflex vs OrcaFlex")
    print("=" * 70)
    print(f"S-N curve: {sn_curve.name}")
    print(f"Sea states: {len(SEA_STATES)}")
    total_hours = sum(s["hours"] for s in SEA_STATES)
    print(f"Total exposure: {total_hours} hours ({total_hours / 8766:.2f} years)")

    # Show S-N curve mapping
    print_sn_curve_mapping(sn_curve)

    # --- Determine transfer function source ---
    if args.wn_sim is not None:
        wn_path = Path(args.wn_sim)
        if not wn_path.exists():
            print(f"\nError: white noise .sim file not found: {wn_path}")
            return

        # Identify TF from white noise simulation
        print(f"\n{'='*70}")
        print("Identify transfer function from white noise .sim")
        print(f"{'='*70}")
        tf, f = extract_transfer_function_from_sim(
            sim_path=wn_path,
            line_name=args.line,
            variable=args.variable,
            arclength=args.arclength,
        )
    else:
        # Synthetic TF (no .sim file)
        print(f"\n{'='*70}")
        print("Using synthetic transfer function (no --wn-sim provided)")
        print(f"{'='*70}")
        tf, f = create_synthetic_transfer_function()

    # --- Run spectraflex fatigue ---
    sfx = run_spectraflex_fatigue(tf, f, sn_curve, method=args.method)
    print_spectraflex_results(sfx)

    # --- Run OrcaFlex comparison if --sim provided ---
    if args.sim is not None:
        sim_path = Path(args.sim)
        if not sim_path.exists():
            print(f"\nError: model .sim file not found: {sim_path}")
            return

        arc_from = args.arc_from if args.arc_from is not None else max(0.0, args.arclength - 1.0)
        arc_to = args.arc_to if args.arc_to is not None else args.arclength + 1.0

        print(f"\n{'='*70}")
        print("OrcaFlex FatigueAnalysis (Spectral response RAOs)")
        print(f"{'='*70}")
        print(f"  Model file: {sim_path}")
        print(f"  Arc length range: {arc_from:.1f} - {arc_to:.1f} m")
        ofx = run_orcaflex_fatigue(
            sim_file=sim_path,
            line_name=args.line,
            sn_curve=sn_curve,
            arclengths=[(arc_from, arc_to)],
        )
        print(f"  Max damage: {ofx['max_damage']:.6f}")
        print(f"    at arclength={ofx['max_arclength']:.1f} m, theta={ofx['max_theta']:.1f} deg")

        # Compare
        print_comparison(sfx_damage=sfx["total_damage"], ofx_damage=ofx["max_damage"])

        print()
        print("Note: spectraflex uses a single stress variable (axial only by default),")
        print("      while OrcaFlex combines axial + bending stress at multiple theta")
        print("      positions around the pipe. Some difference is expected unless the")
        print("      extraction variable and theta position align exactly.")
    else:
        print(f"\n{'='*70}")
        print("OrcaFlex Comparison (skipped — no --sim provided)")
        print(f"{'='*70}")
        print("To add OrcaFlex comparison, also provide --sim:")
        if args.wn_sim:
            print(f"  python examples/fatigue_comparison.py --wn-sim {args.wn_sim} \\")
        else:
            print("  python examples/fatigue_comparison.py --wn-sim path/to/wn.sim \\")
        print("      --sim path/to/model.sim --line Riser")
        print()
        print("OrcaFlex FatigueAnalysis settings:")
        print("  AnalysisType:       Spectral (response RAOs)")
        print(f"  LoadCaseCount:      {len(SEA_STATES)}")
        print("  ThetaCount:         16")
        print("  RadialPosition:     Outer")

    print("\nDone!")


if __name__ == "__main__":
    main()
