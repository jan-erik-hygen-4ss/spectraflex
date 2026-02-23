"""OrcaFlex FatigueAnalysis integration for spectral fatigue comparison.

Provides functions to configure and run OrcaFlex's native spectral fatigue
analysis as a comparison/validation tool, reusing spectraflex's DNV S-N
curve library.

OrcaFlex FatigueAnalysis performs spectral fatigue via response RAOs + wave
spectra using Dirlik cycle counting. This module maps spectraflex SNCurve
parameters to OrcaFlex properties and orchestrates the analysis.

OrcaFlex FatigueAnalysis Data Name Reference
---------------------------------------------
The API uses ``SetData(name, index, value)`` rather than attribute access on
sub-objects.  Key data names and their indexing:

Global (index = -1):
    AnalysisType, DamageCalculation, LoadCaseCount, ThetaCount,
    JonswapParametersModeFatigue ("Automatic" | "Partially specified" | "Fully specified"),
    SNCurveCount, SNCurvem1, SNCurveLogA1, SNCurveRegionBoundary,
    SNCurvem2, SNCurveLogA2 (read-only)

Per arc length range (row-based — use InsertDataRow/GetDataRowCount):
    FromArclength, ToArclength

Per arc length range (indexed by arc range):
    RadialPosition, SCF, ThicknessCorrectionFactor, SNCurveName

Per load case (indexed by load case):
    LoadCaseFileName, ComponentName, LoadCaseExposureTime,
    WaveHs, WaveTz, WaveTp, WaveGamma
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spectraflex.fatigue import SNCurve

if TYPE_CHECKING:
    import OrcFxAPI as ofx


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class SpectralLoadCase:
    """A single load case for OrcaFlex spectral fatigue analysis.

    Parameters
    ----------
    sim_file : str or Path
        Path to the .sim file.
    line_name : str
        OrcaFlex line object name (maps to ComponentName in the API).
    exposure_time : float
        Exposure time [seconds].
    hs : float
        Significant wave height [m].
    tz : float
        Zero-crossing period [s].
    tp : float or None
        Peak period [s] (optional).
    gamma : float or None
        JONSWAP gamma (optional).
    """

    sim_file: str | Path
    line_name: str
    exposure_time: float
    hs: float
    tz: float
    tp: float | None = None
    gamma: float | None = None


@dataclass
class OrcaFlexFatigueConfig:
    """Configuration for OrcaFlex spectral fatigue analysis.

    Parameters
    ----------
    load_cases : list of SpectralLoadCase
        Load cases to include in the analysis.
    sn_curve : SNCurve
        S-N curve for damage calculation.
    arclengths : list of tuple
        (from, to) arc length pairs [m] defining analysis range.
    theta_count : int
        Number of circumferential positions (typically 8 or 16).
    scf : float
        Stress concentration factor (per arc length range).
    thickness_correction : float
        Thickness correction factor (per arc length range).
    radial_position : str
        "Outer" or "Inner" wall position (per arc length range).
    """

    load_cases: list[SpectralLoadCase]
    sn_curve: SNCurve
    arclengths: list[tuple[float, float]]
    theta_count: int = 16
    scf: float = 1.0
    thickness_correction: float = 1.0
    radial_position: str = "Outer"


@dataclass
class OrcaFlexFatigueResult:
    """Results from an OrcaFlex spectral fatigue analysis.

    Parameters
    ----------
    overall_damage : np.ndarray
        Total damage across all load cases, shape (n_arclengths, n_theta).
    load_case_damage : np.ndarray
        Per-load-case damage, shape (n_load_cases, n_arclengths, n_theta).
    theta : np.ndarray
        Circumferential positions [degrees].
    arclengths : np.ndarray
        Arc length positions [m].
    max_damage : float
        Maximum damage value across all positions.
    max_damage_arclength : float
        Arc length [m] at maximum damage.
    max_damage_theta : float
        Theta [degrees] at maximum damage.
    config : OrcaFlexFatigueConfig
        Configuration used for the analysis.
    """

    overall_damage: np.ndarray
    load_case_damage: np.ndarray
    theta: np.ndarray
    arclengths: np.ndarray
    max_damage: float
    max_damage_arclength: float
    max_damage_theta: float
    config: OrcaFlexFatigueConfig


# =============================================================================
# S-N Curve Mapping
# =============================================================================


def sn_curve_to_orcaflex(
    sn_curve: SNCurve,
    stress_factor: float = 1e3,
) -> dict[str, Any]:
    """Map a spectraflex SNCurve to OrcaFlex fatigue analysis property names/values.

    Converts log_a from spectraflex stress units (MPa) to OrcaFlex stress
    units (kPa by default).  The S-N equation is:

        log10(N) = log_a - m * log10(S)

    When converting from MPa to kPa (factor = 1000):

        log_a_kPa = log_a_MPa + m * log10(1000) = log_a_MPa + 3 * m

    OrcaFlex S-N properties must be set in this specific order:
    1. SNCurvem1, SNCurveLogA1 (first slope)
    2. SNCurveRegionBoundary = n_transition (cycle count boundary)
    3. SNCurvem2 (only settable after boundary is set)

    SNCurveLogA2 is read-only — OrcaFlex auto-calculates it from continuity.

    Parameters
    ----------
    sn_curve : SNCurve
        spectraflex S-N curve definition (stress in MPa).
    stress_factor : float, optional
        Ratio of OrcaFlex stress unit to MPa.  Default 1e3 (kPa/MPa),
        matching OrcaFlex's default unit system (kN, m → kPa).
        Set to 1.0 if your OrcaFlex model already uses MPa.

    Returns
    -------
    dict[str, Any]
        Ordered mapping of OrcaFlex property names to values.
        Keys: SNCurvem1, SNCurveLogA1, SNCurveRegionBoundary, SNCurvem2.

    Examples
    --------
    >>> from spectraflex.fatigue import SNCurve
    >>> curve = SNCurve.dnv_d()
    >>> props = sn_curve_to_orcaflex(curve)
    >>> props["SNCurvem1"]
    3.0
    >>> props["SNCurveLogA1"]  # 12.164 + 3 * 3.0 = 21.164
    21.164
    >>> props["SNCurveRegionBoundary"]
    10000000.0
    """
    log_factor = np.log10(stress_factor)

    return {
        "SNCurvem1": sn_curve.m1,
        "SNCurveLogA1": sn_curve.log_a1 + sn_curve.m1 * log_factor,
        "SNCurveRegionBoundary": float(sn_curve.n_transition),
        "SNCurvem2": sn_curve.m2,
    }


# =============================================================================
# OrcaFlex FatigueAnalysis Creation
# =============================================================================


def create_fatigue_analysis(
    config: OrcaFlexFatigueConfig,
) -> ofx.FatigueAnalysis:
    """Create and configure an OrcFxAPI.FatigueAnalysis object.

    Parameters
    ----------
    config : OrcaFlexFatigueConfig
        Full configuration for the fatigue analysis.

    Returns
    -------
    ofx.FatigueAnalysis
        Configured fatigue analysis ready for calculation.

    Raises
    ------
    ImportError
        If OrcFxAPI is not installed.
    FileNotFoundError
        If any sim file does not exist.

    Notes
    -----
    Setup order matters for the OrcaFlex FatigueAnalysis API:

    1. Set AnalysisType and LoadCaseCount.
    2. Insert arc length range rows (row-based data).
    3. Set arc length range properties (RadialPosition, SCF, etc.).
    4. Set S-N curve properties (only available after arc length ranges exist).
    5. Set load case properties (LoadCaseFileName, ComponentName, wave params).
    """
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required for create_fatigue_analysis. "
            "Install with: pip install OrcFxAPI"
        ) from e

    fa = ofx.FatigueAnalysis()

    # --- 1. Global settings ---
    fa.AnalysisType = "Spectral (response RAOs)"
    fa.ThetaCount = config.theta_count
    fa.LoadCaseCount = len(config.load_cases)

    # --- 2. Arc length ranges (row-based: must InsertDataRow first) ---
    for i, (arc_from, arc_to) in enumerate(config.arclengths):
        fa.InsertDataRow("FromArclength", i)
        fa.SetData("FromArclength", i, arc_from)
        fa.SetData("ToArclength", i, arc_to)

    # --- 3. Per-arc-range properties ---
    for i in range(len(config.arclengths)):
        fa.SetData("RadialPosition", i, config.radial_position)
        fa.SetData("SCF", i, config.scf)
        fa.SetData("ThicknessCorrectionFactor", i, config.thickness_correction)

    # --- 4. S-N curve (global, but only available after arc ranges exist) ---
    sn_props = sn_curve_to_orcaflex(config.sn_curve)
    fa.SNCurvem1 = sn_props["SNCurvem1"]
    fa.SNCurveLogA1 = sn_props["SNCurveLogA1"]
    fa.SNCurveRegionBoundary = sn_props["SNCurveRegionBoundary"]
    fa.SNCurvem2 = sn_props["SNCurvem2"]

    # --- 5. JONSWAP parameter mode (must be set before Tp/Gamma) ---
    # If any load case specifies Tp or Gamma, switch from "Automatic" to
    # "Partially specified" so OrcaFlex accepts those values.
    has_tp_or_gamma = any(
        lc.tp is not None or lc.gamma is not None for lc in config.load_cases
    )
    if has_tp_or_gamma:
        fa.JonswapParametersModeFatigue = "Partially specified"

    # --- 6. Load cases ---
    for i, lc in enumerate(config.load_cases):
        sim_path = Path(lc.sim_file)
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulation file not found: {sim_path}")

        fa.SetData("LoadCaseFileName", i, str(sim_path))
        fa.SetData("LoadCaseLineName", i, lc.line_name)
        fa.SetData("LoadCaseExposureTime", i, lc.exposure_time)
        fa.SetData("WaveHs", i, lc.hs)
        fa.SetData("WaveTz", i, lc.tz)

        if lc.tp is not None:
            fa.SetData("WaveTp", i, lc.tp)
        if lc.gamma is not None:
            fa.SetData("WaveGamma", i, lc.gamma)

    return fa


# =============================================================================
# Result Extraction
# =============================================================================


def extract_results(
    fa: ofx.FatigueAnalysis,
    config: OrcaFlexFatigueConfig,
) -> OrcaFlexFatigueResult:
    """Extract numpy arrays from a completed OrcaFlex fatigue analysis.

    Parameters
    ----------
    fa : ofx.FatigueAnalysis
        Completed fatigue analysis (after Calculate() has been called).
    config : OrcaFlexFatigueConfig
        Configuration used for the analysis.

    Returns
    -------
    OrcaFlexFatigueResult
        Structured result with damage arrays and metadata.

    Notes
    -----
    OrcaFlex returns structured numpy arrays from its result properties:

    - ``fa.theta`` → 1D float array of theta values [degrees]
    - ``fa.overallDamage`` → structured array with fields 'Damage', 'Life';
      shape (n_arclengths, n_theta)
    - ``fa.loadCaseDamage`` → structured array with fields 'MeanResponse',
      'Damage', 'DamageRate'; shape (n_load_cases, n_arclengths, n_theta)
    - ``fa.outputPointDetails`` → structured array with field 'arclength'
    """
    # Theta positions from the analysis
    theta = fa.theta

    # Output point details give us the arc lengths
    details = fa.outputPointDetails
    arclengths = details["arclength"]

    # Overall damage: structured array → extract 'Damage' field
    overall_raw = fa.overallDamage
    overall_damage = overall_raw["Damage"]

    # Per-load-case damage
    lc_raw = fa.loadCaseDamage
    load_case_damage = lc_raw["Damage"]

    # Find maximum damage location
    max_idx = np.unravel_index(np.argmax(overall_damage), overall_damage.shape)
    max_damage = float(overall_damage[max_idx])
    max_damage_arclength = float(arclengths[max_idx[0]])
    max_damage_theta = float(theta[max_idx[1]]) if len(max_idx) > 1 else 0.0

    return OrcaFlexFatigueResult(
        overall_damage=np.asarray(overall_damage),
        load_case_damage=np.asarray(load_case_damage),
        theta=np.asarray(theta),
        arclengths=np.asarray(arclengths),
        max_damage=max_damage,
        max_damage_arclength=max_damage_arclength,
        max_damage_theta=max_damage_theta,
        config=config,
    )


# =============================================================================
# Orchestrator
# =============================================================================


def run_spectral_fatigue(
    config: OrcaFlexFatigueConfig,
) -> OrcaFlexFatigueResult:
    """Run a complete OrcaFlex spectral fatigue analysis.

    Creates the FatigueAnalysis, runs the calculation, and extracts results.

    Parameters
    ----------
    config : OrcaFlexFatigueConfig
        Full configuration for the fatigue analysis.

    Returns
    -------
    OrcaFlexFatigueResult
        Structured result with damage arrays and metadata.

    Examples
    --------
    >>> from spectraflex.fatigue import SNCurve
    >>> config = OrcaFlexFatigueConfig(
    ...     load_cases=[
    ...         SpectralLoadCase(
    ...             sim_file="sim1.sim",
    ...             line_name="Riser",
    ...             exposure_time=3600,
    ...             hs=3.0, tz=8.0,
    ...         ),
    ...     ],
    ...     sn_curve=SNCurve.dnv_d(),
    ...     arclengths=[(0, 100)],
    ... )
    >>> result = run_spectral_fatigue(config)  # doctest: +SKIP
    >>> print(f"Max damage: {result.max_damage:.6f}")  # doctest: +SKIP
    """
    fa = create_fatigue_analysis(config)
    fa.Calculate()
    return extract_results(fa, config)


# =============================================================================
# Comparison
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing OrcaFlex and spectraflex fatigue results.

    Parameters
    ----------
    ofx_damage : float
        Maximum damage from OrcaFlex analysis.
    sfx_damage : float
        Maximum damage from spectraflex analysis.
    ratio : float
        OrcaFlex / spectraflex damage ratio.
    abs_diff : float
        Absolute difference (OrcaFlex - spectraflex).
    rel_diff : float
        Relative difference as fraction of spectraflex damage.
    """

    ofx_damage: float
    sfx_damage: float
    ratio: float
    abs_diff: float
    rel_diff: float


def compare_results(
    ofx_damage: float,
    sfx_damage: float,
) -> ComparisonResult:
    """Compare OrcaFlex and spectraflex fatigue damage values.

    Parameters
    ----------
    ofx_damage : float
        Damage value from OrcaFlex fatigue analysis.
    sfx_damage : float
        Damage value from spectraflex fatigue analysis.

    Returns
    -------
    ComparisonResult
        Comparison metrics including ratio, absolute and relative differences.

    Examples
    --------
    >>> result = compare_results(ofx_damage=0.0105, sfx_damage=0.0100)
    >>> result.ratio
    1.05
    >>> result.rel_diff
    0.05
    """
    if sfx_damage == 0:
        ratio = float("inf") if ofx_damage > 0 else 1.0
        rel_diff = float("inf") if ofx_damage > 0 else 0.0
    else:
        ratio = ofx_damage / sfx_damage
        rel_diff = (ofx_damage - sfx_damage) / sfx_damage

    abs_diff = ofx_damage - sfx_damage

    return ComparisonResult(
        ofx_damage=ofx_damage,
        sfx_damage=sfx_damage,
        ratio=ratio,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )
