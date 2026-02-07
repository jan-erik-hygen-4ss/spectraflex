"""Time history extraction from OrcaFlex simulation files.

Provides functions to extract wave elevation and response time histories
from completed .sim files using OrcFxAPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import OrcFxAPI as ofx


def get_analysis_period(
    model: ofx.Model,
    skip_buildup: bool = True,
) -> tuple[float, float]:
    """Get the analysis period from an OrcaFlex model.

    Parameters
    ----------
    model : ofx.Model
        Loaded OrcaFlex model.
    skip_buildup : bool, optional
        If True (default), return the main simulation period only,
        excluding the build-up stage.

    Returns
    -------
    tuple[float, float]
        (start_time, end_time) in seconds.
    """
    if skip_buildup:
        t_start = model.general.StageDuration[0]
        t_end = t_start + model.general.StageDuration[1]
    else:
        t_start = 0.0
        t_end = sum(model.general.StageDuration[:2])

    return t_start, t_end


def get_sample_interval(
    model: ofx.Model,
    period: tuple[float, float] | None = None,
) -> float:
    """Get the sample interval from an OrcaFlex model.

    Parameters
    ----------
    model : ofx.Model
        Loaded OrcaFlex model.
    period : tuple, optional
        (start_time, end_time). If None, uses main simulation period.

    Returns
    -------
    float
        Sample interval dt [s].
    """
    if period is None:
        period = get_analysis_period(model)

    try:
        import OrcFxAPI as ofx

        spec_period = ofx.SpecifiedPeriod(*period)
    except ImportError:
        spec_period = period

    sample_times = model.environment.SampleTimes(spec_period)
    return float(sample_times[1] - sample_times[0])


def extract_wave_elevation(
    model: ofx.Model,
    period: tuple[float, float] | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Extract wave elevation time history from an OrcaFlex model.

    Parameters
    ----------
    model : ofx.Model
        Loaded OrcaFlex model (must be in simulated state).
    period : tuple, optional
        (start_time, end_time). If None, uses main simulation period.
    position : tuple, optional
        (x, y, z) position for wave elevation, default (0, 0, 0).

    Returns
    -------
    np.ndarray
        Wave elevation time history [m].
    """
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required for extract_wave_elevation. "
            "Install with: pip install OrcFxAPI"
        ) from e

    if period is None:
        period = get_analysis_period(model)

    spec_period = ofx.SpecifiedPeriod(*period)
    wave_env = ofx.oeEnvironment(*position)

    wave = model.environment.TimeHistory("Elevation", spec_period, wave_env)
    return np.array(wave)


def extract_time_histories(
    model: ofx.Model,
    results: list[dict[str, Any]],
    period: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Extract multiple time histories from an OrcaFlex model.

    Uses GetMultipleTimeHistories for efficient batch extraction.

    Parameters
    ----------
    model : ofx.Model
        Loaded OrcaFlex model (must be in simulated state).
    results : list of dict
        Result specifications, each with keys:
        - object: str, OrcaFlex object name
        - variable: str, variable name (e.g., "Rotation 1", "Effective Tension")
        - arclength: float, optional, arc length for line objects
        - label: str, optional, friendly name for the result
    period : tuple, optional
        (start_time, end_time). If None, uses main simulation period.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping labels to time history arrays.

    Examples
    --------
    >>> import OrcFxAPI as ofx
    >>> model = ofx.Model("simulation.sim")
    >>> results = [
    ...     {"object": "Riser", "variable": "Rotation 1", "arclength": 0.0, "label": "UFJ"},
    ...     {"object": "Riser", "variable": "Effective Tension", "arclength": 0.0},
    ... ]
    >>> data = extract_time_histories(model, results)
    >>> data["UFJ"].shape
    (5120,)
    """
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required for extract_time_histories. "
            "Install with: pip install OrcFxAPI"
        ) from e

    if period is None:
        period = get_analysis_period(model)

    spec_period = ofx.SpecifiedPeriod(*period)

    # Build specifications
    specs = []
    labels = []

    for res in results:
        obj = model[res["object"]]
        var_name = res["variable"]
        label = res.get("label", f"{res['object']}_{var_name}")
        labels.append(label)

        # Build ObjectExtra if needed
        if "arclength" in res and res["arclength"] is not None:
            obj_extra = ofx.oeArcLength(res["arclength"])
        else:
            obj_extra = None

        specs.append(ofx.TimeHistorySpecification(obj, var_name, obj_extra))

    # Batch extract
    all_th = ofx.GetMultipleTimeHistories(specs, spec_period)

    # Convert to dict
    result = {}
    for i, label in enumerate(labels):
        result[label] = np.array(all_th[:, i])

    return result


def extract_from_sim(
    sim_path: str | Path,
    results: list[dict[str, Any]],
    wave_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Extract wave and response time histories from a .sim file.

    Convenience function that loads a .sim file, extracts all requested
    time histories, and returns them along with metadata.

    Parameters
    ----------
    sim_path : str or Path
        Path to the .sim file.
    results : list of dict
        Result specifications.
    wave_position : tuple, optional
        (x, y, z) for wave elevation.

    Returns
    -------
    dict
        Dictionary with keys:
        - wave: wave elevation array
        - responses: dict of response arrays
        - dt: sample interval
        - period: (t_start, t_end)
        - sim_path: path to sim file
    """
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required for extract_from_sim. "
            "Install with: pip install OrcFxAPI"
        ) from e

    sim_path = Path(sim_path)
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")

    model = ofx.Model(str(sim_path))
    period = get_analysis_period(model)
    dt = get_sample_interval(model, period)

    wave = extract_wave_elevation(model, period, wave_position)
    responses = extract_time_histories(model, results, period)

    return {
        "wave": wave,
        "responses": responses,
        "dt": dt,
        "period": period,
        "sim_path": str(sim_path),
    }


def list_available_results(
    model: ofx.Model,
    object_name: str,
) -> list[str]:
    """List available result variables for an object.

    Parameters
    ----------
    model : ofx.Model
        Loaded OrcaFlex model.
    object_name : str
        Name of the OrcaFlex object.

    Returns
    -------
    list[str]
        List of available variable names.

    Notes
    -----
    This requires accessing OrcaFlex data names, which can be done
    in the GUI by right-clicking a data field → "Data names".
    """
    obj = model[object_name]

    # Common result variables for different object types
    common_vars = [
        "X",
        "Y",
        "Z",
        "Rotation 1",
        "Rotation 2",
        "Rotation 3",
        "Effective Tension",
        "Bend Moment",
        "Wall Tension",
        "Declination",
        "Azimuth",
        "Curvature",
    ]

    # Try each variable and return those that work
    available = []
    for var in common_vars:
        try:
            # Just check if the variable is valid by getting its value
            _ = obj.StaticResult(var)
            available.append(var)
        except Exception:
            pass

    return available
