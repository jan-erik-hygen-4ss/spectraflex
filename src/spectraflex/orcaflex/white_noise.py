"""White noise model generation for OrcaFlex.

Generates YAML variation files (.yml) or modified .dat files that configure
OrcaFlex models for white noise / response calculation wave type analysis.
This module only generates files - simulation execution is left to the user.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any


def generate(
    template: str | Path,
    hs: float,
    freq_range: tuple[float, float],
    duration: float = 512.0,
    wave_direction: float = 0.0,
    output_dir: str | Path = ".",
    format: str = "yml",
    buildup_duration: float | None = None,
    current_speed: float | None = None,
    current_direction: float | None = None,
    extra_data: dict[str, Any] | None = None,
) -> Path:
    """Generate a single white noise model file.

    Creates a YAML variation file that modifies the template to use
    OrcaFlex's "Response calculation" wave type with the specified parameters.

    Parameters
    ----------
    template : str or Path
        Path to the base OrcaFlex model (.dat or .yml).
    hs : float
        Significant wave height [m]. Sets the white noise energy level via
        m₀ = (Hs/4)².
    freq_range : tuple[float, float]
        (min_freq, max_freq) in Hz for the white noise spectrum.
    duration : float, optional
        Simulation duration [s], default 512.0.
    wave_direction : float, optional
        Wave direction [deg], default 0.0.
    output_dir : str or Path, optional
        Output directory for generated file, default current directory.
    format : str, optional
        Output format: "yml" (YAML variation) or "dat". Default "yml".
    buildup_duration : float, optional
        Build-up stage duration [s]. If None, defaults to 1/min_freq.
    current_speed : float, optional
        Current speed [m/s]. If provided, sets uniform current.
    current_direction : float, optional
        Current direction [deg]. Required if current_speed is provided.
    extra_data : dict, optional
        Additional data to include in the YAML file.

    Returns
    -------
    Path
        Path to the generated file.

    Notes
    -----
    The generated YAML file uses 1-based indexing for StageDuration:
    - StageDuration[1] = build-up stage
    - StageDuration[2] = main simulation stage

    This differs from the Python API which uses 0-based indexing.
    """
    template = Path(template)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_freq, max_freq = freq_range

    # Default buildup duration: at least one full period of lowest frequency
    if buildup_duration is None:
        buildup_duration = 1.0 / min_freq

    # Generate filename
    name_parts = [
        template.stem,
        f"Hs{hs:.1f}",
        f"Dir{wave_direction:.0f}",
    ]
    if current_speed is not None:
        name_parts.append(f"Curr{current_speed:.1f}")
    name = "_".join(name_parts)

    if format == "yml":
        output_path = output_dir / f"{name}.yml"
        _write_yaml_variation(
            output_path=output_path,
            template=template,
            hs=hs,
            min_freq=min_freq,
            max_freq=max_freq,
            duration=duration,
            buildup_duration=buildup_duration,
            wave_direction=wave_direction,
            current_speed=current_speed,
            current_direction=current_direction,
            extra_data=extra_data,
        )
    elif format == "dat":
        output_path = output_dir / f"{name}.dat"
        _write_dat_file(
            output_path=output_path,
            template=template,
            hs=hs,
            min_freq=min_freq,
            max_freq=max_freq,
            duration=duration,
            buildup_duration=buildup_duration,
            wave_direction=wave_direction,
            current_speed=current_speed,
            current_direction=current_direction,
        )
    else:
        raise ValueError(f"Unknown format: {format}. Use 'yml' or 'dat'.")

    return output_path


def generate_batch(
    template: str | Path,
    matrix: dict[str, list[Any]],
    freq_range: tuple[float, float],
    duration: float = 512.0,
    output_dir: str | Path = "./batch",
    format: str = "yml",
    buildup_duration: float | None = None,
    extra_data: dict[str, Any] | None = None,
) -> list[Path]:
    """Generate batch of white noise model files for parameter sweep.

    Creates YAML variation files for all combinations of parameters
    in the matrix.

    Parameters
    ----------
    template : str or Path
        Path to the base OrcaFlex model.
    matrix : dict[str, list]
        Parameter matrix. Keys can include:
        - "hs": list of significant wave heights [m]
        - "wave_direction": list of wave directions [deg]
        - "current_speed": list of current speeds [m/s]
        - "current_direction": list of current directions [deg]
        Other keys are passed through to extra_data.
    freq_range : tuple[float, float]
        (min_freq, max_freq) in Hz.
    duration : float, optional
        Simulation duration [s], default 512.0.
    output_dir : str or Path, optional
        Output directory, default "./batch".
    format : str, optional
        Output format: "yml" or "dat". Default "yml".
    buildup_duration : float, optional
        Build-up stage duration [s].
    extra_data : dict, optional
        Additional static data for all files.

    Returns
    -------
    list[Path]
        List of paths to generated files.

    Examples
    --------
    >>> cases = generate_batch(
    ...     template="riser.dat",
    ...     matrix={
    ...         "hs": [0.5, 1.0, 2.0, 4.0],
    ...         "wave_direction": [0, 45, 90, 135, 180],
    ...         "current_speed": [0.0, 0.5, 1.0],
    ...     },
    ...     freq_range=(0.02, 0.25),
    ... )
    >>> len(cases)  # 4 * 5 * 3 = 60 cases
    60
    """
    # Separate known parameters from extra parameters
    known_params = {"hs", "wave_direction", "current_speed", "current_direction"}
    param_matrix = {k: v for k, v in matrix.items() if k in known_params}
    extra_params = {k: v for k, v in matrix.items() if k not in known_params}

    # Default values for required parameters
    if "hs" not in param_matrix:
        param_matrix["hs"] = [2.0]
    if "wave_direction" not in param_matrix:
        param_matrix["wave_direction"] = [0.0]

    # Generate all combinations
    keys = list(param_matrix.keys())
    values = [param_matrix[k] for k in keys]
    combinations = list(itertools.product(*values))

    output_paths = []
    for combo in combinations:
        params = dict(zip(keys, combo))

        # Build extra_data from static extra_params and any extra_data provided
        case_extra = {}
        if extra_data:
            case_extra.update(extra_data)
        for k, v in extra_params.items():
            # For extra params, use first value if it's a list
            case_extra[k] = v[0] if isinstance(v, list) else v

        path = generate(
            template=template,
            hs=params["hs"],
            freq_range=freq_range,
            duration=duration,
            wave_direction=params.get("wave_direction", 0.0),
            output_dir=output_dir,
            format=format,
            buildup_duration=buildup_duration,
            current_speed=params.get("current_speed"),
            current_direction=params.get("current_direction"),
            extra_data=case_extra if case_extra else None,
        )
        output_paths.append(path)

    return output_paths


def _write_yaml_variation(
    output_path: Path,
    template: Path,
    hs: float,
    min_freq: float,
    max_freq: float,
    duration: float,
    buildup_duration: float,
    wave_direction: float,
    current_speed: float | None,
    current_direction: float | None,
    extra_data: dict[str, Any] | None,
) -> None:
    """Write a YAML variation file for OrcaFlex."""
    # Convert frequencies to periods for OrcaFlex
    # OrcaFlex uses WaveTp for response calculation range
    max_period = 1.0 / min_freq  # longest period = lowest frequency

    lines = [
        "# Auto-generated white noise variation file for spectraflex",
        f"# Template: {template}",
        f"# Hs: {hs} m, Direction: {wave_direction} deg",
        f"# Frequency range: {min_freq} - {max_freq} Hz",
        "",
    ]

    # General data - stage durations
    # YAML uses 1-based indexing for arrays
    lines.extend(
        [
            "General:",
            f"  StageDuration[1]: {buildup_duration}",
            f"  StageDuration[2]: {duration}",
            "",
        ]
    )

    # Environment data - wave settings
    lines.extend(
        [
            "Environment:",
            "  WaveType: Response calculation",
            f"  WaveDirection: {wave_direction}",
            f"  WaveHs: {hs}",
            "  WaveOrigin: [0, 0]",
            "  WaveTimeOrigin: 0",
            "  WaveSpectrumDiscretisationMethod: Equal energy",
            "  WaveNumberOfComponents: 300",
            f"  WaveSpectrumMinRelFrequency: {min_freq / (1 / max_period):.4f}",
            f"  WaveSpectrumMaxRelFrequency: {max_freq / (1 / max_period):.4f}",
            "",
        ]
    )

    # Current settings if specified
    if current_speed is not None:
        curr_dir = (
            current_direction if current_direction is not None else wave_direction
        )
        lines.extend(
            [
                "  ActiveCurrent: ~",
                "  RefCurrentSpeed: " + str(current_speed),
                "  RefCurrentDirection: " + str(curr_dir),
                "",
            ]
        )

    # Extra data if provided
    if extra_data:
        for section, data in extra_data.items():
            if isinstance(data, dict):
                lines.append(f"{section}:")
                for key, value in data.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")
            else:
                lines.append(f"{section}: {data}")
                lines.append("")

    output_path.write_text("\n".join(lines))


def _write_dat_file(
    output_path: Path,
    template: Path,
    hs: float,
    min_freq: float,
    max_freq: float,
    duration: float,
    buildup_duration: float,
    wave_direction: float,
    current_speed: float | None,
    current_direction: float | None,
) -> None:
    """Write a modified .dat file using OrcFxAPI."""
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required to generate .dat files. "
            "Use format='yml' instead, or install OrcFxAPI."
        ) from e

    model = ofx.Model(str(template))

    # Set stage durations
    model.general.StageDuration[0] = buildup_duration
    model.general.StageDuration[1] = duration

    # Set wave parameters
    env = model.environment
    env.WaveType = "Response calculation"
    env.WaveDirection = wave_direction
    env.WaveHs = hs
    env.WaveSpectrumDiscretisationMethod = "Equal energy"
    env.WaveNumberOfComponents = 300

    # Set frequency range via relative frequencies
    max_period = 1.0 / min_freq
    env.WaveSpectrumMinRelFrequency = min_freq / (1 / max_period)
    env.WaveSpectrumMaxRelFrequency = max_freq / (1 / max_period)

    # Set current if specified
    if current_speed is not None:
        env.RefCurrentSpeed = current_speed
        if current_direction is not None:
            env.RefCurrentDirection = current_direction

    model.SaveData(str(output_path))


def get_case_config(path: Path) -> dict[str, Any]:
    """Extract configuration from a generated file name.

    Parameters
    ----------
    path : Path
        Path to a generated white noise file.

    Returns
    -------
    dict
        Configuration parameters extracted from filename.
    """
    stem = path.stem
    config: dict[str, Any] = {}

    # Parse filename parts
    parts = stem.split("_")
    for part in parts:
        if part.startswith("Hs"):
            config["hs"] = float(part[2:])
        elif part.startswith("Dir"):
            config["wave_direction"] = float(part[3:])
        elif part.startswith("Curr"):
            config["current_speed"] = float(part[4:])

    return config
