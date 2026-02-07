"""Batch generation utilities for parameter sweeps.

Provides functions to generate case matrices for batch simulations,
manage case configurations, and track batch processing status.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CaseConfig:
    """Configuration for a single simulation case.

    Attributes
    ----------
    hs : float
        Significant wave height [m].
    wave_direction : float
        Wave direction [deg].
    current_speed : float
        Current speed [m/s].
    current_direction : float
        Current direction [deg].
    extra : dict
        Additional configuration parameters.
    label : str
        Case label/identifier.
    """

    hs: float = 2.0
    wave_direction: float = 0.0
    current_speed: float = 0.0
    current_direction: float = 0.0
    extra: dict = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self._generate_label()

    def _generate_label(self) -> str:
        """Generate a label from config values."""
        parts = [
            f"Hs{self.hs:.1f}",
            f"Dir{self.wave_direction:.0f}",
        ]
        if self.current_speed > 0:
            parts.append(f"Curr{self.current_speed:.1f}")
        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "hs": self.hs,
            "wave_direction": self.wave_direction,
            "current_speed": self.current_speed,
            "current_direction": self.current_direction,
        }
        d.update(self.extra)
        return d


def generate_case_matrix(
    hs: list[float] | float = 2.0,
    wave_direction: list[float] | float = 0.0,
    current_speed: list[float] | float = 0.0,
    current_direction: list[float] | float | None = None,
    **extra: list[Any] | Any,
) -> list[CaseConfig]:
    """Generate a matrix of case configurations.

    Creates all combinations of the specified parameters.

    Parameters
    ----------
    hs : float or list of float
        Significant wave height(s) [m].
    wave_direction : float or list of float
        Wave direction(s) [deg].
    current_speed : float or list of float
        Current speed(s) [m/s].
    current_direction : float or list of float, optional
        Current direction(s) [deg]. If None, defaults to wave_direction.
    **extra
        Additional parameters to vary.

    Returns
    -------
    list[CaseConfig]
        List of case configurations.

    Examples
    --------
    >>> cases = generate_case_matrix(
    ...     hs=[1.0, 2.0, 4.0],
    ...     wave_direction=[0, 45, 90],
    ...     current_speed=[0.0, 0.5],
    ... )
    >>> len(cases)  # 3 * 3 * 2 = 18
    18
    """

    # Ensure all parameters are lists
    def to_list(x: Any) -> list:
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    hs_list = to_list(hs)
    wave_dir_list = to_list(wave_direction)
    curr_speed_list = to_list(current_speed)

    # Handle extra parameters
    extra_lists = {k: to_list(v) for k, v in extra.items()}

    # Generate combinations
    # If current_direction is None, use wave_direction for each case (not a separate dimension)
    if current_direction is None:
        base_combos = [
            (h, w, c, w)  # current_direction follows wave_direction
            for h, w, c in itertools.product(hs_list, wave_dir_list, curr_speed_list)
        ]
    else:
        curr_dir_list = to_list(current_direction)
        base_combos = list(
            itertools.product(hs_list, wave_dir_list, curr_speed_list, curr_dir_list)
        )

    # Generate extra combinations if any
    if extra_lists:
        extra_keys = list(extra_lists.keys())
        extra_values = [extra_lists[k] for k in extra_keys]
        extra_combos = list(itertools.product(*extra_values))
    else:
        extra_keys = []
        extra_combos = [()]

    # Combine all
    cases = []
    for base in base_combos:
        for extra_combo in extra_combos:
            extra_dict = dict(zip(extra_keys, extra_combo))
            case = CaseConfig(
                hs=base[0],
                wave_direction=base[1],
                current_speed=base[2],
                current_direction=base[3],
                extra=extra_dict,
            )
            cases.append(case)

    return cases


def matrix_to_dataframe(cases: list[CaseConfig]) -> Any:
    """Convert case matrix to a pandas DataFrame.

    Parameters
    ----------
    cases : list[CaseConfig]
        List of case configurations.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per case.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for matrix_to_dataframe") from e

    records = []
    for i, case in enumerate(cases):
        record = {
            "case_id": i,
            "label": case.label,
            **case.to_dict(),
        }
        records.append(record)

    return pd.DataFrame(records)


def find_completed_sims(
    output_dir: str | Path,
    pattern: str = "*.sim",
) -> list[Path]:
    """Find completed simulation files in a directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory to search.
    pattern : str, optional
        Glob pattern for sim files, default "*.sim".

    Returns
    -------
    list[Path]
        List of paths to .sim files.
    """
    output_dir = Path(output_dir)
    return sorted(output_dir.glob(pattern))


def find_spectra_files(
    output_dir: str | Path,
    pattern: str = "*_spectra.npz",
) -> list[Path]:
    """Find extracted spectra files in a directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory to search.
    pattern : str, optional
        Glob pattern for spectra files, default "*_spectra.npz".

    Returns
    -------
    list[Path]
        List of paths to .npz files.
    """
    output_dir = Path(output_dir)
    return sorted(output_dir.glob(pattern))


def get_batch_status(
    output_dir: str | Path,
    expected_cases: list[CaseConfig] | int | None = None,
) -> dict[str, Any]:
    """Get status of a batch run.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing batch output.
    expected_cases : list[CaseConfig] or int, optional
        Expected cases or count for completion percentage.

    Returns
    -------
    dict
        Status information:
        - n_sims: number of .sim files found
        - n_spectra: number of spectra files found
        - n_expected: expected count (if provided)
        - completion: completion percentage (if expected provided)
        - sim_files: list of sim file paths
        - spectra_files: list of spectra file paths
    """
    output_dir = Path(output_dir)

    sim_files = find_completed_sims(output_dir)
    spectra_files = find_spectra_files(output_dir)

    status: dict[str, Any] = {
        "n_sims": len(sim_files),
        "n_spectra": len(spectra_files),
        "sim_files": sim_files,
        "spectra_files": spectra_files,
    }

    if expected_cases is not None:
        if isinstance(expected_cases, list):
            n_expected = len(expected_cases)
        else:
            n_expected = expected_cases

        status["n_expected"] = n_expected
        status["completion"] = len(spectra_files) / n_expected if n_expected > 0 else 0

    return status


def config_from_filename(
    path: Path,
) -> dict[str, float]:
    """Extract configuration from a generated filename.

    Parameters
    ----------
    path : Path
        Path to a generated file.

    Returns
    -------
    dict
        Extracted configuration values.
    """
    stem = path.stem

    # Remove _spectra suffix if present
    if stem.endswith("_spectra"):
        stem = stem[:-8]

    config: dict[str, float] = {}
    parts = stem.split("_")

    for part in parts:
        if part.startswith("Hs"):
            try:
                config["hs"] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith("Dir"):
            try:
                config["wave_direction"] = float(part[3:])
            except ValueError:
                pass
        elif part.startswith("Curr"):
            try:
                config["current_speed"] = float(part[4:])
            except ValueError:
                pass

    return config


def match_spectra_to_configs(
    spectra_files: list[Path],
    cases: list[CaseConfig],
) -> list[tuple[Path, CaseConfig]]:
    """Match spectra files to their configurations.

    Parameters
    ----------
    spectra_files : list[Path]
        List of spectra file paths.
    cases : list[CaseConfig]
        List of case configurations.

    Returns
    -------
    list[tuple[Path, CaseConfig]]
        List of (file, config) pairs.
    """
    matched = []

    for path in spectra_files:
        file_config = config_from_filename(path)

        for case in cases:
            case_dict = case.to_dict()
            match = True

            for key, value in file_config.items():
                if key in case_dict:
                    if not np.isclose(value, case_dict[key], rtol=0.01):
                        match = False
                        break

            if match:
                matched.append((path, case))
                break

    return matched
