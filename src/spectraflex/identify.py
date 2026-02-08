"""Transfer function identification from time histories and OrcaFlex simulations.

Provides functions to identify complex transfer functions H(f) using
cross-spectral analysis of input (wave elevation) and output (responses)
time histories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy import signal

from spectraflex import transfer_function


def from_time_histories(
    wave_elevation: np.ndarray,
    responses: dict[str, np.ndarray],
    dt: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> xr.Dataset:
    """Identify transfer functions from time history arrays.

    Uses Welch's method with cross-spectral density estimation to compute
    H(f) = S_xy(f) / S_xx(f).

    Parameters
    ----------
    wave_elevation : np.ndarray
        1D array of wave elevation time history [m].
    responses : dict[str, np.ndarray]
        Dictionary mapping response variable names to 1D time history arrays.
    dt : float
        Sample interval [s].
    nperseg : int, optional
        Length of each FFT segment, default 1024.
    noverlap : int, optional
        Number of overlapping points, default nperseg // 2 (50% overlap).
    window : str, optional
        Window function, default "hann".
    detrend : str or bool, optional
        Detrend option for scipy.signal, default "constant" (remove mean).

    Returns
    -------
    xr.Dataset
        TransferFunction Dataset with magnitude, phase, coherence,
        and optionally Sxx, Syy.

    Notes
    -----
    The transfer function is estimated as:
        H(f) = S_xy(f) / S_xx(f)

    Coherence is:
        γ²(f) = |S_xy(f)|² / (S_xx(f) * S_yy(f))

    Values of γ² close to 1 indicate a linear relationship at that frequency.
    """
    wave = np.asarray(wave_elevation, dtype=np.float64)
    fs = 1.0 / dt

    if noverlap is None:
        noverlap = nperseg // 2

    variable_names = list(responses.keys())
    n_var = len(variable_names)

    # Compute input auto-spectrum
    f, sxx = signal.welch(
        wave,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
    )

    # Only use positive frequencies (excluding DC)
    pos_mask = f > 0
    f = f[pos_mask]
    sxx = sxx[pos_mask]
    n_freq = len(f)

    # Initialize output arrays
    magnitude = np.zeros((n_freq, n_var))
    phase = np.zeros((n_freq, n_var))
    coherence = np.zeros((n_freq, n_var))
    syy = np.zeros((n_freq, n_var))

    for i, (name, resp) in enumerate(responses.items()):
        resp = np.asarray(resp, dtype=np.float64)

        if len(resp) != len(wave):
            raise ValueError(
                f"Response '{name}' length {len(resp)} does not match "
                f"wave length {len(wave)}"
            )

        # Compute output auto-spectrum
        _, syy_full = signal.welch(
            resp,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density",
        )
        syy[:, i] = syy_full[pos_mask]

        # Compute cross-spectral density
        _, sxy = signal.csd(
            wave,
            resp,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density",
        )
        sxy = sxy[pos_mask]

        # Compute transfer function H(f) = S_xy / S_xx
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            h = sxy / sxx
            h = np.where(np.isfinite(h), h, 0.0)

        magnitude[:, i] = np.abs(h)
        phase[:, i] = np.angle(h)

        # Compute coherence γ² = |S_xy|² / (S_xx * S_yy)
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.abs(sxy) ** 2 / (sxx * syy[:, i])
            coh = np.where(np.isfinite(coh), coh, 0.0)
            coh = np.clip(coh, 0.0, 1.0)

        coherence[:, i] = coh

    return transfer_function.create(
        frequency=f,
        magnitude=magnitude,
        phase=phase,
        coherence=coherence,
        variable_names=variable_names,
        sxx=sxx,
        syy=syy,
        nperseg=nperseg,
        noverlap=noverlap,
        sample_interval=dt,
        window=window,
    )


def from_sim(
    sim_path: str | Path,
    results: list[dict[str, Any]],
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    config: dict[str, Any] | None = None,
    wave_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> xr.Dataset:
    """Identify transfer functions from an OrcaFlex simulation file.

    Extracts wave elevation and response time histories from a completed
    .sim file and identifies H(f) using cross-spectral analysis.

    Parameters
    ----------
    sim_path : str or Path
        Path to the OrcaFlex .sim file.
    results : list of dict
        List of result specifications, each with keys:
        - object: str, OrcaFlex object name (e.g., "Riser")
        - variable: str, OrcaFlex variable name (e.g., "Rotation 1")
        - arclength: float, optional, arc length for line objects
        - label: str, optional, friendly name for the variable
    nperseg : int, optional
        FFT segment length, default 1024.
    noverlap : int, optional
        Overlap points, default nperseg // 2.
    window : str, optional
        Window function, default "hann".
    config : dict, optional
        Configuration metadata to store in the Dataset.
    wave_position : tuple of float, optional
        (x, y, z) position for wave elevation extraction, default (0, 0, 0).

    Returns
    -------
    xr.Dataset
        TransferFunction Dataset.

    Raises
    ------
    ImportError
        If OrcFxAPI is not available.
    FileNotFoundError
        If the .sim file doesn't exist.

    Notes
    -----
    Requires OrcFxAPI and an OrcaFlex licence. The simulation must already
    be completed (state = SimulationComplete).
    """
    try:
        import OrcFxAPI as ofx
    except ImportError as e:
        raise ImportError(
            "OrcFxAPI is required for from_sim(). Install with: pip install OrcFxAPI"
        ) from e

    sim_path = Path(sim_path)
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")

    # Load the simulation
    model = ofx.Model(str(sim_path))

    # Get analysis period (skip build-up stage)
    t_start = model.general.StageDuration[0]
    t_end = t_start + model.general.StageDuration[1]
    period = ofx.SpecifiedPeriod(t_start, t_end)

    # Get sample interval
    sample_times = model.environment.SampleTimes(period)
    dt = float(sample_times[1] - sample_times[0])

    # Extract wave elevation
    wave_env = ofx.oeEnvironment(*wave_position)
    wave = np.array(model.environment.TimeHistory("Elevation", period, wave_env))

    # Build time history specifications for batch extraction
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

    # Batch extract all time histories
    all_th = ofx.GetMultipleTimeHistories(specs, period)

    # Convert to dict of arrays
    responses = {}
    for i, label in enumerate(labels):
        responses[label] = np.array(all_th[:, i])

    # Identify transfer functions
    tf = from_time_histories(
        wave_elevation=wave,
        responses=responses,
        dt=dt,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )

    # Add metadata
    tf.attrs["orcaflex_model"] = str(sim_path)
    tf.attrs["sim_duration"] = float(t_end - t_start)
    if config is not None:
        tf.attrs["config"] = config

    return tf


def from_spectra(
    spectra_path: str | Path,
    config: dict[str, Any] | None = None,
    freq_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Create TransferFunction from pre-computed spectra file.

    Loads Sxx, Syy, Sxy from an .npz file (e.g., created by a post-calculation
    action) and computes H(f) without needing an OrcaFlex licence.

    Parameters
    ----------
    spectra_path : str or Path
        Path to the .npz file containing:
        - frequency: 1D array of frequencies [Hz]
        - Sxx: 1D array of input auto-spectrum
        - Syy: 2D array of output auto-spectra (n_freq, n_var)
        - Sxy: 2D complex array of cross-spectra (n_freq, n_var)
        - variable_names: list of variable names
    config : dict, optional
        Configuration metadata.
    freq_range : tuple of float, optional
        (f_min, f_max) to filter frequencies to the valid range where
        the white noise wave had energy. Recommended to match the
        frequency range used in the OrcaFlex response calculation.

    Returns
    -------
    xr.Dataset
        TransferFunction Dataset.
    """
    spectra_path = Path(spectra_path)
    if not spectra_path.exists():
        raise FileNotFoundError(f"Spectra file not found: {spectra_path}")

    data = np.load(spectra_path, allow_pickle=True)

    frequency = data["frequency"]
    sxx = data["Sxx"]
    syy = data["Syy"]
    sxy = data["Sxy"]
    variable_names = list(data["variable_names"])

    # Ensure 2D shape
    if syy.ndim == 1:
        syy = syy[:, np.newaxis]
    if sxy.ndim == 1:
        sxy = sxy[:, np.newaxis]

    # Filter to valid frequency range if specified
    if freq_range is not None:
        f_min, f_max = freq_range
        mask = (frequency >= f_min) & (frequency <= f_max)
        frequency = frequency[mask]
        sxx = sxx[mask]
        syy = syy[mask, :]
        sxy = sxy[mask, :]

    n_freq, n_var = syy.shape

    # Compute transfer function H(f) = S_xy / S_xx
    magnitude = np.zeros((n_freq, n_var))
    phase = np.zeros((n_freq, n_var))
    coherence = np.zeros((n_freq, n_var))

    for i in range(n_var):
        with np.errstate(divide="ignore", invalid="ignore"):
            h = sxy[:, i] / sxx
            h = np.where(np.isfinite(h), h, 0.0)

        magnitude[:, i] = np.abs(h)
        phase[:, i] = np.angle(h)

        # Coherence
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.abs(sxy[:, i]) ** 2 / (sxx * syy[:, i])
            coh = np.where(np.isfinite(coh), coh, 0.0)
            coh = np.clip(coh, 0.0, 1.0)

        coherence[:, i] = coh

    tf = transfer_function.create(
        frequency=frequency,
        magnitude=magnitude,
        phase=phase,
        coherence=coherence,
        variable_names=variable_names,
        sxx=sxx,
        syy=syy,
        config=config,
    )

    tf.attrs["spectra_file"] = str(spectra_path)

    return tf


def coherence_mask(
    tf: xr.Dataset,
    threshold: float = 0.5,
) -> xr.DataArray:
    """Create a boolean mask for frequencies with sufficient coherence.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    threshold : float, optional
        Minimum coherence value, default 0.5.

    Returns
    -------
    xr.DataArray
        Boolean mask, True where coherence >= threshold.
    """
    return tf["coherence"] >= threshold


def apply_coherence_mask(
    tf: xr.Dataset,
    threshold: float = 0.5,
    fill_value: float = 0.0,
) -> xr.Dataset:
    """Set transfer function values to fill_value where coherence is low.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    threshold : float, optional
        Minimum coherence value, default 0.5.
    fill_value : float, optional
        Value to use where coherence is below threshold, default 0.0.

    Returns
    -------
    xr.Dataset
        New Dataset with masked values.
    """
    mask = coherence_mask(tf, threshold)

    tf_masked = tf.copy(deep=True)
    tf_masked["magnitude"] = tf["magnitude"].where(mask, fill_value)
    tf_masked["phase"] = tf["phase"].where(mask, 0.0)

    return tf_masked
