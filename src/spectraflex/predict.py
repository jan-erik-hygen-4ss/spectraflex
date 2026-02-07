"""Spectral response prediction and time series synthesis.

Given a transfer function H(f) and an input wave spectrum S_xx(f),
computes the response spectrum S_yy(f) = |H(f)|² * S_xx(f) and
derives statistics or synthesizes time series.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from spectraflex import statistics as stats


def response_spectrum(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    interpolate: bool = True,
) -> xr.Dataset:
    """Compute response spectrum from transfer function and wave spectrum.

    S_yy(f) = |H(f)|² * S_xx(f)

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset with magnitude, phase, coherence.
    wave_spectrum : xr.DataArray
        Wave spectrum S_xx(f) [m²/Hz] with frequency coordinate.
    interpolate : bool, optional
        If True, interpolate wave spectrum to transfer function frequencies.
        Default True.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - Sxx: input wave spectrum on TF frequency grid
        - Syy: response spectrum (frequency, variable)
        - magnitude: |H(f)|² used for the computation
    """
    tf_freq = tf.coords["frequency"].values
    variable_names = tf.coords["variable"].values

    # Get wave spectrum values on TF frequency grid
    if interpolate:
        sxx_interp = wave_spectrum.interp(frequency=tf_freq, method="linear")
        sxx = sxx_interp.values
    else:
        # Assume frequencies match
        sxx = wave_spectrum.values

    # Ensure non-negative after interpolation
    sxx = np.maximum(sxx, 0.0)

    # Compute response spectrum S_yy = |H|² * S_xx
    h_mag_sq = tf["magnitude"].values ** 2  # shape (n_freq, n_var)
    syy = h_mag_sq * sxx[:, np.newaxis]  # broadcast to (n_freq, n_var)

    # Build output Dataset
    ds = xr.Dataset(
        data_vars={
            "Sxx": (["frequency"], sxx),
            "Syy": (["frequency", "variable"], syy),
            "H_magnitude_sq": (["frequency", "variable"], h_mag_sq),
        },
        coords={
            "frequency": tf_freq,
            "variable": variable_names,
        },
        attrs={
            "description": "Response spectrum from transfer function prediction",
        },
    )

    # Copy wave spectrum attributes
    if hasattr(wave_spectrum, "attrs"):
        for key in ["hs", "tp", "gamma"]:
            if key in wave_spectrum.attrs:
                ds.attrs[f"wave_{key}"] = wave_spectrum.attrs[key]

    return ds


def response_statistics(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    duration: float = 10800.0,
    interpolate: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute response statistics for each variable.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    wave_spectrum : xr.DataArray
        Wave spectrum S_xx(f) [m²/Hz].
    duration : float, optional
        Duration for MPM calculation [s], default 10800 (3 hours).
    interpolate : bool, optional
        If True, interpolate wave spectrum to TF frequencies.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {variable_name: {statistic_name: value, ...}, ...}
        Statistics include: m0, m2, m4, hs, tp, tz, mpm, sigma
    """
    resp = response_spectrum(tf, wave_spectrum, interpolate=interpolate)
    f = resp.coords["frequency"].values
    variable_names = resp.coords["variable"].values

    result = {}
    for i, var in enumerate(variable_names):
        syy = resp["Syy"].values[:, i]
        result[str(var)] = stats.all_statistics(f, syy, duration=duration)

    return result


def statistics(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    duration: float = 10800.0,
    interpolate: bool = True,
) -> dict[str, dict[str, float]]:
    """Alias for response_statistics for API compatibility."""
    return response_statistics(tf, wave_spectrum, duration, interpolate)


def synthesize_timeseries(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    duration: float,
    dt: float,
    seed: int | None = None,
    interpolate: bool = True,
) -> xr.Dataset:
    """Synthesize response time series from transfer function and wave spectrum.

    Uses spectral synthesis with random phases:
        x(t) = Σ A_i * cos(2π*f_i*t + φ_wave_i + φ_tf_i)

    where:
        - A_i = |H(f_i)| * sqrt(2 * S_xx(f_i) * Δf)
        - φ_wave_i ~ Uniform(0, 2π) (random wave phase)
        - φ_tf_i = arg(H(f_i)) (transfer function phase)

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    wave_spectrum : xr.DataArray
        Wave spectrum S_xx(f) [m²/Hz].
    duration : float
        Duration of time series [s].
    dt : float
        Time step [s].
    seed : int, optional
        Random seed for reproducibility.
    interpolate : bool, optional
        If True, interpolate wave spectrum to TF frequencies.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - wave: synthesized wave elevation time series
        - {variable_name}: response time series for each variable
        - time: time coordinate [s]
    """
    rng = np.random.default_rng(seed)

    tf_freq = tf.coords["frequency"].values
    variable_names = tf.coords["variable"].values
    n_freq = len(tf_freq)

    # Get wave spectrum on TF frequency grid
    if interpolate:
        sxx_interp = wave_spectrum.interp(frequency=tf_freq, method="linear")
        sxx = np.maximum(sxx_interp.values, 0.0)
    else:
        sxx = np.maximum(wave_spectrum.values, 0.0)

    # Frequency spacing
    if n_freq > 1:
        df = np.diff(tf_freq)
        df = np.concatenate([[df[0]], df])  # extend to match length
    else:
        df = np.ones(1)

    # Wave amplitudes
    a_wave = np.sqrt(2.0 * sxx * df)

    # Random wave phases
    phi_wave = rng.uniform(0, 2 * np.pi, n_freq)

    # Time array
    t = np.arange(0, duration, dt)
    n_t = len(t)

    # Synthesize wave elevation
    wave_ts = np.zeros(n_t)
    for k in range(n_freq):
        wave_ts += a_wave[k] * np.cos(2 * np.pi * tf_freq[k] * t + phi_wave[k])

    # Synthesize response time series
    h_mag = tf["magnitude"].values  # (n_freq, n_var)
    h_phase = tf["phase"].values  # (n_freq, n_var)

    responses = {}
    for i, var in enumerate(variable_names):
        resp_ts = np.zeros(n_t)
        for k in range(n_freq):
            a_resp = a_wave[k] * h_mag[k, i]
            phi_resp = phi_wave[k] + h_phase[k, i]
            resp_ts += a_resp * np.cos(2 * np.pi * tf_freq[k] * t + phi_resp)
        responses[str(var)] = resp_ts

    # Build output Dataset
    data_vars = {"wave": (["time"], wave_ts)}
    for var, ts in responses.items():
        data_vars[var] = (["time"], ts)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": t},
        attrs={
            "description": "Synthesized time series from spectral method",
            "duration": duration,
            "dt": dt,
            "seed": seed,
        },
    )

    return ds


def synthesize_timeseries_fft(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    n_samples: int,
    dt: float,
    seed: int | None = None,
) -> xr.Dataset:
    """Synthesize time series using inverse FFT (faster for long durations).

    This is an alternative to synthesize_timeseries that uses FFT for
    efficiency when generating long time series.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    wave_spectrum : xr.DataArray
        Wave spectrum S_xx(f) [m²/Hz].
    n_samples : int
        Number of time samples (should be power of 2 for efficiency).
    dt : float
        Time step [s].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with wave and response time series.
    """
    rng = np.random.default_rng(seed)

    # FFT frequencies
    fs = 1.0 / dt
    fft_freq = np.fft.rfftfreq(n_samples, dt)
    n_fft = len(fft_freq)
    df = fs / n_samples

    # Interpolate TF and wave spectrum to FFT frequencies
    tf_mag_interp = tf["magnitude"].interp(frequency=fft_freq, method="linear")
    tf_phase_interp = tf["phase"].interp(frequency=fft_freq, method="linear")
    sxx_interp = wave_spectrum.interp(frequency=fft_freq, method="linear")

    h_mag = tf_mag_interp.fillna(0.0).values  # (n_fft, n_var)
    h_phase = tf_phase_interp.fillna(0.0).values
    sxx = np.maximum(sxx_interp.fillna(0.0).values, 0.0)

    variable_names = tf.coords["variable"].values

    # Wave amplitudes in frequency domain
    a_wave = np.sqrt(sxx * df) * n_samples

    # Random phases
    phi_wave = rng.uniform(0, 2 * np.pi, n_fft)

    # Wave FFT coefficients (complex)
    wave_fft = a_wave * np.exp(1j * phi_wave)
    wave_fft[0] = 0  # zero DC component

    # Inverse FFT for wave
    wave_ts = np.fft.irfft(wave_fft, n=n_samples)

    # Response time series
    responses = {}
    for i, var in enumerate(variable_names):
        h_complex = h_mag[:, i] * np.exp(1j * h_phase[:, i])
        resp_fft = wave_fft * h_complex
        resp_ts = np.fft.irfft(resp_fft, n=n_samples)
        responses[str(var)] = resp_ts

    # Build output
    t = np.arange(n_samples) * dt
    data_vars = {"wave": (["time"], wave_ts)}
    for var, ts in responses.items():
        data_vars[var] = (["time"], ts)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": t},
        attrs={
            "description": "Synthesized time series from FFT method",
            "n_samples": n_samples,
            "dt": dt,
            "seed": seed,
        },
    )

    return ds


def cross_check_coherence(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray,
    coherence_threshold: float = 0.5,
) -> xr.DataArray:
    """Compute reliability of predictions based on coherence.

    Returns a weighted average coherence for each variable, where
    weights are proportional to the response spectral energy at each
    frequency.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction Dataset.
    wave_spectrum : xr.DataArray
        Wave spectrum.
    coherence_threshold : float, optional
        Threshold for "good" coherence, default 0.5.

    Returns
    -------
    xr.DataArray
        Array with weighted coherence for each variable (0 to 1).
        Higher values indicate more reliable predictions.
    """
    resp = response_spectrum(tf, wave_spectrum)
    syy = resp["Syy"].values  # (n_freq, n_var)
    coh = tf["coherence"].values  # (n_freq, n_var)

    # Energy-weighted coherence
    energy_weights = syy / (syy.sum(axis=0, keepdims=True) + 1e-10)
    weighted_coh = (coh * energy_weights).sum(axis=0)

    return xr.DataArray(
        weighted_coh,
        coords={"variable": tf.coords["variable"]},
        dims=["variable"],
        name="weighted_coherence",
        attrs={"coherence_threshold": coherence_threshold},
    )
