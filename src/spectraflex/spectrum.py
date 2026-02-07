"""Wave spectrum definitions.

Provides functions to create standard wave spectra (JONSWAP, Pierson-Moskowitz)
and user-defined spectra for use with spectraflex.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def jonswap(
    hs: float,
    tp: float,
    f: np.ndarray,
    gamma: float = 3.3,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
) -> xr.DataArray:
    """Create a JONSWAP wave spectrum.

    The JONSWAP (Joint North Sea Wave Project) spectrum is an extension of the
    Pierson-Moskowitz spectrum for developing seas with a sharper peak.

    Parameters
    ----------
    hs : float
        Significant wave height [m].
    tp : float
        Peak period [s].
    f : np.ndarray
        Frequency array [Hz].
    gamma : float, optional
        Peak enhancement factor, default 3.3.
    sigma_a : float, optional
        Spectral width parameter for f < fp, default 0.07.
    sigma_b : float, optional
        Spectral width parameter for f > fp, default 0.09.

    Returns
    -------
    xr.DataArray
        Wave spectrum S(f) [m²/Hz] as a DataArray with frequency coordinate.

    Notes
    -----
    The JONSWAP spectrum is defined as:

        S(f) = α * g² / (2π)⁴ / f⁵ * exp(-5/4 * (fp/f)⁴) * γ^r

    where:
        - α is the Phillips constant (derived from Hs)
        - fp = 1/Tp is the peak frequency
        - r = exp(-(f - fp)² / (2 * σ² * fp²))
        - σ = σ_a for f < fp, σ = σ_b for f > fp

    The implementation normalizes to ensure ∫S(f)df = (Hs/4)².
    """
    f = np.asarray(f, dtype=np.float64)
    fp = 1.0 / tp
    g = 9.80665  # standard gravity [m/s²]

    # Avoid division by zero
    f_safe = np.where(f > 0, f, np.inf)

    # Pierson-Moskowitz base spectrum (unnormalized)
    pm = (g**2 / (2 * np.pi) ** 4 / f_safe**5) * np.exp(-1.25 * (fp / f_safe) ** 4)

    # JONSWAP peak enhancement
    sigma = np.where(f < fp, sigma_a, sigma_b)
    r = np.exp(-((f - fp) ** 2) / (2 * sigma**2 * fp**2))
    enhancement = gamma**r

    # Unnormalized JONSWAP
    s_unnorm = pm * enhancement

    # Normalize to target m0 = (Hs/4)²
    m0_target = (hs / 4.0) ** 2
    m0_current = np.trapezoid(s_unnorm, f)

    if m0_current > 0:
        s = s_unnorm * (m0_target / m0_current)
    else:
        s = np.zeros_like(f)

    # Handle f=0 case
    s = np.where(f > 0, s, 0.0)

    return xr.DataArray(
        s,
        coords={"frequency": f},
        dims=["frequency"],
        name="S",
        attrs={
            "units": "m^2/Hz",
            "long_name": "JONSWAP wave spectrum",
            "hs": hs,
            "tp": tp,
            "gamma": gamma,
        },
    )


def pierson_moskowitz(hs: float, tp: float, f: np.ndarray) -> xr.DataArray:
    """Create a Pierson-Moskowitz wave spectrum.

    The Pierson-Moskowitz spectrum represents a fully developed sea state.

    Parameters
    ----------
    hs : float
        Significant wave height [m].
    tp : float
        Peak period [s].
    f : np.ndarray
        Frequency array [Hz].

    Returns
    -------
    xr.DataArray
        Wave spectrum S(f) [m²/Hz] as a DataArray with frequency coordinate.

    Notes
    -----
    The Pierson-Moskowitz spectrum is defined as:

        S(f) = α * g² / (2π)⁴ / f⁵ * exp(-5/4 * (fp/f)⁴)

    where α is chosen to give the specified Hs via m0 = (Hs/4)².
    This is equivalent to JONSWAP with γ = 1.
    """
    # PM is JONSWAP with gamma=1 (no peak enhancement)
    da = jonswap(hs=hs, tp=tp, f=f, gamma=1.0)
    da.attrs["long_name"] = "Pierson-Moskowitz wave spectrum"
    return da


def from_array(
    f: np.ndarray,
    s: np.ndarray,
    name: str = "S",
    attrs: dict | None = None,
) -> xr.DataArray:
    """Create a wave spectrum DataArray from arrays.

    Parameters
    ----------
    f : np.ndarray
        Frequency array [Hz].
    s : np.ndarray
        Spectral density array [m²/Hz].
    name : str, optional
        Name for the DataArray, default "S".
    attrs : dict, optional
        Additional attributes.

    Returns
    -------
    xr.DataArray
        Wave spectrum as a DataArray with frequency coordinate.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes or invalid values.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if f.shape != s.shape:
        raise ValueError(f"f shape {f.shape} does not match s shape {s.shape}")

    if not np.all(f > 0):
        raise ValueError("frequency values must be positive")

    if not np.all(np.diff(f) > 0):
        raise ValueError("frequency values must be monotonically increasing")

    if not np.all(s >= 0):
        raise ValueError("spectral density values must be non-negative")

    da_attrs = {"units": "m^2/Hz", "long_name": "wave spectrum"}
    if attrs:
        da_attrs.update(attrs)

    return xr.DataArray(
        s,
        coords={"frequency": f},
        dims=["frequency"],
        name=name,
        attrs=da_attrs,
    )


def white_noise(
    hs: float,
    f: np.ndarray,
    f_min: float | None = None,
    f_max: float | None = None,
) -> xr.DataArray:
    """Create a white noise (flat) wave spectrum.

    This spectrum has constant spectral density across the specified frequency
    range, used for transfer function identification in OrcaFlex.

    Parameters
    ----------
    hs : float
        Significant wave height [m]. The spectrum is scaled so that
        m0 = (Hs/4)² within the frequency range.
    f : np.ndarray
        Frequency array [Hz].
    f_min : float, optional
        Minimum frequency for the flat region. Default is min(f).
    f_max : float, optional
        Maximum frequency for the flat region. Default is max(f).

    Returns
    -------
    xr.DataArray
        White noise spectrum as a DataArray with frequency coordinate.
    """
    f = np.asarray(f, dtype=np.float64)

    if f_min is None:
        f_min = float(f.min())
    if f_max is None:
        f_max = float(f.max())

    # Create flat spectrum in range
    m0_target = (hs / 4.0) ** 2
    bandwidth = f_max - f_min

    if bandwidth <= 0:
        raise ValueError("f_max must be greater than f_min")

    # Constant spectral density to achieve target m0
    s_flat = m0_target / bandwidth

    # Apply to frequency range
    s = np.where((f >= f_min) & (f <= f_max), s_flat, 0.0)

    return xr.DataArray(
        s,
        coords={"frequency": f},
        dims=["frequency"],
        name="S",
        attrs={
            "units": "m^2/Hz",
            "long_name": "white noise wave spectrum",
            "hs": hs,
            "f_min": f_min,
            "f_max": f_max,
        },
    )


def scale_to_hs(spectrum: xr.DataArray, hs: float) -> xr.DataArray:
    """Scale a spectrum to a target significant wave height.

    Parameters
    ----------
    spectrum : xr.DataArray
        Input wave spectrum with frequency coordinate.
    hs : float
        Target significant wave height [m].

    Returns
    -------
    xr.DataArray
        Scaled spectrum with the specified Hs.
    """
    f = spectrum.coords["frequency"].values
    s = spectrum.values

    m0_current = np.trapezoid(s, f)
    m0_target = (hs / 4.0) ** 2

    if m0_current > 0:
        scale = m0_target / m0_current
    else:
        scale = 0.0

    scaled = spectrum * scale
    scaled.attrs = spectrum.attrs.copy()
    scaled.attrs["hs"] = hs

    return scaled


def frequency_array(
    f_min: float = 0.01,
    f_max: float = 0.5,
    n_freq: int = 256,
    spacing: str = "linear",
) -> np.ndarray:
    """Create a frequency array for spectrum calculations.

    Parameters
    ----------
    f_min : float, optional
        Minimum frequency [Hz], default 0.01.
    f_max : float, optional
        Maximum frequency [Hz], default 0.5.
    n_freq : int, optional
        Number of frequency points, default 256.
    spacing : str, optional
        "linear" for linear spacing, "log" for logarithmic spacing.
        Default "linear".

    Returns
    -------
    np.ndarray
        Frequency array [Hz].
    """
    if spacing == "linear":
        return np.linspace(f_min, f_max, n_freq)
    elif spacing == "log":
        return np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
    else:
        raise ValueError(f"Unknown spacing: {spacing}. Use 'linear' or 'log'.")
