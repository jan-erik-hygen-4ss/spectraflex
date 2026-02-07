"""Spectral statistics calculations.

Provides functions to compute spectral moments, significant heights,
peak periods, and expected extreme values from power spectra.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def spectral_moments(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
    orders: tuple[int, ...] = (0, 1, 2, 4),
) -> dict[int, float]:
    """Compute spectral moments of a power spectrum.

    The n-th spectral moment is defined as:
        m_n = ∫ f^n * S(f) df

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array [units²/Hz].
    orders : tuple of int, optional
        Which moments to compute, default (0, 1, 2, 4).

    Returns
    -------
    dict[int, float]
        Dictionary mapping moment order to value.

    Examples
    --------
    >>> moments = spectral_moments(f, s)
    >>> m0, m2 = moments[0], moments[2]
    """
    f_arr = np.asarray(f, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)

    if f_arr.shape != s_arr.shape:
        raise ValueError(f"f shape {f_arr.shape} does not match s shape {s_arr.shape}")

    moments = {}
    for n in orders:
        integrand = (f_arr**n) * s_arr
        moments[n] = float(np.trapezoid(integrand, f_arr))

    return moments


def m0(f: np.ndarray | xr.DataArray, s: np.ndarray | xr.DataArray) -> float:
    """Compute zeroth spectral moment (variance).

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.

    Returns
    -------
    float
        Zeroth moment m0 = ∫S(f)df.
    """
    return spectral_moments(f, s, orders=(0,))[0]


def m2(f: np.ndarray | xr.DataArray, s: np.ndarray | xr.DataArray) -> float:
    """Compute second spectral moment.

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.

    Returns
    -------
    float
        Second moment m2 = ∫f²S(f)df.
    """
    return spectral_moments(f, s, orders=(2,))[2]


def m4(f: np.ndarray | xr.DataArray, s: np.ndarray | xr.DataArray) -> float:
    """Compute fourth spectral moment.

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.

    Returns
    -------
    float
        Fourth moment m4 = ∫f⁴S(f)df.
    """
    return spectral_moments(f, s, orders=(4,))[4]


def hs_from_spectrum(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
) -> float:
    """Compute significant wave height from a spectrum.

    Hs = 4 * sqrt(m0)

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array [m²/Hz].

    Returns
    -------
    float
        Significant wave height [m].
    """
    m0_val = m0(f, s)
    return 4.0 * np.sqrt(m0_val)


def hs_from_m0(m0_val: float) -> float:
    """Compute significant height from zeroth moment.

    Hs = 4 * sqrt(m0)

    This works for any response variable, not just waves.

    Parameters
    ----------
    m0_val : float
        Zeroth spectral moment (variance).

    Returns
    -------
    float
        Significant height Hs = 4*sqrt(m0).
    """
    return 4.0 * np.sqrt(m0_val)


def tp_from_spectrum(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
) -> float:
    """Compute peak period from a spectrum.

    Tp = 1 / fp, where fp is the frequency at maximum spectral density.

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.

    Returns
    -------
    float
        Peak period [s].
    """
    f_arr = np.asarray(f, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)

    idx = np.argmax(s_arr)
    fp = f_arr[idx]

    if fp <= 0:
        return np.inf

    return 1.0 / fp


def tz_from_moments(m0_val: float, m2_val: float) -> float:
    """Compute zero-crossing period from spectral moments.

    Tz = sqrt(m0 / m2)

    Parameters
    ----------
    m0_val : float
        Zeroth spectral moment.
    m2_val : float
        Second spectral moment.

    Returns
    -------
    float
        Zero-crossing period [s].
    """
    if m2_val <= 0:
        return np.inf
    return np.sqrt(m0_val / m2_val)


def tz_from_spectrum(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
) -> float:
    """Compute zero-crossing period from a spectrum.

    Tz = sqrt(m0 / m2)

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.

    Returns
    -------
    float
        Zero-crossing period [s].
    """
    moments = spectral_moments(f, s, orders=(0, 2))
    return tz_from_moments(moments[0], moments[2])


def bandwidth_parameter(m0_val: float, m2_val: float, m4_val: float) -> float:
    """Compute spectral bandwidth parameter epsilon.

    ε = sqrt(1 - m2² / (m0 * m4))

    This measures how narrow-banded the spectrum is:
    - ε → 0: narrow-banded (regular, nearly sinusoidal)
    - ε → 1: broad-banded (irregular)

    Parameters
    ----------
    m0_val : float
        Zeroth spectral moment.
    m2_val : float
        Second spectral moment.
    m4_val : float
        Fourth spectral moment.

    Returns
    -------
    float
        Bandwidth parameter ε in [0, 1].
    """
    if m0_val <= 0 or m4_val <= 0:
        return 1.0

    eps_sq = 1.0 - (m2_val**2) / (m0_val * m4_val)
    # Numerical issues can make this slightly negative
    eps_sq = max(0.0, eps_sq)
    return np.sqrt(eps_sq)


def mpm_rayleigh(
    m0_val: float,
    duration: float,
    m2_val: float | None = None,
    tz: float | None = None,
) -> float:
    """Compute most probable maximum assuming Rayleigh distribution.

    MPM = σ * sqrt(2 * ln(N))

    where σ = sqrt(m0) and N = duration / Tz is the number of cycles.

    Parameters
    ----------
    m0_val : float
        Zeroth spectral moment (variance).
    duration : float
        Duration of the sea state [s].
    m2_val : float, optional
        Second spectral moment. Required if tz is not provided.
    tz : float, optional
        Zero-crossing period [s]. If not provided, computed from m0/m2.

    Returns
    -------
    float
        Most probable maximum value.

    Notes
    -----
    For a 3-hour storm: duration = 3 * 3600 = 10800 s
    """
    if tz is None:
        if m2_val is None:
            raise ValueError("Either m2_val or tz must be provided")
        tz = tz_from_moments(m0_val, m2_val)

    if tz <= 0:
        return np.inf

    sigma = np.sqrt(m0_val)
    n_cycles = duration / tz

    if n_cycles <= 1:
        return sigma

    return sigma * np.sqrt(2.0 * np.log(n_cycles))


def mpm_from_spectrum(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
    duration: float,
) -> float:
    """Compute most probable maximum from a spectrum.

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.
    duration : float
        Duration [s].

    Returns
    -------
    float
        Most probable maximum value.
    """
    moments = spectral_moments(f, s, orders=(0, 2))
    return mpm_rayleigh(moments[0], duration, m2_val=moments[2])


def extreme_rayleigh(
    m0_val: float,
    duration: float,
    m2_val: float | None = None,
    tz: float | None = None,
    probability: float = 0.5,
) -> float:
    """Compute extreme value with given exceedance probability.

    For a Rayleigh process, the extreme value exceeded with probability p is:

        x_p = σ * sqrt(2 * ln(N / (-ln(1-p))))

    When p = 0.5, this gives the median maximum (MPM).
    When p = 0.368 (= 1/e), this gives the expected maximum.

    Parameters
    ----------
    m0_val : float
        Zeroth spectral moment (variance).
    duration : float
        Duration [s].
    m2_val : float, optional
        Second spectral moment.
    tz : float, optional
        Zero-crossing period [s].
    probability : float, optional
        Exceedance probability, default 0.5 (MPM).

    Returns
    -------
    float
        Extreme value.
    """
    if tz is None:
        if m2_val is None:
            raise ValueError("Either m2_val or tz must be provided")
        tz = tz_from_moments(m0_val, m2_val)

    if tz <= 0:
        return np.inf

    sigma = np.sqrt(m0_val)
    n_cycles = duration / tz

    if n_cycles <= 1:
        return sigma

    if probability <= 0 or probability >= 1:
        raise ValueError("probability must be in (0, 1)")

    # CDF of maximum of N Rayleigh variables: F(x) = (1 - exp(-x²/(2σ²)))^N
    # Solving for x given F(x) = 1-p:
    # 1-p = (1 - exp(-x²/(2σ²)))^N
    # (1-p)^(1/N) = 1 - exp(-x²/(2σ²))
    # exp(-x²/(2σ²)) = 1 - (1-p)^(1/N)
    # x²/(2σ²) = -ln(1 - (1-p)^(1/N))
    # x = σ * sqrt(2 * (-ln(1 - (1-p)^(1/N))))

    inner = 1.0 - (1.0 - probability) ** (1.0 / n_cycles)
    if inner <= 0:
        return np.inf

    return sigma * np.sqrt(-2.0 * np.log(inner))


def all_statistics(
    f: np.ndarray | xr.DataArray,
    s: np.ndarray | xr.DataArray,
    duration: float = 10800.0,
) -> dict[str, float]:
    """Compute comprehensive statistics from a spectrum.

    Parameters
    ----------
    f : np.ndarray or xr.DataArray
        Frequency array [Hz].
    s : np.ndarray or xr.DataArray
        Spectral density array.
    duration : float, optional
        Duration for extreme calculations [s], default 10800 (3 hours).

    Returns
    -------
    dict
        Dictionary with keys:
        - m0, m1, m2, m4: spectral moments
        - hs: significant height (4*sqrt(m0))
        - tp: peak period
        - tz: zero-crossing period
        - epsilon: bandwidth parameter
        - mpm: most probable maximum
        - sigma: standard deviation (sqrt(m0))
    """
    moments = spectral_moments(f, s, orders=(0, 1, 2, 4))
    m0_val = moments[0]
    m2_val = moments[2]
    m4_val = moments[4]

    return {
        "m0": m0_val,
        "m1": moments[1],
        "m2": m2_val,
        "m4": m4_val,
        "hs": hs_from_m0(m0_val),
        "tp": tp_from_spectrum(f, s),
        "tz": tz_from_moments(m0_val, m2_val),
        "epsilon": bandwidth_parameter(m0_val, m2_val, m4_val),
        "mpm": mpm_rayleigh(m0_val, duration, m2_val=m2_val),
        "sigma": np.sqrt(m0_val),
    }
