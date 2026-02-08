"""Spectral fatigue analysis and damage calculation.

Provides functions to calculate fatigue damage from spectral response analysis
using DNV-RP-C203 S-N curves and standard spectral fatigue methods (narrow-band
Rayleigh and Dirlik).

This module enables fatigue estimation from white noise simulations by combining
transfer functions with wave spectra to compute stress response spectra, then
calculating accumulated damage using Miner's rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.special import gamma as gamma_func

from spectraflex import statistics


# =============================================================================
# S-N Curve Definition
# =============================================================================


@dataclass
class SNCurve:
    """Bilinear S-N curve for fatigue analysis (DNV-RP-C203 format).

    S-N curve with two slopes:
    - For N <= N_transition: log10(N) = log_a1 - m1 * log10(S)
    - For N > N_transition:  log10(N) = log_a2 - m2 * log10(S)

    Parameters
    ----------
    m1 : float
        Slope for N <= 10^7 (typically 3.0 for steel).
    log_a1 : float
        Intercept log10(a1) for first segment.
    m2 : float
        Slope for N > 10^7 (typically 5.0 for steel).
    log_a2 : float
        Intercept log10(a2) for second segment.
    n_transition : float
        Cycle count at slope change, default 10^7.
    name : str
        Curve identifier (e.g., "DNV-D").
    t_ref : float
        Reference thickness for thickness correction [mm].
    k : float
        Thickness exponent for correction.

    Examples
    --------
    >>> curve = SNCurve.dnv_d()
    >>> n_cycles = curve.cycles_to_failure(100.0)  # 100 MPa stress range
    >>> print(f"Cycles to failure: {n_cycles:.0f}")
    """

    m1: float
    log_a1: float
    m2: float
    log_a2: float
    n_transition: float = 1e7
    name: str = ""
    t_ref: float = 25.0
    k: float = 0.0

    def cycles_to_failure(self, stress_range: float | np.ndarray) -> np.ndarray:
        """Calculate cycles to failure from stress range.

        Parameters
        ----------
        stress_range : float or array
            Stress range [MPa].

        Returns
        -------
        np.ndarray
            Number of cycles to failure.
        """
        s = np.asarray(stress_range, dtype=np.float64)
        s = np.maximum(s, 1e-10)  # Avoid log(0)

        # Calculate N for both slopes
        log_n1 = self.log_a1 - self.m1 * np.log10(s)
        log_n2 = self.log_a2 - self.m2 * np.log10(s)

        # Use appropriate slope based on cycle count
        n1 = 10**log_n1
        n2 = 10**log_n2

        # Use first slope where N <= transition, second slope otherwise
        return np.where(n1 <= self.n_transition, n1, n2)

    def stress_at_transition(self) -> float:
        """Calculate stress range at the slope transition point.

        Returns
        -------
        float
            Stress range [MPa] where N = n_transition.
        """
        # From: log10(n_transition) = log_a1 - m1 * log10(S)
        # Solve: log10(S) = (log_a1 - log10(n_transition)) / m1
        log_s = (self.log_a1 - np.log10(self.n_transition)) / self.m1
        return 10**log_s

    def with_scf(self, scf: float) -> SNCurve:
        """Return new curve with stress concentration factor applied.

        The SCF effectively shifts the S-N curve by reducing the
        apparent fatigue strength. Applied as: S_eff = S * SCF

        Parameters
        ----------
        scf : float
            Stress concentration factor (>= 1.0).

        Returns
        -------
        SNCurve
            New curve with SCF applied.
        """
        # SCF multiplies stress, which shifts log_a by m * log10(SCF)
        log_scf = np.log10(scf)
        return SNCurve(
            m1=self.m1,
            log_a1=self.log_a1 - self.m1 * log_scf,
            m2=self.m2,
            log_a2=self.log_a2 - self.m2 * log_scf,
            n_transition=self.n_transition,
            name=f"{self.name} (SCF={scf})",
            t_ref=self.t_ref,
            k=self.k,
        )

    def with_thickness(self, t: float) -> SNCurve:
        """Return new curve with thickness correction applied.

        Thickness correction per DNV-RP-C203:
        S_eff = S * (t / t_ref)^k

        Parameters
        ----------
        t : float
            Actual thickness [mm].

        Returns
        -------
        SNCurve
            New curve with thickness correction applied.
        """
        if self.k == 0 or t <= self.t_ref:
            return self  # No correction needed

        thickness_factor = (t / self.t_ref) ** self.k
        return self.with_scf(thickness_factor)

    # =========================================================================
    # DNV-RP-C203 Standard Curves (In Air)
    # =========================================================================

    @classmethod
    def dnv_b1(cls, in_air: bool = True) -> SNCurve:
        """DNV B1 curve - Ground welds, base material."""
        if in_air:
            return cls(m1=4.0, log_a1=15.117, m2=5.0, log_a2=17.146, k=0.0, name="DNV-B1")
        else:
            return cls(m1=4.0, log_a1=14.917, m2=5.0, log_a2=16.856, k=0.0, name="DNV-B1-SW")

    @classmethod
    def dnv_b2(cls, in_air: bool = True) -> SNCurve:
        """DNV B2 curve - Ground welds."""
        if in_air:
            return cls(m1=4.0, log_a1=14.885, m2=5.0, log_a2=16.856, k=0.0, name="DNV-B2")
        else:
            return cls(m1=4.0, log_a1=14.685, m2=5.0, log_a2=16.566, k=0.0, name="DNV-B2-SW")

    @classmethod
    def dnv_c(cls, in_air: bool = True) -> SNCurve:
        """DNV C curve - Butt welds, automatic."""
        if in_air:
            return cls(m1=3.0, log_a1=12.592, m2=5.0, log_a2=16.320, k=0.15, name="DNV-C")
        else:
            return cls(m1=3.0, log_a1=12.192, m2=5.0, log_a2=15.720, k=0.15, name="DNV-C-SW")

    @classmethod
    def dnv_c1(cls, in_air: bool = True) -> SNCurve:
        """DNV C1 curve - Butt welds."""
        if in_air:
            return cls(m1=3.0, log_a1=12.449, m2=5.0, log_a2=16.081, k=0.15, name="DNV-C1")
        else:
            return cls(m1=3.0, log_a1=12.049, m2=5.0, log_a2=15.481, k=0.15, name="DNV-C1-SW")

    @classmethod
    def dnv_c2(cls, in_air: bool = True) -> SNCurve:
        """DNV C2 curve - Butt welds."""
        if in_air:
            return cls(m1=3.0, log_a1=12.301, m2=5.0, log_a2=15.835, k=0.15, name="DNV-C2")
        else:
            return cls(m1=3.0, log_a1=11.901, m2=5.0, log_a2=15.235, k=0.15, name="DNV-C2-SW")

    @classmethod
    def dnv_d(cls, in_air: bool = True) -> SNCurve:
        """DNV D curve - Fillet welds (most common, default)."""
        if in_air:
            return cls(m1=3.0, log_a1=12.164, m2=5.0, log_a2=15.606, k=0.20, name="DNV-D")
        else:
            return cls(m1=3.0, log_a1=11.764, m2=5.0, log_a2=15.006, k=0.20, name="DNV-D-SW")

    @classmethod
    def dnv_e(cls, in_air: bool = True) -> SNCurve:
        """DNV E curve - Attached plates."""
        if in_air:
            return cls(m1=3.0, log_a1=12.010, m2=5.0, log_a2=15.350, k=0.20, name="DNV-E")
        else:
            return cls(m1=3.0, log_a1=11.610, m2=5.0, log_a2=14.750, k=0.20, name="DNV-E-SW")

    @classmethod
    def dnv_f(cls, in_air: bool = True) -> SNCurve:
        """DNV F curve - Complex joints."""
        if in_air:
            return cls(m1=3.0, log_a1=11.855, m2=5.0, log_a2=15.091, k=0.25, name="DNV-F")
        else:
            return cls(m1=3.0, log_a1=11.455, m2=5.0, log_a2=14.491, k=0.25, name="DNV-F-SW")

    @classmethod
    def dnv_f1(cls, in_air: bool = True) -> SNCurve:
        """DNV F1 curve - Complex joints."""
        if in_air:
            return cls(m1=3.0, log_a1=11.699, m2=5.0, log_a2=14.832, k=0.25, name="DNV-F1")
        else:
            return cls(m1=3.0, log_a1=11.299, m2=5.0, log_a2=14.232, k=0.25, name="DNV-F1-SW")

    @classmethod
    def dnv_f3(cls, in_air: bool = True) -> SNCurve:
        """DNV F3 curve - Complex joints."""
        if in_air:
            return cls(m1=3.0, log_a1=11.546, m2=5.0, log_a2=14.576, k=0.25, name="DNV-F3")
        else:
            return cls(m1=3.0, log_a1=11.146, m2=5.0, log_a2=13.976, k=0.25, name="DNV-F3-SW")

    @classmethod
    def dnv_g(cls, in_air: bool = True) -> SNCurve:
        """DNV G curve - Complex joints."""
        if in_air:
            return cls(m1=3.0, log_a1=11.398, m2=5.0, log_a2=14.330, k=0.25, name="DNV-G")
        else:
            return cls(m1=3.0, log_a1=10.998, m2=5.0, log_a2=13.730, k=0.25, name="DNV-G-SW")

    @classmethod
    def dnv_w1(cls, in_air: bool = True) -> SNCurve:
        """DNV W1 curve - Worst weld quality."""
        if in_air:
            return cls(m1=3.0, log_a1=11.261, m2=5.0, log_a2=14.101, k=0.25, name="DNV-W1")
        else:
            return cls(m1=3.0, log_a1=10.861, m2=5.0, log_a2=13.501, k=0.25, name="DNV-W1-SW")

    @classmethod
    def dnv_w2(cls, in_air: bool = True) -> SNCurve:
        """DNV W2 curve - Worst weld quality."""
        if in_air:
            return cls(m1=3.0, log_a1=11.107, m2=5.0, log_a2=13.845, k=0.25, name="DNV-W2")
        else:
            return cls(m1=3.0, log_a1=10.707, m2=5.0, log_a2=13.245, k=0.25, name="DNV-W2-SW")

    @classmethod
    def dnv_w3(cls, in_air: bool = True) -> SNCurve:
        """DNV W3 curve - Worst weld quality (most conservative)."""
        if in_air:
            return cls(m1=3.0, log_a1=10.970, m2=5.0, log_a2=13.617, k=0.25, name="DNV-W3")
        else:
            return cls(m1=3.0, log_a1=10.570, m2=5.0, log_a2=13.017, k=0.25, name="DNV-W3-SW")


# =============================================================================
# Spectral Parameters
# =============================================================================


def peak_rate(m2: float, m4: float) -> float:
    """Expected number of peaks per second.

    Parameters
    ----------
    m2 : float
        Second spectral moment.
    m4 : float
        Fourth spectral moment.

    Returns
    -------
    float
        Expected peaks per second, E[P] = sqrt(m4/m2).
    """
    if m2 <= 0:
        return 0.0
    return np.sqrt(m4 / m2)


def zero_crossing_rate(m0: float, m2: float) -> float:
    """Expected number of zero-crossings per second.

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment (variance).
    m2 : float
        Second spectral moment.

    Returns
    -------
    float
        Expected zero-crossings per second, E[0] = sqrt(m2/m0).
    """
    if m0 <= 0:
        return 0.0
    return np.sqrt(m2 / m0)


def irregularity_factor(m0: float, m2: float, m4: float) -> float:
    """Irregularity factor (ratio of zero-crossings to peaks).

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment.
    m2 : float
        Second spectral moment.
    m4 : float
        Fourth spectral moment.

    Returns
    -------
    float
        Irregularity factor gamma = m2 / sqrt(m0 * m4).
        gamma = 1 for narrow-band, gamma < 1 for wide-band.
    """
    denom = np.sqrt(m0 * m4)
    if denom <= 0:
        return 0.0
    return m2 / denom


# =============================================================================
# Narrow-Band (Rayleigh) Fatigue Damage
# =============================================================================


def narrow_band_damage(
    m0: float,
    sn_curve: SNCurve,
    exposure_time: float,
    m2: float | None = None,
    nu_0: float | None = None,
) -> float:
    """Calculate narrow-band fatigue damage using Rayleigh distribution.

    For a narrow-band Gaussian process, stress ranges follow a Rayleigh
    distribution. This gives a closed-form solution for damage.

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment of stress PSD [MPa^2].
    sn_curve : SNCurve
        S-N curve for damage calculation.
    exposure_time : float
        Exposure time [seconds].
    m2 : float, optional
        Second spectral moment. Required if nu_0 not provided.
    nu_0 : float, optional
        Zero-crossing rate [Hz]. If not provided, calculated from m0, m2.

    Returns
    -------
    float
        Miner's sum damage ratio. D >= 1 indicates failure.

    Notes
    -----
    Damage formula (single slope):
        D = (nu_0 * T) * (2*sqrt(2*m0))^m * Gamma(1 + m/2) / a

    For bilinear curves, this uses the primary slope (m1) which is
    conservative for high-cycle fatigue.
    """
    if m0 <= 0:
        return 0.0

    # Get zero-crossing rate
    if nu_0 is None:
        if m2 is None:
            raise ValueError("Either m2 or nu_0 must be provided")
        nu_0 = zero_crossing_rate(m0, m2)

    # Use primary slope (conservative)
    m = sn_curve.m1
    a = 10**sn_curve.log_a1

    # Rayleigh damage formula
    # E[S^m] for Rayleigh distribution = (2*sqrt(2*m0))^m * Gamma(1 + m/2)
    rms = np.sqrt(m0)
    expected_s_m = (2 * np.sqrt(2) * rms) ** m * gamma_func(1 + m / 2)

    # Total cycles
    n_cycles = nu_0 * exposure_time

    # Damage = sum(n_i / N_i) = n_total * E[1/N] = n_total * E[S^m] / a
    damage = n_cycles * expected_s_m / a

    return float(damage)


# =============================================================================
# Dirlik Spectral Fatigue
# =============================================================================


def dirlik_coefficients(
    m0: float, m1: float, m2: float, m4: float
) -> dict[str, float]:
    """Calculate Dirlik distribution coefficients.

    Parameters
    ----------
    m0, m1, m2, m4 : float
        Spectral moments.

    Returns
    -------
    dict
        Coefficients D1, D2, D3, Q, R, and auxiliary values.
    """
    # Auxiliary parameters
    xm = m1 / m0 * np.sqrt(m2 / m4)
    gamma = m2 / np.sqrt(m0 * m4)

    # Dirlik coefficients
    d1 = 2 * (xm - gamma**2) / (1 + gamma**2)

    denom = 1 - gamma - d1 + d1**2
    if abs(denom) < 1e-10:
        denom = 1e-10

    r = (gamma - xm - d1**2) / denom
    d2 = (1 - gamma - d1 + d1**2) / (1 - r) if abs(1 - r) > 1e-10 else 0.0
    d3 = 1 - d1 - d2

    q_denom = d1 if abs(d1) > 1e-10 else 1e-10
    q = 1.25 * (gamma - d3 - d2 * r) / q_denom

    return {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "Q": q,
        "R": r,
        "xm": xm,
        "gamma": gamma,
    }


def dirlik_pdf(
    stress_range: np.ndarray,
    m0: float,
    m1: float,
    m2: float,
    m4: float,
) -> np.ndarray:
    """Dirlik probability density function for stress ranges.

    Parameters
    ----------
    stress_range : array
        Stress range values [MPa].
    m0, m1, m2, m4 : float
        Spectral moments.

    Returns
    -------
    np.ndarray
        Probability density at each stress range value.

    Notes
    -----
    The Dirlik PDF is:
        p(S) = (D1/Q * exp(-Z/Q) + D2*Z/R^2 * exp(-Z^2/(2R^2)) + D3*Z * exp(-Z^2/2)) / (2*sqrt(m0))

    where Z = S / (2*sqrt(m0))
    """
    s = np.asarray(stress_range, dtype=np.float64)
    rms = np.sqrt(m0)

    coef = dirlik_coefficients(m0, m1, m2, m4)
    d1, d2, d3 = coef["D1"], coef["D2"], coef["D3"]
    q, r = coef["Q"], coef["R"]

    # Normalized stress range
    z = s / (2 * rms)

    # Three components of Dirlik PDF
    # Term 1: Exponential (small cycles)
    if abs(q) > 1e-10:
        term1 = (d1 / q) * np.exp(-z / q)
    else:
        term1 = np.zeros_like(z)

    # Term 2: Rayleigh-like (medium cycles)
    if abs(r) > 1e-10:
        term2 = (d2 * z / r**2) * np.exp(-(z**2) / (2 * r**2))
    else:
        term2 = np.zeros_like(z)

    # Term 3: Rayleigh (large cycles)
    term3 = d3 * z * np.exp(-(z**2) / 2)

    # Combined PDF
    pdf = (term1 + term2 + term3) / (2 * rms)

    return np.maximum(pdf, 0.0)


def dirlik_damage(
    m0: float,
    m1: float,
    m2: float,
    m4: float,
    sn_curve: SNCurve,
    exposure_time: float,
) -> float:
    """Calculate Dirlik spectral fatigue damage.

    Parameters
    ----------
    m0, m1, m2, m4 : float
        Spectral moments of stress PSD.
    sn_curve : SNCurve
        S-N curve for damage calculation.
    exposure_time : float
        Exposure time [seconds].

    Returns
    -------
    float
        Miner's sum damage ratio. D >= 1 indicates failure.

    Notes
    -----
    Dirlik damage for single-slope S-N curve:
        D = (nu_p * T / a) * (2*sqrt(m0))^m * integral

    where:
        integral = D1*Q^m*Gamma(1+m) + sqrt(2)^m*Gamma(1+m/2)*(D2*|R|^m + D3)
    """
    if m0 <= 0:
        return 0.0

    coef = dirlik_coefficients(m0, m1, m2, m4)
    d1, d2, d3 = coef["D1"], coef["D2"], coef["D3"]
    q, r = coef["Q"], coef["R"]

    # Peak rate
    nu_p = peak_rate(m2, m4)

    # Use primary slope
    m = sn_curve.m1
    a = 10**sn_curve.log_a1

    # Closed-form integral of S^m * p(S) for Dirlik distribution
    rms = np.sqrt(m0)

    # E[S^m] for Dirlik
    # = (2*sqrt(m0))^m * [D1*Q^m*Gamma(1+m) + sqrt(2)^m*Gamma(1+m/2)*(D2*|R|^m + D3)]
    term1 = d1 * (abs(q) ** m) * gamma_func(1 + m) if abs(q) > 1e-10 else 0.0
    term2 = (np.sqrt(2) ** m) * gamma_func(1 + m / 2) * (d2 * (abs(r) ** m) + d3)

    expected_s_m = (2 * rms) ** m * (term1 + term2)

    # Total cycles
    n_cycles = nu_p * exposure_time

    # Damage
    damage = n_cycles * expected_s_m / a

    return float(damage)


# =============================================================================
# High-Level Functions
# =============================================================================


def damage_from_spectrum(
    frequency: np.ndarray | xr.DataArray,
    stress_psd: np.ndarray | xr.DataArray,
    sn_curve: SNCurve,
    exposure_time: float,
    method: str = "dirlik",
) -> dict[str, float]:
    """Calculate fatigue damage from stress response spectrum.

    Parameters
    ----------
    frequency : array
        Frequency values [Hz].
    stress_psd : array
        One-sided stress power spectral density [MPa^2/Hz].
    sn_curve : SNCurve
        S-N curve for damage calculation.
    exposure_time : float
        Exposure time [seconds].
    method : {"dirlik", "narrow_band"}
        Damage calculation method. Dirlik is more accurate for wide-band
        spectra, narrow_band is simpler but assumes narrow bandwidth.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - damage: Miner's sum damage ratio (D >= 1 indicates failure)
        - life_seconds: Estimated fatigue life [seconds]
        - n_cycles: Expected number of stress cycles
        - damage_rate: Damage per second [1/s]
        - bandwidth: Spectral bandwidth parameter epsilon
    """
    f = np.asarray(frequency, dtype=np.float64)
    s = np.asarray(stress_psd, dtype=np.float64)

    # Calculate spectral moments
    moments = statistics.spectral_moments(f, s, orders=(0, 1, 2, 4))
    m0 = moments[0]
    m1 = moments[1]
    m2 = moments[2]
    m4 = moments[4]

    # Bandwidth parameter
    bandwidth = statistics.bandwidth_parameter(m0, m2, m4)

    # Calculate damage
    if method == "dirlik":
        damage = dirlik_damage(m0, m1, m2, m4, sn_curve, exposure_time)
        nu = peak_rate(m2, m4)
    elif method == "narrow_band":
        damage = narrow_band_damage(m0, sn_curve, exposure_time, m2=m2)
        nu = zero_crossing_rate(m0, m2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dirlik' or 'narrow_band'.")

    # Derived quantities
    n_cycles = nu * exposure_time
    damage_rate = damage / exposure_time if exposure_time > 0 else 0.0
    life_seconds = exposure_time / damage if damage > 0 else np.inf

    return {
        "damage": damage,
        "life_seconds": life_seconds,
        "n_cycles": n_cycles,
        "damage_rate": damage_rate,
        "bandwidth": bandwidth,
    }


def damage_from_transfer_function(
    tf: xr.Dataset,
    wave_spectrum: xr.DataArray | np.ndarray,
    sn_curve: SNCurve,
    exposure_time: float,
    method: str = "dirlik",
    variable: str | None = None,
    wave_frequency: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate fatigue damage from transfer function and wave spectrum.

    Combines H(f) with S_wave(f) to get S_stress(f), then calculates damage.

    Parameters
    ----------
    tf : xr.Dataset
        TransferFunction dataset. Must be in stress units (MPa/m).
    wave_spectrum : xr.DataArray or array
        Wave elevation spectrum S_wave(f) [m^2/Hz].
    sn_curve : SNCurve
        S-N curve for damage calculation.
    exposure_time : float
        Exposure time [seconds].
    method : {"dirlik", "narrow_band"}
        Damage calculation method.
    variable : str, optional
        Variable name to analyze. If None and single variable, uses that.
    wave_frequency : array, optional
        Frequency values for wave_spectrum if it's a plain array.

    Returns
    -------
    dict
        Dictionary with damage results and intermediate values:
        - damage: Miner's sum damage ratio
        - life_seconds: Estimated fatigue life
        - n_cycles: Expected number of cycles
        - damage_rate: Damage per second
        - bandwidth: Spectral bandwidth parameter
        - stress_m0: Zeroth moment of stress spectrum [MPa^2]
        - stress_rms: RMS stress [MPa]
    """
    # Get transfer function frequency and magnitude
    tf_freq = tf.coords["frequency"].values

    # Select variable
    variables = list(tf.coords["variable"].values)
    if variable is None:
        if len(variables) == 1:
            variable = variables[0]
        else:
            raise ValueError(
                f"Multiple variables present: {variables}. Specify 'variable' parameter."
            )

    h_mag = tf["magnitude"].sel(variable=variable).values

    # Get wave spectrum at transfer function frequencies
    if isinstance(wave_spectrum, xr.DataArray):
        # Interpolate to TF frequencies
        s_wave = wave_spectrum.interp(frequency=tf_freq, method="linear").values
    else:
        s_wave = np.asarray(wave_spectrum, dtype=np.float64)
        if wave_frequency is not None:
            # Interpolate
            s_wave = np.interp(tf_freq, wave_frequency, s_wave)

    # Ensure non-negative
    s_wave = np.maximum(s_wave, 0.0)

    # Response spectrum: S_stress(f) = |H(f)|^2 * S_wave(f)
    s_stress = h_mag**2 * s_wave

    # Calculate damage
    result = damage_from_spectrum(
        frequency=tf_freq,
        stress_psd=s_stress,
        sn_curve=sn_curve,
        exposure_time=exposure_time,
        method=method,
    )

    # Add stress spectrum info
    m0 = statistics.m0(tf_freq, s_stress)
    result["stress_m0"] = m0
    result["stress_rms"] = np.sqrt(m0)

    return result
