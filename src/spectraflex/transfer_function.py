"""TransferFunction xarray Dataset factory and validation.

A TransferFunction is an xarray.Dataset containing identified transfer function(s)
for one operating configuration. This module provides factory functions to create
validated Datasets and standalone functions to operate on them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr

# Required data variables in a TransferFunction Dataset
REQUIRED_DATA_VARS = frozenset({"magnitude", "phase", "coherence"})

# Required coordinates
REQUIRED_COORDS = frozenset({"frequency", "variable"})

# Required attributes
REQUIRED_ATTRS = frozenset({"created"})


def create(
    frequency: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    coherence: np.ndarray,
    variable_names: list[str],
    sxx: np.ndarray | None = None,
    syy: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    **attrs: Any,
) -> xr.Dataset:
    """Create a validated TransferFunction Dataset.

    Parameters
    ----------
    frequency : np.ndarray
        1D array of frequency values in Hz, shape (n_freq,).
    magnitude : np.ndarray
        Transfer function magnitude |H(f)| in response_units / m.
        Shape (n_freq,) for single variable or (n_freq, n_var) for multiple.
    phase : np.ndarray
        Transfer function phase arg(H(f)) in radians.
        Shape (n_freq,) for single variable or (n_freq, n_var) for multiple.
    coherence : np.ndarray
        Coherence γ²(f), values in [0, 1].
        Shape (n_freq,) for single variable or (n_freq, n_var) for multiple.
    variable_names : list[str]
        Names of the response variables.
    sxx : np.ndarray, optional
        Input auto-spectrum, shape (n_freq,). If None, not included.
    syy : np.ndarray, optional
        Output auto-spectrum, shape (n_freq,) or (n_freq, n_var). If None, not included.
    config : dict, optional
        Configuration parameters (e.g., {"hs": 2.0, "draft": 21.0, "heading": 0.0}).
    **attrs
        Additional attributes to store in the Dataset.

    Returns
    -------
    xr.Dataset
        A validated TransferFunction Dataset.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid values.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    coherence = np.asarray(coherence, dtype=np.float64)

    n_freq = len(frequency)
    n_var = len(variable_names)

    # Handle 1D arrays for single variable case
    if magnitude.ndim == 1:
        magnitude = magnitude[:, np.newaxis]
    if phase.ndim == 1:
        phase = phase[:, np.newaxis]
    if coherence.ndim == 1:
        coherence = coherence[:, np.newaxis]

    # Validate shapes
    if magnitude.shape != (n_freq, n_var):
        raise ValueError(
            f"magnitude shape {magnitude.shape} does not match "
            f"expected ({n_freq}, {n_var})"
        )
    if phase.shape != (n_freq, n_var):
        raise ValueError(
            f"phase shape {phase.shape} does not match expected ({n_freq}, {n_var})"
        )
    if coherence.shape != (n_freq, n_var):
        raise ValueError(
            f"coherence shape {coherence.shape} does not match "
            f"expected ({n_freq}, {n_var})"
        )

    # Validate frequency is positive and monotonically increasing
    if not np.all(frequency > 0):
        raise ValueError("frequency values must be positive")
    if not np.all(np.diff(frequency) > 0):
        raise ValueError("frequency values must be monotonically increasing")

    # Validate coherence is in [0, 1]
    if not np.all((coherence >= 0) & (coherence <= 1)):
        raise ValueError("coherence values must be in [0, 1]")

    # Validate magnitude is non-negative
    if not np.all(magnitude >= 0):
        raise ValueError("magnitude values must be non-negative")

    # Build data variables
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "magnitude": (("frequency", "variable"), magnitude),
        "phase": (("frequency", "variable"), phase),
        "coherence": (("frequency", "variable"), coherence),
    }

    # Add optional spectra
    if sxx is not None:
        sxx = np.asarray(sxx, dtype=np.float64)
        if sxx.shape != (n_freq,):
            raise ValueError(
                f"sxx shape {sxx.shape} does not match expected ({n_freq},)"
            )
        data_vars["Sxx"] = (("frequency",), sxx)

    if syy is not None:
        syy = np.asarray(syy, dtype=np.float64)
        if syy.ndim == 1:
            syy = syy[:, np.newaxis]
        if syy.shape != (n_freq, n_var):
            raise ValueError(
                f"syy shape {syy.shape} does not match expected ({n_freq}, {n_var})"
            )
        data_vars["Syy"] = (("frequency", "variable"), syy)

    # Build coordinates
    coords = {
        "frequency": frequency,
        "variable": variable_names,
    }

    # Build attributes
    ds_attrs: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if config is not None:
        ds_attrs["config"] = config
    ds_attrs.update(attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)

    return ds


def validate(ds: xr.Dataset) -> None:
    """Validate that a Dataset conforms to the TransferFunction schema.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Raises
    ------
    ValueError
        If the Dataset doesn't conform to the TransferFunction schema.
    """
    # Check required coordinates
    missing_coords = REQUIRED_COORDS - set(ds.coords)
    if missing_coords:
        raise ValueError(f"Missing required coordinates: {missing_coords}")

    # Check required data variables
    missing_vars = REQUIRED_DATA_VARS - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"Missing required data variables: {missing_vars}")

    # Check required attributes
    missing_attrs = REQUIRED_ATTRS - set(ds.attrs)
    if missing_attrs:
        raise ValueError(f"Missing required attributes: {missing_attrs}")

    # Validate dimensions of data variables
    expected_dims = ("frequency", "variable")
    for var in REQUIRED_DATA_VARS:
        if ds[var].dims != expected_dims:
            raise ValueError(
                f"Data variable '{var}' has dims {ds[var].dims}, "
                f"expected {expected_dims}"
            )

    # Validate frequency values
    freq = ds.coords["frequency"].values
    if not np.all(freq > 0):
        raise ValueError("frequency values must be positive")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("frequency values must be monotonically increasing")

    # Validate coherence range
    coh = ds["coherence"].values
    if not np.all((coh >= 0) & (coh <= 1)):
        raise ValueError("coherence values must be in [0, 1]")

    # Validate magnitude is non-negative
    mag = ds["magnitude"].values
    if not np.all(mag >= 0):
        raise ValueError("magnitude values must be non-negative")


def is_valid(ds: xr.Dataset) -> bool:
    """Check if a Dataset is a valid TransferFunction.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if the Dataset is a valid TransferFunction, False otherwise.
    """
    try:
        validate(ds)
        return True
    except ValueError:
        return False


def complex_transfer_function(ds: xr.Dataset) -> xr.DataArray:
    """Get the complex-valued transfer function H(f) from a TransferFunction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A valid TransferFunction Dataset.

    Returns
    -------
    xr.DataArray
        Complex transfer function H(f) = |H| * exp(1j * phase).
        Shape (frequency, variable).
    """
    h_complex = ds["magnitude"] * np.exp(1j * ds["phase"])
    h_complex.name = "H"
    return h_complex


def from_complex(
    frequency: np.ndarray,
    h_complex: np.ndarray,
    coherence: np.ndarray,
    variable_names: list[str],
    sxx: np.ndarray | None = None,
    syy: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    **attrs: Any,
) -> xr.Dataset:
    """Create a TransferFunction Dataset from complex-valued H(f).

    Parameters
    ----------
    frequency : np.ndarray
        1D array of frequency values in Hz, shape (n_freq,).
    h_complex : np.ndarray
        Complex transfer function H(f).
        Shape (n_freq,) for single variable or (n_freq, n_var) for multiple.
    coherence : np.ndarray
        Coherence γ²(f), values in [0, 1].
        Shape (n_freq,) for single variable or (n_freq, n_var) for multiple.
    variable_names : list[str]
        Names of the response variables.
    sxx : np.ndarray, optional
        Input auto-spectrum, shape (n_freq,).
    syy : np.ndarray, optional
        Output auto-spectrum, shape (n_freq,) or (n_freq, n_var).
    config : dict, optional
        Configuration parameters.
    **attrs
        Additional attributes.

    Returns
    -------
    xr.Dataset
        A validated TransferFunction Dataset.
    """
    h_complex = np.asarray(h_complex, dtype=np.complex128)
    magnitude = np.abs(h_complex)
    phase = np.angle(h_complex)

    return create(
        frequency=frequency,
        magnitude=magnitude,
        phase=phase,
        coherence=coherence,
        variable_names=variable_names,
        sxx=sxx,
        syy=syy,
        config=config,
        **attrs,
    )


def select_variables(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Select a subset of variables from a TransferFunction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A valid TransferFunction Dataset.
    variables : list[str]
        Names of variables to select.

    Returns
    -------
    xr.Dataset
        A new TransferFunction Dataset with only the selected variables.
    """
    return ds.sel(variable=variables)


def select_frequency_range(
    ds: xr.Dataset, f_min: float | None = None, f_max: float | None = None
) -> xr.Dataset:
    """Select a frequency range from a TransferFunction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A valid TransferFunction Dataset.
    f_min : float, optional
        Minimum frequency (inclusive). If None, no lower bound.
    f_max : float, optional
        Maximum frequency (inclusive). If None, no upper bound.

    Returns
    -------
    xr.Dataset
        A new TransferFunction Dataset with only the selected frequency range.
    """
    freq = ds.coords["frequency"]
    mask = np.ones(len(freq), dtype=bool)

    if f_min is not None:
        mask &= freq >= f_min
    if f_max is not None:
        mask &= freq <= f_max

    return ds.isel(frequency=mask)


def compare(
    tf1: xr.Dataset,
    tf2: xr.Dataset,
    variable: str | None = None,
) -> dict[str, float]:
    """Compare two transfer functions.

    Computes correlation and difference statistics between the magnitudes
    of two transfer functions. Useful for validating that H(f) is consistent
    across different Hs values (linearity check).

    Parameters
    ----------
    tf1 : xr.Dataset
        First TransferFunction Dataset.
    tf2 : xr.Dataset
        Second TransferFunction Dataset.
    variable : str, optional
        Variable name to compare. If None and there's only one variable,
        uses that. If None and multiple variables exist, raises ValueError.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - correlation: Pearson correlation coefficient of magnitudes
        - mean_rel_diff: Mean relative difference (|m1-m2| / max(m1,m2))
        - max_rel_diff: Maximum relative difference
        - rms_diff: RMS difference in magnitude

    Raises
    ------
    ValueError
        If frequencies don't match or variable selection is ambiguous.

    Examples
    --------
    >>> tf_hs2 = identify.from_spectra("hs2.0_spectra.npz")
    >>> tf_hs3 = identify.from_spectra("hs3.0_spectra.npz")
    >>> stats = transfer_function.compare(tf_hs2, tf_hs3)
    >>> print(f"Correlation: {stats['correlation']:.4f}")
    Correlation: 0.9950
    """
    # Validate frequencies match
    f1 = tf1.coords["frequency"].values
    f2 = tf2.coords["frequency"].values

    if len(f1) != len(f2) or not np.allclose(f1, f2, rtol=1e-10):
        raise ValueError(
            f"Frequencies don't match: tf1 has {len(f1)} points, "
            f"tf2 has {len(f2)} points"
        )

    # Select variable
    vars1 = list(tf1.coords["variable"].values)
    vars2 = list(tf2.coords["variable"].values)

    if variable is None:
        if len(vars1) == 1 and len(vars2) == 1:
            variable = vars1[0]
            if vars1[0] != vars2[0]:
                # Different names but single variable - allow comparison
                pass
        else:
            raise ValueError(
                f"Multiple variables present. Specify which to compare. "
                f"tf1 has: {vars1}, tf2 has: {vars2}"
            )

    # Get magnitudes
    if variable in vars1:
        mag1 = tf1["magnitude"].sel(variable=variable).values
    else:
        # Single variable case with different name
        mag1 = tf1["magnitude"].values[:, 0]

    if variable in vars2:
        mag2 = tf2["magnitude"].sel(variable=variable).values
    else:
        mag2 = tf2["magnitude"].values[:, 0]

    # Compute statistics
    correlation = float(np.corrcoef(mag1, mag2)[0, 1])

    # Relative difference: |m1-m2| / max(m1, m2), avoiding div by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(mag1 - mag2) / np.maximum(mag1, mag2)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)

    mean_rel_diff = float(np.mean(rel_diff))
    max_rel_diff = float(np.max(rel_diff))

    # RMS difference
    rms_diff = float(np.sqrt(np.mean((mag1 - mag2) ** 2)))

    return {
        "correlation": correlation,
        "mean_rel_diff": mean_rel_diff,
        "max_rel_diff": max_rel_diff,
        "rms_diff": rms_diff,
    }


def average(
    tf_list: list[xr.Dataset],
    weights: str = "coherence",
) -> xr.Dataset:
    """Average multiple transfer functions.

    Combines multiple transfer function estimates into a single averaged
    estimate. Useful for reducing noise when multiple white noise simulations
    are available.

    Parameters
    ----------
    tf_list : list of xr.Dataset
        Transfer functions to average. Must have the same frequencies
        and variables.
    weights : {"coherence", "equal"}, optional
        Weighting scheme:
        - "coherence": Weight by coherence (higher coherence = more reliable)
        - "equal": Simple unweighted mean
        Default is "coherence".

    Returns
    -------
    xr.Dataset
        Averaged transfer function. The coherence in the result is the
        mean coherence across inputs.

    Raises
    ------
    ValueError
        If tf_list is empty, frequencies don't match, or variables don't match.

    Notes
    -----
    For phase averaging, complex averaging is used:
        H_avg = sum(w_i * H_i) / sum(w_i)
    where H_i = |H_i| * exp(1j * phase_i)

    This correctly handles phase wrapping and produces a meaningful
    average even when phases differ.

    Examples
    --------
    >>> tf_list = [
    ...     identify.from_spectra("hs2.0_spectra.npz"),
    ...     identify.from_spectra("hs3.0_spectra.npz"),
    ...     identify.from_spectra("hs4.0_spectra.npz"),
    ... ]
    >>> tf_avg = transfer_function.average(tf_list, weights="coherence")
    """
    if not tf_list:
        raise ValueError("tf_list cannot be empty")

    if len(tf_list) == 1:
        return tf_list[0].copy(deep=True)

    # Validate all have same frequencies and variables
    ref = tf_list[0]
    ref_freq = ref.coords["frequency"].values
    ref_vars = list(ref.coords["variable"].values)

    for i, tf in enumerate(tf_list[1:], start=1):
        tf_freq = tf.coords["frequency"].values
        tf_vars = list(tf.coords["variable"].values)

        if len(tf_freq) != len(ref_freq) or not np.allclose(
            tf_freq, ref_freq, rtol=1e-10
        ):
            raise ValueError(
                f"Frequency mismatch: tf_list[0] has {len(ref_freq)} points, "
                f"tf_list[{i}] has {len(tf_freq)} points"
            )

        if tf_vars != ref_vars:
            raise ValueError(
                f"Variable mismatch: tf_list[0] has {ref_vars}, "
                f"tf_list[{i}] has {tf_vars}"
            )

    n_tf = len(tf_list)
    n_freq = len(ref_freq)
    n_var = len(ref_vars)

    # Stack arrays: shape (n_tf, n_freq, n_var)
    mags = np.array([tf["magnitude"].values for tf in tf_list])
    phases = np.array([tf["phase"].values for tf in tf_list])
    cohs = np.array([tf["coherence"].values for tf in tf_list])

    # Compute weights
    if weights == "coherence":
        # Normalize coherence weights per frequency/variable
        w = cohs / cohs.sum(axis=0, keepdims=True)
        # Handle case where all coherences are zero
        w = np.where(np.isfinite(w), w, 1.0 / n_tf)
    elif weights == "equal":
        w = np.ones_like(mags) / n_tf
    else:
        raise ValueError(f"weights must be 'coherence' or 'equal', got '{weights}'")

    # Complex averaging for magnitude and phase
    h_complex = mags * np.exp(1j * phases)
    h_avg = (h_complex * w).sum(axis=0)

    avg_magnitude = np.abs(h_avg)
    avg_phase = np.angle(h_avg)

    # Mean coherence (not weighted - just informational)
    avg_coherence = cohs.mean(axis=0)

    # Average Sxx if present in all
    avg_sxx = None
    if all("Sxx" in tf for tf in tf_list):
        sxx_stack = np.array([tf["Sxx"].values for tf in tf_list])
        avg_sxx = sxx_stack.mean(axis=0)

    # Average Syy if present in all
    avg_syy = None
    if all("Syy" in tf for tf in tf_list):
        syy_stack = np.array([tf["Syy"].values for tf in tf_list])
        avg_syy = syy_stack.mean(axis=0)

    # Create averaged dataset
    return create(
        frequency=ref_freq,
        magnitude=avg_magnitude,
        phase=avg_phase,
        coherence=avg_coherence,
        variable_names=ref_vars,
        sxx=avg_sxx,
        syy=avg_syy,
        averaged_from=n_tf,
        averaging_weights=weights,
    )
