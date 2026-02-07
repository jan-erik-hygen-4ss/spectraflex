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
