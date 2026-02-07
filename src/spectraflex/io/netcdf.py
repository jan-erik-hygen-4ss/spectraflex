"""NetCDF I/O for TransferFunction datasets and libraries.

Provides functions to save and load TransferFunction xarray Datasets
and TransferFunctionLibrary objects to NetCDF format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from spectraflex import transfer_function

if TYPE_CHECKING:
    from spectraflex.library import TransferFunctionLibrary


def save_transfer_function(
    tf: xr.Dataset,
    path: str | Path,
    engine: str = "netcdf4",
) -> None:
    """Save a TransferFunction Dataset to NetCDF file.

    Parameters
    ----------
    tf : xr.Dataset
        A valid TransferFunction Dataset.
    path : str or Path
        Output file path.
    engine : str, optional
        NetCDF engine to use, default "netcdf4".
        Alternatives: "h5netcdf", "scipy".
    """
    path = Path(path)
    transfer_function.validate(tf)

    # Serialize config dict as JSON if present
    tf_copy = tf.copy(deep=True)
    if "config" in tf_copy.attrs and isinstance(tf_copy.attrs["config"], dict):
        tf_copy.attrs["config_json"] = json.dumps(tf_copy.attrs["config"])
        del tf_copy.attrs["config"]

    tf_copy.to_netcdf(path, engine=engine)


def load_transfer_function(
    path: str | Path,
    engine: str | None = None,
) -> xr.Dataset:
    """Load a TransferFunction Dataset from NetCDF file.

    Parameters
    ----------
    path : str or Path
        Input file path.
    engine : str, optional
        NetCDF engine to use. If None, xarray auto-detects.

    Returns
    -------
    xr.Dataset
        Loaded TransferFunction Dataset.
    """
    path = Path(path)

    if engine:
        ds = xr.open_dataset(path, engine=engine)
    else:
        ds = xr.open_dataset(path)

    # Load into memory and close file handle
    tf = ds.load()
    ds.close()

    # Deserialize config
    if "config_json" in tf.attrs:
        tf.attrs["config"] = json.loads(tf.attrs["config_json"])
        del tf.attrs["config_json"]

    transfer_function.validate(tf)
    return tf


def save_library(
    library: TransferFunctionLibrary,
    path: str | Path,
    engine: str = "netcdf4",
) -> None:
    """Save a TransferFunctionLibrary to NetCDF file.

    Parameters
    ----------
    library : TransferFunctionLibrary
        The library to save.
    path : str or Path
        Output file path.
    engine : str, optional
        NetCDF engine to use, default "netcdf4".
    """
    from spectraflex.library import TransferFunctionLibrary

    if not isinstance(library, TransferFunctionLibrary):
        raise TypeError(f"Expected TransferFunctionLibrary, got {type(library)}")

    path = Path(path)
    combined = library.to_dataset()

    # Serialize config information as JSON
    combined.attrs["config_params_json"] = json.dumps(library.configs)
    combined.attrs["config_keys_json"] = json.dumps(library.config_keys)

    combined.to_netcdf(path, engine=engine)


def load_library(
    path: str | Path,
    engine: str | None = None,
) -> TransferFunctionLibrary:
    """Load a TransferFunctionLibrary from NetCDF file.

    Parameters
    ----------
    path : str or Path
        Input file path.
    engine : str, optional
        NetCDF engine to use. If None, xarray auto-detects.

    Returns
    -------
    TransferFunctionLibrary
        Loaded library.
    """
    from spectraflex.library import TransferFunctionLibrary

    path = Path(path)

    if engine:
        combined = xr.open_dataset(path, engine=engine)
    else:
        combined = xr.open_dataset(path)

    # Load into memory
    combined = combined.load()

    # Extract config information
    config_keys = json.loads(combined.attrs["config_keys_json"])
    configs = json.loads(combined.attrs["config_params_json"])

    lib = TransferFunctionLibrary(config_keys=config_keys)

    # Reconstruct individual datasets
    frequency = combined.coords["frequency"].values
    variable = list(combined.coords["variable"].values)

    for i, config in enumerate(configs):
        ds = transfer_function.create(
            frequency=frequency,
            magnitude=combined["magnitude"].values[:, :, i],
            phase=combined["phase"].values[:, :, i],
            coherence=combined["coherence"].values[:, :, i],
            variable_names=variable,
            config=config,
        )
        lib._configs.append(config)
        lib._datasets.append(ds)

    combined.close()
    return lib


def save_spectra(
    path: str | Path,
    frequency: Any,
    sxx: Any,
    syy: Any,
    sxy: Any,
    variable_names: list[str],
    **attrs: Any,
) -> None:
    """Save raw spectra to .npz file (for post-calc action output).

    Parameters
    ----------
    path : str or Path
        Output file path (should end in .npz).
    frequency : array-like
        Frequency array [Hz].
    sxx : array-like
        Input auto-spectrum.
    syy : array-like
        Output auto-spectra, shape (n_freq, n_var).
    sxy : array-like
        Cross-spectra (complex), shape (n_freq, n_var).
    variable_names : list[str]
        Names of response variables.
    **attrs
        Additional metadata to store.
    """
    import numpy as np

    path = Path(path)

    np.savez(
        path,
        frequency=np.asarray(frequency),
        Sxx=np.asarray(sxx),
        Syy=np.asarray(syy),
        Sxy=np.asarray(sxy),
        variable_names=np.array(variable_names, dtype=object),
        **attrs,
    )


def load_spectra(
    path: str | Path,
) -> dict[str, Any]:
    """Load raw spectra from .npz file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary with keys: frequency, Sxx, Syy, Sxy, variable_names,
        plus any additional saved attributes.
    """
    import numpy as np

    path = Path(path)
    data = np.load(path, allow_pickle=True)

    result = {
        "frequency": data["frequency"],
        "Sxx": data["Sxx"],
        "Syy": data["Syy"],
        "Sxy": data["Sxy"],
        "variable_names": list(data["variable_names"]),
    }

    # Add any extra attributes
    for key in data.files:
        if key not in result:
            result[key] = data[key]

    return result
