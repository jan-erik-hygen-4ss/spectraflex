"""TransferFunctionLibrary — collection of transfer functions across configurations.

Provides a container for multiple TransferFunction datasets indexed by
configuration parameters, with methods for lookup, interpolation, and I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from spectraflex import transfer_function


class TransferFunctionLibrary:
    """Collection of transfer functions across operating configurations.

    The library stores multiple TransferFunction datasets, each identified
    by a unique configuration (e.g., hs, heading, current_speed, draft).
    It provides methods for:
    - Adding transfer functions with their configs
    - Selecting by exact config match
    - Looking up by nearest-neighbour or interpolation
    - Saving/loading to NetCDF format

    Parameters
    ----------
    config_keys : list[str], optional
        Names of the configuration parameters. If not provided, will be
        inferred from the first added transfer function's config.

    Attributes
    ----------
    config_keys : list[str]
        Names of configuration parameters.
    configs : list[dict]
        List of configuration dictionaries.
    datasets : list[xr.Dataset]
        List of TransferFunction datasets.

    Examples
    --------
    >>> lib = TransferFunctionLibrary()
    >>> lib.add(tf1)  # tf1 has config in attrs
    >>> lib.add(tf2)
    >>> tf = lib.select(heading=45.0, draft=21.0)
    >>> tf = lib.lookup(heading=40.0, draft=21.5, method="nearest")
    """

    def __init__(self, config_keys: list[str] | None = None) -> None:
        self._config_keys: list[str] | None = config_keys
        self._configs: list[dict[str, Any]] = []
        self._datasets: list[xr.Dataset] = []

    @property
    def config_keys(self) -> list[str]:
        """Configuration parameter names."""
        if self._config_keys is None:
            raise ValueError("No config_keys defined. Add a transfer function first.")
        return self._config_keys

    @property
    def configs(self) -> list[dict[str, Any]]:
        """List of configuration dictionaries."""
        return self._configs.copy()

    @property
    def datasets(self) -> list[xr.Dataset]:
        """List of TransferFunction datasets."""
        return self._datasets.copy()

    def __len__(self) -> int:
        """Number of transfer functions in the library."""
        return len(self._datasets)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "TransferFunctionLibrary(empty)"
        keys = self._config_keys or []
        return f"TransferFunctionLibrary(n_configs={len(self)}, config_keys={keys})"

    def add(
        self,
        tf: xr.Dataset,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a transfer function to the library.

        Parameters
        ----------
        tf : xr.Dataset
            A valid TransferFunction Dataset.
        config : dict, optional
            Configuration parameters. If not provided, will be extracted
            from tf.attrs["config"].

        Raises
        ------
        ValueError
            If config is missing or doesn't match existing config_keys.
        """
        # Validate the transfer function
        transfer_function.validate(tf)

        # Get config
        if config is None:
            if "config" not in tf.attrs:
                raise ValueError(
                    "No config provided and tf.attrs['config'] is missing. "
                    "Provide config explicitly or ensure tf has config attribute."
                )
            config = tf.attrs["config"]

        if not isinstance(config, dict):
            raise ValueError(f"config must be a dict, got {type(config)}")

        # Initialize or validate config_keys
        if self._config_keys is None:
            self._config_keys = sorted(config.keys())
        else:
            config_key_set = set(config.keys())
            expected_set = set(self._config_keys)
            if config_key_set != expected_set:
                raise ValueError(
                    f"Config keys {sorted(config_key_set)} don't match "
                    f"expected keys {self._config_keys}"
                )

        # Check for duplicate config
        for existing_config in self._configs:
            if self._configs_equal(config, existing_config):
                raise ValueError(f"Duplicate config already exists: {config}")

        # Store
        self._configs.append(config.copy())
        self._datasets.append(tf.copy(deep=True))

    def _configs_equal(self, config1: dict[str, Any], config2: dict[str, Any]) -> bool:
        """Check if two configs are equal (handling float comparison)."""
        if set(config1.keys()) != set(config2.keys()):
            return False
        for key in config1:
            v1, v2 = config1[key], config2[key]
            if isinstance(v1, float) and isinstance(v2, float):
                if not np.isclose(v1, v2):
                    return False
            elif v1 != v2:
                return False
        return True

    def select(self, **kwargs: Any) -> xr.Dataset:
        """Select a transfer function by exact config match.

        Parameters
        ----------
        **kwargs
            Configuration parameters to match exactly.

        Returns
        -------
        xr.Dataset
            The matching TransferFunction Dataset.

        Raises
        ------
        KeyError
            If no exact match is found.
        ValueError
            If multiple matches found (shouldn't happen with unique configs).
        """
        matches = []
        for i, config in enumerate(self._configs):
            if self._config_matches(config, kwargs):
                matches.append(i)

        if len(matches) == 0:
            raise KeyError(f"No transfer function found matching {kwargs}")
        if len(matches) > 1:
            raise ValueError(f"Multiple matches found for {kwargs}: {matches}")

        return self._datasets[matches[0]].copy(deep=True)

    def _config_matches(self, config: dict[str, Any], query: dict[str, Any]) -> bool:
        """Check if config matches query (query may be partial)."""
        for key, value in query.items():
            if key not in config:
                return False
            config_val = config[key]
            if isinstance(value, float) and isinstance(config_val, float):
                if not np.isclose(value, config_val):
                    return False
            elif config_val != value:
                return False
        return True

    def lookup(
        self,
        method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Look up a transfer function by nearest-neighbour or interpolation.

        Parameters
        ----------
        method : str, optional
            Lookup method: "nearest" or "linear". Default "nearest".
        **kwargs
            Configuration parameters for lookup.

        Returns
        -------
        xr.Dataset
            The looked-up TransferFunction Dataset.

        Raises
        ------
        ValueError
            If library is empty or method is unsupported.
        """
        if len(self) == 0:
            raise ValueError("Library is empty")

        if method == "nearest":
            return self._lookup_nearest(kwargs)
        elif method == "linear":
            return self._lookup_interpolate(kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'linear'.")

    def _lookup_nearest(self, query: dict[str, Any]) -> xr.Dataset:
        """Find nearest config using normalized Euclidean distance."""
        # Build config arrays for distance calculation
        config_keys = self.config_keys
        query_values = np.array([float(query.get(k, 0.0)) for k in config_keys])

        config_array = np.array(
            [[float(c.get(k, 0.0)) for k in config_keys] for c in self._configs]
        )

        # Normalize by range to handle different scales
        ranges = config_array.max(axis=0) - config_array.min(axis=0)
        ranges = np.where(ranges > 0, ranges, 1.0)  # avoid division by zero

        normalized_configs = config_array / ranges
        normalized_query = query_values / ranges

        # Find nearest
        distances = np.linalg.norm(normalized_configs - normalized_query, axis=1)
        nearest_idx = int(np.argmin(distances))

        return self._datasets[nearest_idx].copy(deep=True)

    def _lookup_interpolate(self, query: dict[str, Any]) -> xr.Dataset:
        """Interpolate transfer function using inverse distance weighting.

        Uses inverse distance weighting (IDW) with k nearest neighbors.
        """
        if len(self) == 1:
            return self._datasets[0].copy(deep=True)

        config_keys = self.config_keys
        query_values = np.array([float(query.get(k, 0.0)) for k in config_keys])

        config_array = np.array(
            [[float(c.get(k, 0.0)) for k in config_keys] for c in self._configs]
        )

        # Normalize
        ranges = config_array.max(axis=0) - config_array.min(axis=0)
        ranges = np.where(ranges > 0, ranges, 1.0)

        normalized_configs = config_array / ranges
        normalized_query = query_values / ranges

        # Calculate distances
        distances = np.linalg.norm(normalized_configs - normalized_query, axis=1)

        # Check for exact match
        min_dist = distances.min()
        if min_dist < 1e-10:
            return self._datasets[int(np.argmin(distances))].copy(deep=True)

        # Use k nearest neighbors for interpolation (k = min(4, n_configs))
        k = min(4, len(self))
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]

        # Inverse distance weights
        weights = 1.0 / nearest_distances
        weights = weights / weights.sum()

        # Interpolate magnitude, phase, coherence
        ref_ds = self._datasets[nearest_indices[0]]
        frequency = ref_ds.coords["frequency"].values
        variable = ref_ds.coords["variable"].values

        mag_interp = np.zeros_like(ref_ds["magnitude"].values)
        phase_interp = np.zeros_like(ref_ds["phase"].values)
        coh_interp = np.zeros_like(ref_ds["coherence"].values)

        for idx, w in zip(nearest_indices, weights):
            ds = self._datasets[idx]
            mag_interp += w * ds["magnitude"].values
            phase_interp += w * ds["phase"].values
            coh_interp += w * ds["coherence"].values

        # Create interpolated dataset
        result = transfer_function.create(
            frequency=frequency,
            magnitude=mag_interp,
            phase=phase_interp,
            coherence=coh_interp,
            variable_names=list(variable),
            config=query,
            interpolated=True,
            interpolation_method="idw",
            interpolation_k=k,
        )

        return result

    def to_dataset(self) -> xr.Dataset:
        """Combine all transfer functions into a single Dataset with config dimension.

        Returns
        -------
        xr.Dataset
            Combined dataset with dimensions (frequency, variable, config).
        """
        if len(self) == 0:
            raise ValueError("Library is empty")

        # Create config coordinate labels
        config_labels = [self._config_to_label(c) for c in self._configs]

        # Stack all datasets
        ref_ds = self._datasets[0]
        n_freq = len(ref_ds.coords["frequency"])
        n_var = len(ref_ds.coords["variable"])
        n_config = len(self)

        magnitude = np.zeros((n_freq, n_var, n_config))
        phase = np.zeros((n_freq, n_var, n_config))
        coherence = np.zeros((n_freq, n_var, n_config))

        for i, ds in enumerate(self._datasets):
            magnitude[:, :, i] = ds["magnitude"].values
            phase[:, :, i] = ds["phase"].values
            coherence[:, :, i] = ds["coherence"].values

        combined = xr.Dataset(
            data_vars={
                "magnitude": (["frequency", "variable", "config"], magnitude),
                "phase": (["frequency", "variable", "config"], phase),
                "coherence": (["frequency", "variable", "config"], coherence),
            },
            coords={
                "frequency": ref_ds.coords["frequency"].values,
                "variable": ref_ds.coords["variable"].values,
                "config": config_labels,
            },
        )

        return combined

    def _config_to_label(self, config: dict[str, Any]) -> str:
        """Create a string label from config dict."""
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        return "|".join(parts)

    def _label_to_config(self, label: str) -> dict[str, Any]:
        """Parse a config label back to dict."""
        config: dict[str, Any] = {}
        for part in label.split("|"):
            key, value = part.split("=")
            # Try to parse as number
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value
        return config

    def save(self, path: str | Path) -> None:
        """Save library to NetCDF file.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in .nc).
        """
        path = Path(path)
        combined = self.to_dataset()

        # Add config params as JSON string attribute for reliable serialization
        import json

        combined.attrs["config_params_json"] = json.dumps(self._configs)
        combined.attrs["config_keys_json"] = json.dumps(self._config_keys)

        combined.to_netcdf(path, engine="netcdf4")

    @classmethod
    def load(cls, path: str | Path) -> TransferFunctionLibrary:
        """Load library from NetCDF file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        TransferFunctionLibrary
            Loaded library.
        """
        import json

        path = Path(path)
        combined = xr.open_dataset(path)

        # Extract config information
        config_keys = json.loads(combined.attrs["config_keys_json"])
        configs = json.loads(combined.attrs["config_params_json"])

        lib = cls(config_keys=config_keys)

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

    def get_config_range(self, key: str) -> tuple[float, float]:
        """Get the range of values for a config parameter.

        Parameters
        ----------
        key : str
            Configuration parameter name.

        Returns
        -------
        tuple[float, float]
            (min_value, max_value)
        """
        if len(self) == 0:
            raise ValueError("Library is empty")

        values = [c[key] for c in self._configs if key in c]
        if not values:
            raise KeyError(f"Key '{key}' not found in configs")

        return float(min(values)), float(max(values))

    def get_unique_values(self, key: str) -> list[Any]:
        """Get unique values for a config parameter.

        Parameters
        ----------
        key : str
            Configuration parameter name.

        Returns
        -------
        list
            Sorted unique values.
        """
        if len(self) == 0:
            raise ValueError("Library is empty")

        values = set()
        for c in self._configs:
            if key in c:
                values.add(c[key])

        return sorted(values)

    def filter(self, **kwargs: Any) -> TransferFunctionLibrary:
        """Create a new library containing only configs matching the filter.

        Parameters
        ----------
        **kwargs
            Configuration parameters to match.

        Returns
        -------
        TransferFunctionLibrary
            Filtered library.
        """
        filtered = TransferFunctionLibrary(config_keys=self._config_keys)

        for config, ds in zip(self._configs, self._datasets):
            if self._config_matches(config, kwargs):
                filtered._configs.append(config.copy())
                filtered._datasets.append(ds.copy(deep=True))

        return filtered
