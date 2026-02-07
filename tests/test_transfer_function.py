"""Tests for spectraflex.transfer_function module.

Tests cover:
- create(): factory function for TransferFunction xarray.Dataset
- validate(): schema validation for TransferFunction datasets
- Dataset structure: dimensions, coordinates, data variables, attributes
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from spectraflex.transfer_function import create, validate


class TestCreate:
    """Tests for the create() factory function."""

    def test_create_returns_dataset(self, frequency_array: np.ndarray) -> None:
        """create() should return an xarray.Dataset."""
        n_freq = len(frequency_array)
        n_var = 2
        variable_names = ["UFJ_Angle", "LFJ_Angle"]

        magnitude = np.ones((n_freq, n_var))
        phase = np.zeros((n_freq, n_var))
        coherence = np.ones((n_freq, n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert isinstance(tf, xr.Dataset)

    def test_create_has_correct_dimensions(
        self, frequency_array: np.ndarray
    ) -> None:
        """Created Dataset should have frequency and variable dimensions."""
        n_freq = len(frequency_array)
        n_var = 3
        variable_names = ["var1", "var2", "var3"]

        magnitude = np.ones((n_freq, n_var))
        phase = np.zeros((n_freq, n_var))
        coherence = np.ones((n_freq, n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert "frequency" in tf.dims
        assert "variable" in tf.dims
        assert tf.dims["frequency"] == n_freq
        assert tf.dims["variable"] == n_var

    def test_create_has_correct_coordinates(
        self, frequency_array: np.ndarray
    ) -> None:
        """Created Dataset should have frequency and variable coordinates."""
        variable_names = ["UFJ", "LFJ"]
        n_var = len(variable_names)

        magnitude = np.ones((len(frequency_array), n_var))
        phase = np.zeros((len(frequency_array), n_var))
        coherence = np.ones((len(frequency_array), n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert_allclose(tf.coords["frequency"].values, frequency_array)
        assert list(tf.coords["variable"].values) == variable_names

    def test_create_has_required_data_vars(
        self, frequency_array: np.ndarray
    ) -> None:
        """Created Dataset should have magnitude, phase, coherence data variables."""
        n_var = 1
        magnitude = np.ones((len(frequency_array), n_var))
        phase = np.zeros((len(frequency_array), n_var))
        coherence = np.ones((len(frequency_array), n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=["response"],
        )

        assert "magnitude" in tf.data_vars
        assert "phase" in tf.data_vars
        assert "coherence" in tf.data_vars

    def test_create_data_var_shapes(self, frequency_array: np.ndarray) -> None:
        """Data variables should have shape (frequency, variable)."""
        n_freq = len(frequency_array)
        n_var = 2
        variable_names = ["A", "B"]

        magnitude = np.random.rand(n_freq, n_var)
        phase = np.random.rand(n_freq, n_var)
        coherence = np.random.rand(n_freq, n_var)

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert tf.magnitude.shape == (n_freq, n_var)
        assert tf.phase.shape == (n_freq, n_var)
        assert tf.coherence.shape == (n_freq, n_var)

    def test_create_preserves_values(self, frequency_array: np.ndarray) -> None:
        """Data values should be preserved exactly."""
        n_var = 2
        variable_names = ["A", "B"]

        magnitude = np.random.rand(len(frequency_array), n_var)
        phase = np.random.rand(len(frequency_array), n_var)
        coherence = np.random.rand(len(frequency_array), n_var)

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert_allclose(tf.magnitude.values, magnitude)
        assert_allclose(tf.phase.values, phase)
        assert_allclose(tf.coherence.values, coherence)

    def test_create_with_config(self, frequency_array: np.ndarray) -> None:
        """Config dict should be stored in attributes."""
        n_var = 1
        magnitude = np.ones((len(frequency_array), n_var))
        phase = np.zeros((len(frequency_array), n_var))
        coherence = np.ones((len(frequency_array), n_var))

        config = {"hs": 2.0, "draft": 21.0, "heading": 45.0, "current_speed": 0.5}

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=["response"],
            config=config,
        )

        assert "config" in tf.attrs
        assert tf.attrs["config"] == config

    def test_create_with_additional_attrs(
        self, frequency_array: np.ndarray
    ) -> None:
        """Additional kwargs should be stored as attributes."""
        n_var = 1
        magnitude = np.ones((len(frequency_array), n_var))
        phase = np.zeros((len(frequency_array), n_var))
        coherence = np.ones((len(frequency_array), n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=["response"],
            orcaflex_model="test.dat",
            sim_duration=512.0,
            sample_interval=0.1,
            nperseg=1024,
        )

        assert tf.attrs["orcaflex_model"] == "test.dat"
        assert tf.attrs["sim_duration"] == 512.0
        assert tf.attrs["sample_interval"] == 0.1
        assert tf.attrs["nperseg"] == 1024

    def test_create_single_variable(self, frequency_array: np.ndarray) -> None:
        """Should work with single variable."""
        magnitude = np.ones((len(frequency_array), 1))
        phase = np.zeros((len(frequency_array), 1))
        coherence = np.ones((len(frequency_array), 1))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=["single_var"],
        )

        assert tf.dims["variable"] == 1
        assert list(tf.coords["variable"].values) == ["single_var"]

    def test_create_many_variables(self, frequency_array: np.ndarray) -> None:
        """Should work with many variables."""
        n_var = 20
        variable_names = [f"var_{i}" for i in range(n_var)]

        magnitude = np.ones((len(frequency_array), n_var))
        phase = np.zeros((len(frequency_array), n_var))
        coherence = np.ones((len(frequency_array), n_var))

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        assert tf.dims["variable"] == n_var


class TestCreateValidation:
    """Tests for input validation in create()."""

    def test_create_rejects_shape_mismatch(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should raise error if array shapes don't match."""
        n_freq = len(frequency_array)

        magnitude = np.ones((n_freq, 2))
        phase = np.ones((n_freq, 3))  # Wrong shape
        coherence = np.ones((n_freq, 2))

        with pytest.raises(ValueError, match="shape|mismatch|dimension"):
            create(
                frequency=frequency_array,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["A", "B"],
            )

    def test_create_rejects_freq_length_mismatch(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should raise error if frequency length doesn't match arrays."""
        wrong_freq = np.linspace(0.01, 0.5, 100)  # Wrong length

        magnitude = np.ones((len(frequency_array), 1))
        phase = np.zeros((len(frequency_array), 1))
        coherence = np.ones((len(frequency_array), 1))

        with pytest.raises(ValueError, match="frequency|shape|length"):
            create(
                frequency=wrong_freq,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["response"],
            )

    def test_create_rejects_var_names_mismatch(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should raise error if variable_names length doesn't match arrays."""
        magnitude = np.ones((len(frequency_array), 3))
        phase = np.zeros((len(frequency_array), 3))
        coherence = np.ones((len(frequency_array), 3))

        with pytest.raises(ValueError, match="variable|names|length"):
            create(
                frequency=frequency_array,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["A", "B"],  # Only 2 names for 3 variables
            )

    def test_create_rejects_negative_frequencies(self) -> None:
        """Should raise error for negative frequencies."""
        freq = np.array([-0.1, 0.0, 0.1, 0.2])

        magnitude = np.ones((4, 1))
        phase = np.zeros((4, 1))
        coherence = np.ones((4, 1))

        with pytest.raises(ValueError, match="negative|positive|frequency"):
            create(
                frequency=freq,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["response"],
            )

    def test_create_rejects_negative_magnitude(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should raise error for negative magnitude values."""
        magnitude = np.ones((len(frequency_array), 1))
        magnitude[50] = -1.0  # Negative value
        phase = np.zeros((len(frequency_array), 1))
        coherence = np.ones((len(frequency_array), 1))

        with pytest.raises(ValueError, match="magnitude|negative|positive"):
            create(
                frequency=frequency_array,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["response"],
            )

    def test_create_rejects_invalid_coherence(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should raise error for coherence outside [0, 1]."""
        magnitude = np.ones((len(frequency_array), 1))
        phase = np.zeros((len(frequency_array), 1))
        coherence = np.ones((len(frequency_array), 1))
        coherence[50] = 1.5  # > 1

        with pytest.raises(ValueError, match="coherence|range|0.*1"):
            create(
                frequency=frequency_array,
                magnitude=magnitude,
                phase=phase,
                coherence=coherence,
                variable_names=["response"],
            )

    def test_create_allows_zero_coherence(
        self, frequency_array: np.ndarray
    ) -> None:
        """Zero coherence should be allowed."""
        magnitude = np.ones((len(frequency_array), 1))
        phase = np.zeros((len(frequency_array), 1))
        coherence = np.zeros((len(frequency_array), 1))  # All zero

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=["response"],
        )

        assert_allclose(tf.coherence.values, 0.0)


class TestValidate:
    """Tests for the validate() function."""

    def test_validate_accepts_valid_dataset(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should not raise for valid TransferFunction dataset."""
        tf = create(
            frequency=frequency_array,
            magnitude=np.ones((len(frequency_array), 1)),
            phase=np.zeros((len(frequency_array), 1)),
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
        )

        # Should not raise
        validate(tf)

    def test_validate_rejects_missing_magnitude(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should raise if magnitude is missing."""
        ds = xr.Dataset(
            {
                "phase": (["frequency", "variable"], np.zeros((len(frequency_array), 1))),
                "coherence": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="magnitude"):
            validate(ds)

    def test_validate_rejects_missing_phase(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should raise if phase is missing."""
        ds = xr.Dataset(
            {
                "magnitude": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
                "coherence": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="phase"):
            validate(ds)

    def test_validate_rejects_missing_coherence(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should raise if coherence is missing."""
        ds = xr.Dataset(
            {
                "magnitude": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
                "phase": (["frequency", "variable"], np.zeros((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="coherence"):
            validate(ds)

    def test_validate_rejects_missing_frequency_coord(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should raise if frequency coordinate is missing."""
        ds = xr.Dataset(
            {
                "magnitude": (["freq", "variable"], np.ones((len(frequency_array), 1))),
                "phase": (["freq", "variable"], np.zeros((len(frequency_array), 1))),
                "coherence": (["freq", "variable"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "freq": frequency_array,  # Wrong name
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="frequency"):
            validate(ds)

    def test_validate_rejects_missing_variable_coord(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should raise if variable coordinate is missing."""
        ds = xr.Dataset(
            {
                "magnitude": (["frequency", "var"], np.ones((len(frequency_array), 1))),
                "phase": (["frequency", "var"], np.zeros((len(frequency_array), 1))),
                "coherence": (["frequency", "var"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "var": ["response"],  # Wrong name
            },
        )

        with pytest.raises(ValueError, match="variable"):
            validate(ds)

    def test_validate_rejects_nan_values(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should reject datasets with NaN values."""
        magnitude = np.ones((len(frequency_array), 1))
        magnitude[50] = np.nan

        ds = xr.Dataset(
            {
                "magnitude": (["frequency", "variable"], magnitude),
                "phase": (["frequency", "variable"], np.zeros((len(frequency_array), 1))),
                "coherence": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="NaN|nan|finite"):
            validate(ds)

    def test_validate_rejects_inf_values(
        self, frequency_array: np.ndarray
    ) -> None:
        """validate() should reject datasets with infinite values."""
        magnitude = np.ones((len(frequency_array), 1))
        magnitude[50] = np.inf

        ds = xr.Dataset(
            {
                "magnitude": (["frequency", "variable"], magnitude),
                "phase": (["frequency", "variable"], np.zeros((len(frequency_array), 1))),
                "coherence": (["frequency", "variable"], np.ones((len(frequency_array), 1))),
            },
            coords={
                "frequency": frequency_array,
                "variable": ["response"],
            },
        )

        with pytest.raises(ValueError, match="inf|finite"):
            validate(ds)


class TestDatasetUsability:
    """Tests for using the TransferFunction Dataset in typical workflows."""

    def test_select_variable(self, frequency_array: np.ndarray) -> None:
        """Should be able to select a single variable."""
        variable_names = ["UFJ", "LFJ", "WH_BM"]
        n_var = len(variable_names)

        magnitude = np.random.rand(len(frequency_array), n_var)
        phase = np.random.rand(len(frequency_array), n_var)
        coherence = np.random.rand(len(frequency_array), n_var)

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            variable_names=variable_names,
        )

        # Select single variable
        ufj = tf.sel(variable="UFJ")
        assert ufj.magnitude.shape == (len(frequency_array),)

    def test_slice_frequency(self, frequency_array: np.ndarray) -> None:
        """Should be able to slice by frequency range."""
        tf = create(
            frequency=frequency_array,
            magnitude=np.ones((len(frequency_array), 1)),
            phase=np.zeros((len(frequency_array), 1)),
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
        )

        # Slice to frequency range
        tf_subset = tf.sel(frequency=slice(0.05, 0.2))
        assert tf_subset.dims["frequency"] < tf.dims["frequency"]
        assert tf_subset.frequency.min() >= 0.05
        assert tf_subset.frequency.max() <= 0.2

    def test_interpolate_frequency(self, frequency_array: np.ndarray) -> None:
        """Should be able to interpolate to new frequencies."""
        tf = create(
            frequency=frequency_array,
            magnitude=np.ones((len(frequency_array), 1)),
            phase=np.zeros((len(frequency_array), 1)),
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
        )

        new_freq = np.linspace(0.05, 0.3, 100)
        tf_interp = tf.interp(frequency=new_freq)

        assert tf_interp.dims["frequency"] == 100
        assert_allclose(tf_interp.frequency.values, new_freq)

    def test_arithmetic_on_magnitude(self, frequency_array: np.ndarray) -> None:
        """Should be able to perform arithmetic on data variables."""
        tf = create(
            frequency=frequency_array,
            magnitude=2.0 * np.ones((len(frequency_array), 1)),
            phase=np.zeros((len(frequency_array), 1)),
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
        )

        # Compute |H|^2
        H_squared = tf.magnitude ** 2
        assert_allclose(H_squared.values, 4.0)

    def test_save_load_netcdf(
        self, frequency_array: np.ndarray, tmp_path
    ) -> None:
        """Should be able to save and load as NetCDF."""
        tf = create(
            frequency=frequency_array,
            magnitude=np.ones((len(frequency_array), 1)),
            phase=np.zeros((len(frequency_array), 1)),
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
            config={"hs": 2.0, "heading": 45.0},
        )

        # Save
        path = tmp_path / "test_tf.nc"
        tf.to_netcdf(path)

        # Load
        tf_loaded = xr.open_dataset(path)

        assert_allclose(tf_loaded.magnitude.values, tf.magnitude.values)
        assert_allclose(tf_loaded.phase.values, tf.phase.values)
        assert_allclose(tf_loaded.coherence.values, tf.coherence.values)


class TestComplexTransferFunction:
    """Tests for complex-valued transfer function representation."""

    def test_magnitude_phase_to_complex(
        self, frequency_array: np.ndarray
    ) -> None:
        """Should be able to reconstruct complex H(f) from magnitude and phase."""
        # Known complex transfer function
        H_complex = 2.0 * np.exp(1j * np.pi / 4)  # magnitude=2, phase=π/4
        H_complex = np.full((len(frequency_array), 1), H_complex)

        magnitude = np.abs(H_complex)
        phase = np.angle(H_complex)

        tf = create(
            frequency=frequency_array,
            magnitude=magnitude,
            phase=phase,
            coherence=np.ones_like(magnitude),
            variable_names=["response"],
        )

        # Reconstruct complex H
        H_reconstructed = tf.magnitude.values * np.exp(1j * tf.phase.values)

        assert_allclose(H_reconstructed, H_complex)

    def test_phase_wrapping(self, frequency_array: np.ndarray) -> None:
        """Phase values should be in [-π, π] or [0, 2π] range."""
        # Create phase values that wrap around
        phase = np.linspace(-2 * np.pi, 2 * np.pi, len(frequency_array))
        phase = phase[:, np.newaxis]

        tf = create(
            frequency=frequency_array,
            magnitude=np.ones((len(frequency_array), 1)),
            phase=phase,
            coherence=np.ones((len(frequency_array), 1)),
            variable_names=["response"],
        )

        # Phase should be preserved as-is (or wrapped)
        # Implementation may choose to wrap or preserve
        assert np.all(np.isfinite(tf.phase.values))
