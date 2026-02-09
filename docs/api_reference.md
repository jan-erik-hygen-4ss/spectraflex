# API Reference

This document provides a complete reference for the spectraflex public API.

## Core Modules

### spectraflex.spectrum

Wave spectrum definitions and utilities.

#### `jonswap(hs, tp, f, gamma=3.3, sigma_a=0.07, sigma_b=0.09)`

Create a JONSWAP wave spectrum.

**Parameters:**
- `hs` (float): Significant wave height [m]
- `tp` (float): Peak period [s]
- `f` (np.ndarray): Frequency array [Hz]
- `gamma` (float): Peak enhancement factor, default 3.3
- `sigma_a` (float): Spectral width for f < fp, default 0.07
- `sigma_b` (float): Spectral width for f > fp, default 0.09

**Returns:** `xr.DataArray` - Wave spectrum S(f) [m²/Hz]

**Example:**
```python
import numpy as np
from spectraflex import spectrum

f = np.linspace(0.01, 0.5, 256)
wave = spectrum.jonswap(hs=3.0, tp=10.0, f=f, gamma=3.3)
```

---

#### `pierson_moskowitz(hs, tp, f)`

Create a Pierson-Moskowitz wave spectrum (fully developed sea).

**Parameters:**
- `hs` (float): Significant wave height [m]
- `tp` (float): Peak period [s]
- `f` (np.ndarray): Frequency array [Hz]

**Returns:** `xr.DataArray` - Wave spectrum S(f) [m²/Hz]

---

#### `white_noise(hs, f, f_min=None, f_max=None)`

Create a white noise (flat) wave spectrum for transfer function identification.

**Parameters:**
- `hs` (float): Significant wave height [m]
- `f` (np.ndarray): Frequency array [Hz]
- `f_min` (float, optional): Minimum frequency for flat region
- `f_max` (float, optional): Maximum frequency for flat region

**Returns:** `xr.DataArray` - White noise spectrum

---

#### `from_array(f, s, name="S", attrs=None)`

Create a wave spectrum DataArray from arrays.

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array [m²/Hz]
- `name` (str): Name for the DataArray
- `attrs` (dict, optional): Additional attributes

**Returns:** `xr.DataArray` - Wave spectrum

---

#### `scale_to_hs(spectrum, hs)`

Scale a spectrum to a target significant wave height.

**Parameters:**
- `spectrum` (xr.DataArray): Input wave spectrum
- `hs` (float): Target significant wave height [m]

**Returns:** `xr.DataArray` - Scaled spectrum

---

#### `frequency_array(f_min=0.01, f_max=0.5, n_freq=256, spacing="linear")`

Create a frequency array for spectrum calculations.

**Parameters:**
- `f_min` (float): Minimum frequency [Hz]
- `f_max` (float): Maximum frequency [Hz]
- `n_freq` (int): Number of frequency points
- `spacing` (str): "linear" or "log"

**Returns:** `np.ndarray` - Frequency array [Hz]

---

### spectraflex.statistics

Spectral statistics calculations.

#### `spectral_moments(f, s, orders=(0, 1, 2, 4))`

Compute spectral moments of a power spectrum.

The n-th spectral moment is: m_n = ∫ f^n × S(f) df

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array
- `orders` (tuple): Which moments to compute

**Returns:** `dict[int, float]` - Dictionary mapping moment order to value

---

#### `hs_from_spectrum(f, s)`

Compute significant wave height from a spectrum: Hs = 4 × √m₀

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array [m²/Hz]

**Returns:** `float` - Significant wave height [m]

---

#### `hs_from_m0(m0_val)`

Compute significant height from zeroth moment: Hs = 4 × √m₀

**Parameters:**
- `m0_val` (float): Zeroth spectral moment (variance)

**Returns:** `float` - Significant height

---

#### `tp_from_spectrum(f, s)`

Compute peak period from a spectrum: Tp = 1/fp

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array

**Returns:** `float` - Peak period [s]

---

#### `tz_from_spectrum(f, s)`

Compute zero-crossing period from a spectrum: Tz = √(m₀/m₂)

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array

**Returns:** `float` - Zero-crossing period [s]

---

#### `mpm_rayleigh(m0_val, duration, m2_val=None, tz=None)`

Compute most probable maximum assuming Rayleigh distribution.

MPM = σ × √(2 × ln(N)), where σ = √m₀ and N = duration/Tz

**Parameters:**
- `m0_val` (float): Zeroth spectral moment (variance)
- `duration` (float): Duration of the sea state [s]
- `m2_val` (float, optional): Second spectral moment
- `tz` (float, optional): Zero-crossing period [s]

**Returns:** `float` - Most probable maximum value

---

#### `all_statistics(f, s, duration=10800.0)`

Compute comprehensive statistics from a spectrum.

**Parameters:**
- `f` (np.ndarray): Frequency array [Hz]
- `s` (np.ndarray): Spectral density array
- `duration` (float): Duration for extreme calculations [s]

**Returns:** `dict` with keys: m0, m1, m2, m4, hs, tp, tz, epsilon, mpm, sigma

---

### spectraflex.transfer_function

TransferFunction xarray Dataset factory and validation.

#### `create(frequency, magnitude, phase, coherence, variable_names, sxx=None, syy=None, config=None, **attrs)`

Create a validated TransferFunction Dataset.

**Parameters:**
- `frequency` (np.ndarray): Frequency values [Hz], shape (n_freq,)
- `magnitude` (np.ndarray): |H(f)| in response_units/m, shape (n_freq, n_var)
- `phase` (np.ndarray): arg(H(f)) in radians, shape (n_freq, n_var)
- `coherence` (np.ndarray): γ²(f) in [0, 1], shape (n_freq, n_var)
- `variable_names` (list[str]): Names of response variables
- `sxx` (np.ndarray, optional): Input auto-spectrum
- `syy` (np.ndarray, optional): Output auto-spectrum
- `config` (dict, optional): Configuration parameters
- `**attrs`: Additional attributes

**Returns:** `xr.Dataset` - Validated TransferFunction Dataset

---

#### `validate(ds)`

Validate that a Dataset conforms to the TransferFunction schema.

**Parameters:**
- `ds` (xr.Dataset): Dataset to validate

**Raises:** `ValueError` if invalid

---

#### `is_valid(ds)`

Check if a Dataset is a valid TransferFunction.

**Parameters:**
- `ds` (xr.Dataset): Dataset to check

**Returns:** `bool` - True if valid

---

#### `complex_transfer_function(ds)`

Get the complex-valued transfer function H(f) from a Dataset.

**Parameters:**
- `ds` (xr.Dataset): A valid TransferFunction Dataset

**Returns:** `xr.DataArray` - Complex H(f) = |H| × exp(j × phase)

---

#### `from_complex(frequency, h_complex, coherence, variable_names, ...)`

Create a TransferFunction Dataset from complex-valued H(f).

**Parameters:**
- `frequency` (np.ndarray): Frequency values [Hz]
- `h_complex` (np.ndarray): Complex transfer function H(f)
- `coherence` (np.ndarray): Coherence values
- `variable_names` (list[str]): Variable names

**Returns:** `xr.Dataset` - TransferFunction Dataset

---

#### `select_variables(ds, variables)`

Select a subset of variables from a TransferFunction Dataset.

**Parameters:**
- `ds` (xr.Dataset): TransferFunction Dataset
- `variables` (list[str]): Variables to select

**Returns:** `xr.Dataset` - Dataset with selected variables

---

#### `select_frequency_range(ds, f_min=None, f_max=None)`

Select a frequency range from a TransferFunction Dataset.

**Parameters:**
- `ds` (xr.Dataset): TransferFunction Dataset
- `f_min` (float, optional): Minimum frequency
- `f_max` (float, optional): Maximum frequency

**Returns:** `xr.Dataset` - Dataset with selected frequency range

---

#### `compare(tf1, tf2, variable=None)`

Compare two transfer functions.

**Parameters:**
- `tf1` (xr.Dataset): First TransferFunction
- `tf2` (xr.Dataset): Second TransferFunction
- `variable` (str, optional): Variable to compare

**Returns:** `dict` with keys: correlation, mean_rel_diff, max_rel_diff, rms_diff

---

#### `average(tf_list, weights="coherence")`

Average multiple transfer functions.

**Parameters:**
- `tf_list` (list[xr.Dataset]): Transfer functions to average
- `weights` (str): "coherence" or "equal"

**Returns:** `xr.Dataset` - Averaged transfer function

---

### spectraflex.identify

Transfer function identification from time histories.

#### `from_time_histories(wave_elevation, responses, dt, nperseg=1024, noverlap=None, window="hann", detrend="constant")`

Identify transfer functions from time history arrays using Welch's method.

H(f) = S_xy(f) / S_xx(f)

**Parameters:**
- `wave_elevation` (np.ndarray): Wave elevation time history [m]
- `responses` (dict[str, np.ndarray]): Response time histories
- `dt` (float): Sample interval [s]
- `nperseg` (int): FFT segment length
- `noverlap` (int, optional): Overlap points
- `window` (str): Window function
- `detrend` (str): Detrend option

**Returns:** `xr.Dataset` - TransferFunction Dataset

---

#### `from_sim(sim_path, results, nperseg=1024, noverlap=None, window="hann", config=None, wave_position=(0, 0, 0))`

Identify transfer functions from an OrcaFlex .sim file.

**Parameters:**
- `sim_path` (str | Path): Path to .sim file
- `results` (list[dict]): Result specifications with keys: object, variable, arclength (optional), label (optional)
- `nperseg` (int): FFT segment length
- `config` (dict, optional): Configuration metadata
- `wave_position` (tuple): (x, y, z) for wave elevation

**Returns:** `xr.Dataset` - TransferFunction Dataset

**Raises:** `ImportError` if OrcFxAPI not available

---

#### `from_spectra(spectra_path, config=None, freq_range=None)`

Create TransferFunction from pre-computed spectra (.npz file).

**Parameters:**
- `spectra_path` (str | Path): Path to .npz file
- `config` (dict, optional): Configuration metadata
- `freq_range` (tuple, optional): (f_min, f_max) to filter

**Returns:** `xr.Dataset` - TransferFunction Dataset

---

#### `coherence_mask(tf, threshold=0.5)`

Create a boolean mask for frequencies with sufficient coherence.

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `threshold` (float): Minimum coherence value

**Returns:** `xr.DataArray` - Boolean mask

---

#### `apply_coherence_mask(tf, threshold=0.5, fill_value=0.0)`

Set transfer function values to fill_value where coherence is low.

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `threshold` (float): Minimum coherence
- `fill_value` (float): Value for low-coherence frequencies

**Returns:** `xr.Dataset` - Masked Dataset

---

### spectraflex.predict

Spectral response prediction and time series synthesis.

#### `response_spectrum(tf, wave_spectrum, interpolate=True)`

Compute response spectrum: S_yy(f) = |H(f)|² × S_xx(f)

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `wave_spectrum` (xr.DataArray): Wave spectrum S_xx(f)
- `interpolate` (bool): Interpolate wave spectrum to TF frequencies

**Returns:** `xr.Dataset` with Sxx, Syy, H_magnitude_sq

---

#### `statistics(tf, wave_spectrum, duration=10800.0, interpolate=True)`

Compute response statistics for each variable.

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `wave_spectrum` (xr.DataArray): Wave spectrum
- `duration` (float): Duration for MPM calculation [s]
- `interpolate` (bool): Interpolate wave spectrum

**Returns:** `dict[str, dict[str, float]]` - {variable: {statistic: value}}

---

#### `synthesize_timeseries(tf, wave_spectrum, duration, dt, seed=None, interpolate=True)`

Synthesize response time series from transfer function and wave spectrum.

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `wave_spectrum` (xr.DataArray): Wave spectrum
- `duration` (float): Duration [s]
- `dt` (float): Time step [s]
- `seed` (int, optional): Random seed

**Returns:** `xr.Dataset` with wave and response time series

---

#### `synthesize_timeseries_fft(tf, wave_spectrum, n_samples, dt, seed=None)`

Synthesize time series using inverse FFT (faster for long durations).

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `wave_spectrum` (xr.DataArray): Wave spectrum
- `n_samples` (int): Number of samples (use power of 2)
- `dt` (float): Time step [s]
- `seed` (int, optional): Random seed

**Returns:** `xr.Dataset` with wave and response time series

---

#### `cross_check_coherence(tf, wave_spectrum, coherence_threshold=0.5)`

Compute reliability of predictions based on coherence.

**Parameters:**
- `tf` (xr.Dataset): TransferFunction Dataset
- `wave_spectrum` (xr.DataArray): Wave spectrum
- `coherence_threshold` (float): Threshold for good coherence

**Returns:** `xr.DataArray` - Weighted coherence per variable

---

### spectraflex.library

#### `class TransferFunctionLibrary`

Collection of transfer functions across operating configurations.

**Constructor:**
```python
TransferFunctionLibrary(config_keys=None)
```

**Properties:**
- `config_keys` (list[str]): Configuration parameter names
- `configs` (list[dict]): List of configurations
- `datasets` (list[xr.Dataset]): List of TransferFunction datasets

**Methods:**

##### `add(tf, config=None)`
Add a transfer function to the library.

##### `select(**kwargs)`
Select by exact config match. Returns `xr.Dataset`.

##### `lookup(method="nearest", **kwargs)`
Look up by nearest-neighbour or interpolation. Methods: "nearest", "linear".

##### `filter(**kwargs)`
Return a new library with matching configs.

##### `save(path)`
Save to NetCDF file.

##### `load(path)` (classmethod)
Load from NetCDF file.

##### `get_config_range(key)`
Get (min, max) for a config parameter.

##### `get_unique_values(key)`
Get sorted unique values for a config parameter.

##### `to_dataset()`
Combine all TFs into single Dataset with config dimension.

---

### spectraflex.io

I/O utilities for TransferFunction datasets and libraries.

#### `save_transfer_function(tf, path, engine="netcdf4")`

Save a TransferFunction Dataset to NetCDF.

#### `load_transfer_function(path, engine=None)`

Load a TransferFunction Dataset from NetCDF.

#### `save_library(library, path, engine="netcdf4")`

Save a TransferFunctionLibrary to NetCDF.

#### `load_library(path, engine=None)`

Load a TransferFunctionLibrary from NetCDF.

#### `save_spectra(path, frequency, sxx, syy, sxy, variable_names, **attrs)`

Save raw spectra to .npz file.

#### `load_spectra(path)`

Load raw spectra from .npz file. Returns dict with frequency, Sxx, Syy, Sxy, variable_names.

---

## OrcaFlex Integration

### spectraflex.orcaflex.white_noise

White noise model generation for OrcaFlex.

#### `generate(template, hs, freq_range, duration=512.0, wave_direction=0.0, output_dir=".", format="yml", ...)`

Generate a single white noise model file (YAML variation).

**Parameters:**
- `template` (Path): Base OrcaFlex model
- `hs` (float): Significant wave height [m]
- `freq_range` (tuple): (min_freq, max_freq) in Hz
- `duration` (float): Simulation duration [s]
- `wave_direction` (float): Wave direction [deg]
- `output_dir` (Path): Output directory
- `format` (str): "yml" or "dat"

**Returns:** `Path` - Generated file path

---

#### `generate_batch(template, matrix, freq_range, duration=512.0, output_dir=".", format="yml")`

Generate multiple white noise model files.

**Parameters:**
- `template` (Path): Base OrcaFlex model
- `matrix` (dict): Parameter matrix, e.g., {"hs": [1, 2, 4], "wave_direction": [0, 45, 90]}
- `freq_range` (tuple): Frequency range
- Other parameters same as `generate()`

**Returns:** `list[Path]` - Generated file paths

---

#### `get_case_config(path)`

Parse configuration from filename.

**Parameters:**
- `path` (Path): File path with encoded config (e.g., "model_Hs2.0_Dir45.yml")

**Returns:** `dict` - Parsed configuration

---

### spectraflex.orcaflex.batch

Batch generation utilities.

#### `class CaseConfig`

Configuration dataclass for a simulation case.

**Attributes:**
- `hs` (float): Significant wave height
- `wave_direction` (float): Wave direction
- `current_speed` (float): Current speed
- `current_direction` (float): Current direction
- `extra` (dict): Additional parameters
- `label` (str): Auto-generated label

---

#### `generate_case_matrix(hs, wave_direction, current_speed, current_direction=None, **extra)`

Generate all combinations of parameters.

**Returns:** `list[CaseConfig]`

---

#### `config_from_filename(path)`

Parse config from filename.

---

#### `match_spectra_to_configs(spectra_files, cases)`

Match spectra files to case configurations.

---

#### `find_completed_sims(directory)`

Find .sim files in directory.

---

#### `find_spectra_files(directory)`

Find *_spectra.npz files in directory.

---

#### `get_batch_status(directory, expected_cases=None)`

Get batch processing status (counts, completion percentage).

---

### spectraflex.orcaflex.post_calc

Post-calculation action script generation.

#### `get_post_calc_script(results, nperseg=1024, noverlap=None, window="hann", wave_position=(0, 0, 0))`

Generate Python script for OrcaFlex post-calculation action.

**Parameters:**
- `results` (list[dict]): Result specifications
- `nperseg` (int): FFT segment length
- `noverlap` (int, optional): Overlap
- `window` (str): Window function
- `wave_position` (tuple): Wave elevation position

**Returns:** `str` - Python script content

---

#### `write_standalone_script(path, results, **kwargs)`

Write post-calculation script to file.

---

#### `attach_post_calc(model, results, script_path=None, **kwargs)`

Attach post-calculation action to OrcaFlex model.

**Requires:** OrcFxAPI

---

### spectraflex.orcaflex.extract

Time history extraction from OrcaFlex simulations.

#### `get_analysis_period(model)`

Get (start, end) times for analysis period (excluding build-up).

#### `get_sample_interval(model, period=None)`

Get sample interval from model.

#### `extract_wave_elevation(model, position=(0, 0, 0), period=None)`

Extract wave elevation time history.

#### `extract_time_histories(model, results, period=None)`

Extract multiple time histories efficiently.

#### `extract_from_sim(sim_path, results, wave_position=(0, 0, 0))`

Extract wave and response time histories from .sim file.

#### `list_available_results(model, object_name)`

List available result variables for an object.

---

## Command-Line Interface

```bash
spectraflex --version
spectraflex --help
```

### Commands

#### `spectraflex identify`

Identify transfer functions from spectra files.

```bash
spectraflex identify input.npz -o output.nc --config '{"hs": 2.0}'
```

#### `spectraflex predict`

Predict response statistics.

```bash
spectraflex predict tf.nc --hs 3.0 --tp 10.0 --gamma 3.3 --duration 10800 -o results.json
```

#### `spectraflex generate`

Generate white noise model files.

```bash
spectraflex generate template.dat --hs 1.0 2.0 4.0 --direction 0 45 90 -o ./models/
```

#### `spectraflex library info`

Show library information.

```bash
spectraflex library info library.nc
```

#### `spectraflex library build`

Build library from spectra files.

```bash
spectraflex library build ./spectra_dir/ -o library.nc
```
