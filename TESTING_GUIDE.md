# Testing Guide for spectraflex

This document provides guidance for the test agent implementing tests for spectraflex.

## Test Structure

All tests go in `tests/`. Import directly from submodules:

```python
from spectraflex.spectrum import jonswap, pierson_moskowitz
from spectraflex.statistics import spectral_moments, hs_from_spectrum, mpm_rayleigh
from spectraflex.transfer_function import create, validate
from spectraflex.identify import from_time_histories
from spectraflex.predict import response_spectrum, statistics, synthesize_timeseries
from spectraflex.library import TransferFunctionLibrary
```

## Test Priority Order

### 1. test_spectrum.py (no OrcFxAPI needed)
- Verify JONSWAP spectrum shape and peak location
- Verify Pierson-Moskowitz is JONSWAP with gamma=1
- Verify integration: `4 * sqrt(∫S(f)df)` equals input Hs
- Test `white_noise()` spectrum is flat in specified range
- Test `scale_to_hs()` correctly rescales
- Edge cases: very small Hs, extreme gamma values

### 2. test_statistics.py (no OrcFxAPI needed)
- Known spectral moments against analytical values
- `hs_from_spectrum()`: verify 4*sqrt(m0)
- `tp_from_spectrum()`: verify peak detection
- `tz_from_spectrum()`: verify sqrt(m0/m2)
- `mpm_rayleigh()`: verify against known formula
- Test with synthetic spectra where moments are analytically known

### 3. test_transfer_function.py (no OrcFxAPI needed)
- `create()`: valid inputs produce valid Dataset
- `create()`: invalid inputs raise ValueError (negative frequencies, bad shapes, coherence outside [0,1])
- `validate()`: correctly identifies valid/invalid Datasets
- `from_complex()`: magnitude and phase extracted correctly
- `select_variables()` and `select_frequency_range()` work correctly
- Verify xarray structure: dims, coords, data_vars, attrs

### 4. test_predict.py (no OrcFxAPI needed)
- Known H(f) + known S_xx → verify S_yy = |H|² * S_xx
- `response_statistics()`: verify moments of response spectrum
- `synthesize_timeseries()`: verify output has correct statistical properties
- `synthesize_timeseries_fft()`: compare with direct method
- Test interpolation when frequencies don't match exactly

### 5. test_identify.py (no OrcFxAPI needed) - KEY TEST
The round-trip test is the most important validation:

```python
import numpy as np
from scipy import signal
from spectraflex import identify

def test_identify_recovers_known_transfer_function():
    """Generate synthetic data from a known H(f), identify, compare."""
    # 1. Define a known transfer function (simple resonance)
    f = np.linspace(0.01, 0.5, 256)
    f0 = 0.1  # resonance frequency
    Q = 5.0   # quality factor
    H_true = 1.0 / (1.0 + 1j * Q * (f / f0 - f0 / f))

    # 2. Generate white noise input
    rng = np.random.default_rng(42)
    dt = 0.1
    fs = 1.0 / dt
    N = 10240  # long enough for good spectral estimates
    wave = rng.normal(0, 1, N)

    # 3. Filter through H(f) to create response using frequency-domain filtering
    wave_fft = np.fft.rfft(wave)
    fft_freq = np.fft.rfftfreq(N, dt)

    # Interpolate H to FFT frequencies
    H_interp = np.interp(fft_freq, f, H_true)
    response_fft = wave_fft * H_interp
    response = np.fft.irfft(response_fft, n=N)

    # 4. Identify H(f) from input/output
    tf = identify.from_time_histories(
        wave_elevation=wave,
        responses={"response": response},
        dt=dt,
        nperseg=1024,
    )

    # 5. Compare with known H(f) - interpolate to same frequencies
    tf_freq = tf.coords["frequency"].values
    H_true_interp = np.interp(tf_freq, f, np.abs(H_true))

    # Allow 10-20% tolerance due to spectral estimation variance
    np.testing.assert_allclose(
        tf["magnitude"].values[:, 0],
        H_true_interp,
        rtol=0.2
    )
```

### 6. test_library.py (no OrcFxAPI needed)
- `add()`: adds transfer functions correctly
- `add()`: rejects duplicate configs
- `add()`: validates config keys match
- `select()`: exact match works
- `select()`: raises KeyError for no match
- `lookup(method="nearest")`: finds closest config
- `lookup(method="linear")`: interpolates correctly
- `save()`/`load()`: round-trip preserves data
- `filter()`: returns correct subset
- `to_dataset()`: combines correctly

### 7. test_io.py (no OrcFxAPI needed)
- `save_transfer_function()`/`load_transfer_function()`: round-trip
- `save_library()`/`load_library()`: round-trip
- `save_spectra()`/`load_spectra()`: round-trip for .npz files

### 8. test_orcaflex_white_noise.py (no OrcFxAPI needed for YAML)
- `generate()`: creates valid YAML file
- `generate()`: YAML contains correct WaveType, WaveHs, frequencies
- `generate_batch()`: creates correct number of files
- `get_case_config()`: parses filenames correctly

### 9. test_orcaflex_batch.py (no OrcFxAPI needed)
- `generate_case_matrix()`: correct number of combinations
- `generate_case_matrix()`: current_direction follows wave_direction when None
- `CaseConfig`: to_dict() and label generation work
- `config_from_filename()`: parses correctly
- `match_spectra_to_configs()`: matches correctly

### 10. test_orcaflex_extract.py (requires OrcFxAPI)
Mark with `@pytest.mark.orcaflex` - skip in CI without licence.

### 11. test_orcaflex_post_calc.py (partial OrcFxAPI)
- `get_post_calc_script()`: generates valid Python script (no licence needed)
- `attach_post_calc()`: requires OrcFxAPI, mark with `@pytest.mark.orcaflex`

## pytest Configuration

Add to `conftest.py`:

```python
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "orcaflex: marks tests as requiring OrcFxAPI (deselect with '-m \"not orcaflex\"')"
    )

@pytest.fixture
def sample_frequency():
    """Standard frequency array for tests."""
    import numpy as np
    return np.linspace(0.01, 0.5, 256)

@pytest.fixture
def sample_transfer_function(sample_frequency):
    """Sample TransferFunction Dataset for tests."""
    import numpy as np
    from spectraflex import transfer_function

    n_freq = len(sample_frequency)
    return transfer_function.create(
        frequency=sample_frequency,
        magnitude=np.ones((n_freq, 2)),
        phase=np.zeros((n_freq, 2)),
        coherence=np.ones((n_freq, 2)) * 0.9,
        variable_names=["var1", "var2"],
        config={"hs": 2.0, "heading": 0.0},
    )
```

## Running Tests

```bash
# All tests (excluding OrcaFlex-dependent)
pytest tests/ -v -m "not orcaflex"

# All tests (if OrcFxAPI available)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=spectraflex --cov-report=html
```

## Key Validation Points

1. **Spectrum integration**: `4 * sqrt(∫S(f)df)` must equal input Hs
2. **Transfer function recovery**: The round-trip test must recover H(f) within ~20% tolerance
3. **Response spectrum**: `S_yy = |H|² * S_xx` must hold exactly
4. **Statistics consistency**: `hs = 4*sqrt(m0)` must be exact
5. **Library persistence**: Save/load must preserve all data exactly
6. **YAML generation**: Must use 1-based indexing for StageDuration
