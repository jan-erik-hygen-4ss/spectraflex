# spectraflex

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: pyright](https://img.shields.io/badge/type%20checked-pyright-blue.svg)](https://github.com/microsoft/pyright)

**Transfer function identification and spectral response prediction for OrcaFlex simulations**

spectraflex enables fast spectral analysis of offshore structures by identifying transfer functions from white noise simulations and using them to predict response statistics for arbitrary sea states.

## Why spectraflex?

Traditional offshore analysis requires running time-domain simulations for every sea state of interest. This is computationally expensive when evaluating many environmental conditions.

spectraflex takes a different approach:

1. **Run once**: Perform a single white noise simulation that excites all frequencies
2. **Identify H(f)**: Extract the transfer function using cross-spectral analysis
3. **Predict instantly**: Calculate response statistics for any wave spectrum in milliseconds

This reduces computation time from hours to seconds when screening many environmental conditions.

## Core Workflow

```
+------------------------------------------------------------------+
|  Phase 1: Simulation                                             |
|  Template -> White Noise Models -> Run Simulations -> Spectra    |
+------------------------------------------------------------------+
|  Phase 2: Identification                                         |
|  Spectra -> Transfer Functions -> Library                        |
+------------------------------------------------------------------+
|  Phase 3: Prediction                                             |
|  Library + Wave Spectrum -> Response Statistics / Time Series    |
+------------------------------------------------------------------+
```

## Features

- **Wave spectra**: JONSWAP, Pierson-Moskowitz, white noise, and custom spectra
- **Spectral statistics**: Moments (m0, m1, m2, m4), Hs, Tz, bandwidth, MPM
- **Transfer function identification**: H1 estimator with coherence quality metrics
- **Response prediction**: Spectrum prediction, statistics, time series synthesis
- **Spectral fatigue**: DNV S-N curves, Dirlik and narrow-band damage calculation
- **Library management**: Store and query transfer functions by operating conditions
- **OrcaFlex integration**: Generate white noise models, extract results, batch processing
- **CLI**: Command-line interface for common workflows

## Installation

```bash
pip install spectraflex
```

For NetCDF support (saving/loading transfer functions):

```bash
pip install spectraflex[io]
```

For plotting:

```bash
pip install spectraflex[plot]
```

### OrcaFlex Integration

OrcaFlex integration requires a licensed installation of OrcFxAPI:

```bash
pip install OrcFxAPI
```

## Quick Start

### Python API

```python
from spectraflex import spectrum, identify, predict

# 1. Identify transfer function from simulation spectra
tf = identify.from_spectra("white_noise_spectra.npz")

# 2. Create wave spectrum for prediction
f = tf.coords["frequency"].values
wave = spectrum.jonswap(hs=5.0, tp=12.0, f=f)

# 3. Predict response statistics
stats = predict.statistics(tf, wave, duration=10800)
print(f"Roll MPM: {stats['Roll']['mpm']:.2f} deg")
```

### Command-Line Interface

```bash
# Generate white noise models from template
spectraflex generate template.dat --hs 2 3 4 --direction 0 45 90

# Identify transfer function from spectra
spectraflex identify spectra.npz -o transfer_function.nc

# Predict response for a sea state
spectraflex predict transfer_function.nc --hs 5.0 --tp 12.0 --duration 10800

# Build a library from multiple spectra files
spectraflex library build ./spectra/ -o library.nc
```

## Documentation

- [Theory](docs/theory.md) - Mathematical foundations of spectral analysis
- [Usage Guide](docs/usage_guide.md) - Comprehensive guide to using spectraflex
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Tutorials](docs/tutorials/) - Step-by-step learning guides:
  1. [Getting Started](docs/tutorials/01_getting_started.md)
  2. [Transfer Function Identification](docs/tutorials/02_transfer_function_identification.md)
  3. [Transfer Function Libraries](docs/tutorials/03_transfer_function_libraries.md)
  4. [Complete Workflow](docs/tutorials/04_complete_workflow.md)

## Key Concepts

### Transfer Function

The frequency-domain relationship between wave input and structural response:

```
H(f) = Response(f) / Wave(f)
```

For power spectra: `S_yy(f) = |H(f)|^2 * S_xx(f)`

### White Noise Identification

White noise has constant spectral density, exciting all frequencies equally. This allows identification of H(f) at all frequencies from a single simulation using cross-spectral analysis:

```
H(f) = S_xy(f) / S_xx(f)
```

### Coherence

The coherence function measures reliability of the identified transfer function:

- gamma^2 near 1: Reliable (strong linear relationship)
- gamma^2 near 0: Unreliable (noise or nonlinearity)

### Spectral Fatigue

Calculate fatigue damage directly from transfer functions using DNV S-N curves:

```python
from spectraflex import fatigue, spectrum

# Define S-N curve (DNV-D for fillet welds)
sn_curve = fatigue.SNCurve.dnv_d()

# Calculate damage from stress transfer function
result = fatigue.damage_from_transfer_function(
    tf=stress_tf,              # Transfer function in MPa/m
    wave_spectrum=wave,        # Wave spectrum
    sn_curve=sn_curve,
    exposure_time=3600 * 24,   # 1 day in seconds
    method="dirlik",           # or "narrow_band"
)

print(f"Fatigue damage: {result['damage']:.2e}")
print(f"Fatigue life: {result['life_seconds'] / 3600 / 24 / 365:.1f} years")
```

## Project Structure

```
src/spectraflex/
├── spectrum.py          # Wave spectrum definitions (JONSWAP, PM, white noise)
├── statistics.py        # Spectral moments and derived statistics
├── transfer_function.py # TransferFunction Dataset creation and validation
├── identify.py          # Transfer function identification
├── predict.py           # Response prediction
├── fatigue.py           # Spectral fatigue (S-N curves, Dirlik, damage)
├── library.py           # TransferFunctionLibrary for managing collections
├── cli.py               # Command-line interface
├── io/                  # I/O utilities (NetCDF, NPZ)
└── orcaflex/            # OrcaFlex integration
    ├── white_noise.py   # White noise model generation
    ├── extract.py       # Result extraction from .sim files
    ├── post_calc.py     # Post-calculation action scripts
    └── batch.py         # Batch processing utilities
```

## Development

```bash
# Clone and setup
git clone https://github.com/spectraflex/spectraflex.git
cd spectraflex
uv venv
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run pyright src/
```

## Requirements

- Python >= 3.13
- NumPy, SciPy, xarray
- OrcFxAPI (optional, for OrcaFlex integration)
- NetCDF4 or h5netcdf (optional, for file I/O)

## License

See [LICENSE](LICENSE) for details.
