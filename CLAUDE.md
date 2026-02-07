# CLAUDE.md — spectraflex

## Project Overview

`spectraflex` identifies transfer functions from OrcaFlex white noise simulations
and uses them for fast spectral response prediction. See `SPECTRAFLEX_DESIGN.md`
for the full architecture and API design.

## Quick Reference

```bash
# Setup
uv venv
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
pyright src/
```

## Project Conventions

### Python Style
- Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- `from __future__ import annotations` in every module
- Type hints on all public functions
- Docstrings: numpy-style
- Use `dataclass` or `pydantic.BaseModel` for simple data containers
- Use `xarray.Dataset` for spectral data (transfer functions, spectra)

### Tooling
- **Package manager**: uv
- **Build system**: hatchling (pyproject.toml, src layout)
- **Linter/formatter**: ruff
- **Testing**: pytest with pytest-cov
- **Type checking**: pyright (strict mode on src/)

### File Layout
```
src/spectraflex/          # All source code
  ├── __init__.py         # Package exports
  ├── transfer_function.py # TransferFunction xarray Dataset factory
  ├── spectrum.py         # Wave spectrum definitions (JONSWAP, PM)
  ├── statistics.py       # Spectral moments, Hs, MPM calculations
  ├── identify.py         # Transfer function identification
  ├── predict.py          # Spectral response prediction
  ├── library.py          # TransferFunctionLibrary collection
  ├── io/                 # I/O utilities
  │   ├── __init__.py
  │   └── netcdf.py       # NetCDF save/load
  └── orcaflex/           # OrcaFlex integration
      ├── __init__.py
      ├── white_noise.py  # White noise model generation
      ├── post_calc.py    # Post-calculation action script
      ├── extract.py      # Time history extraction
      └── batch.py        # Batch generation utilities
tests/                    # All tests
examples/                 # Example scripts (not tested in CI)
SPECTRAFLEX_DESIGN.md     # Architecture & API design doc
```

### Import Conventions
```python
import numpy as np
import xarray as xr
from scipy import signal
import OrcFxAPI as ofx      # only in orcaflex/ subpackage
```

### Commit Messages
- `feat: add spectrum.jonswap()`
- `test: add tests for statistics module`
- `fix: correct m2 calculation in spectral_moments`
- `docs: update SPECTRAFLEX_DESIGN.md`

## Implementation Status

### Phase 1 — Core (Complete)
- [x] `transfer_function.py` — TransferFunction xarray Dataset factory & validation
- [x] `spectrum.py` — jonswap(), pierson_moskowitz(), from_array(), white_noise()
- [x] `statistics.py` — spectral_moments(), hs_from_spectrum(), mpm_rayleigh()
- [x] `identify.py` — from_sim(), from_time_histories(), from_spectra()
- [x] `predict.py` — response_spectrum(), statistics(), synthesize_timeseries()

### Phase 2 — Library & Batch (Complete)
- [x] `library.py` — TransferFunctionLibrary with add, select, lookup, save/load
- [x] `io/netcdf.py` — save/load for TransferFunction and Library
- [x] `orcaflex/white_noise.py` — generate(), generate_batch() for YAML files
- [x] `orcaflex/post_calc.py` — bundled post-calc action script + attach()
- [x] `orcaflex/extract.py` — extract time histories from .sim files
- [x] `orcaflex/batch.py` — batch matrix generation utilities

### Phase 3 — Synthesis & Polish (Pending)
- [x] `predict.synthesize_timeseries()` — spectral synthesis (already implemented)
- [ ] `library.lookup()` with interpolation — basic IDW implemented, may need refinement
- [ ] CLI entry points
- [ ] Examples and documentation

### Phase 4 — ML Integration (Pending)
- [ ] `ml/surrogate.py` — interface definition
- [ ] Integration with enigma for H(f) prediction
- [ ] Training workflow from library data

## Implementation Notes

### TransferFunction is an xarray.Dataset

Do not create a class wrapper around xarray.Dataset. Instead, provide
factory functions that create validated Datasets and standalone functions
that operate on them:

```python
# transfer_function.py

def create(
    frequency: np.ndarray,
    magnitude: np.ndarray,    # shape: (n_freq, n_var)
    phase: np.ndarray,        # shape: (n_freq, n_var)
    coherence: np.ndarray,    # shape: (n_freq, n_var)
    variable_names: list[str],
    config: dict | None = None,
    **attrs,
) -> xr.Dataset:
    """Create a validated TransferFunction Dataset."""
    ...

def validate(ds: xr.Dataset) -> None:
    """Raise ValueError if Dataset doesn't conform to TransferFunction schema."""
    ...
```

### Testing Without OrcFxAPI

Most of the package is testable without OrcaFlex. The round-trip test pattern:

```python
def test_identify_recovers_known_transfer_function():
    """Generate synthetic data from a known H(f), identify, compare."""
    # 1. Define a known transfer function
    f = np.linspace(0.01, 0.5, 256)
    H_true = 5.0 / (1.0 + 1j * (f / 0.1 - 0.1 / f))  # simple resonance

    # 2. Generate white noise input
    rng = np.random.default_rng(42)
    dt = 0.1
    N = 5120
    wave = rng.normal(0, 1, N)

    # 3. Filter through H(f) to create response
    # (use scipy.signal.sosfilt or frequency-domain filtering)
    ...

    # 4. Identify H(f) from input/output
    tf = identify.from_time_histories(wave, {"response": response}, dt)

    # 5. Compare with known H(f)
    np.testing.assert_allclose(tf.magnitude.values[:, 0], np.abs(H_true), rtol=0.1)
```

### OrcaFlex Documentation

When implementing OrcaFlex-specific code, fetch and read the documentation
pages listed in SPECTRAFLEX_DESIGN.md under "OrcaFlex Documentation References".
Key URLs:

- API reference: https://www.orcina.com/webhelp/OrcFxAPI/Content/html/Pythonreference,OrcFxAPIModule.htm
- Results extraction: https://www.orcina.com/webhelp/OrcFxAPI/Content/html/Pythoninterface,Results.htm
- Post-calc actions: https://www.orcina.com/webhelp/OrcaFlex/Content/html/Generaldata,Postcalculationactions.htm
- YAML variation files: https://www.orcina.com/webhelp/OrcaFlex/Content/html/Textdatafiles,Examplesofsettingdata.htm
- Variation models: https://www.orcina.com/webhelp/OrcaFlex/Content/html/Variationmodels.htm

### Key OrcFxAPI Patterns

```python
import OrcFxAPI as ofx

# Load simulation
model = ofx.Model("simulation.sim")

# Analysis period (skip build-up stage 0)
t_start = model.general.StageDuration[0]
t_end = t_start + model.general.StageDuration[1]
period = ofx.SpecifiedPeriod(t_start, t_end)

# Wave elevation
wave = model.environment.TimeHistory("Elevation", period, ofx.oeEnvironment(0, 0, 0))

# Batch extract multiple results efficiently
specs = [
    ofx.TimeHistorySpecification(model["Riser"], "Effective Tension", ofx.oeArcLength(0.0)),
    ofx.TimeHistorySpecification(model["Riser"], "Rotation 1", ofx.oeArcLength(0.0)),
    ofx.TimeHistorySpecification(model["Vessel"], "X", None),
]
all_th = ofx.GetMultipleTimeHistories(specs, period)  # 2D array: (time, variables)

# Sample interval
sample_times = model.environment.SampleTimes(period)
dt = float(sample_times[1] - sample_times[0])
```

### YAML vs Python API Stage Indexing

This is a common gotcha:
- **Python API**: `StageDuration[0]` = build-up, `StageDuration[1]` = main sim (0-based)
- **YAML text files**: `StageDuration[1]` = build-up, `StageDuration[2]` = main sim (1-based)

Be very careful about this when generating YAML variation files in `orcaflex/white_noise.py`.
