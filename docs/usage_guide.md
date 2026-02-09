# Usage Guide

This guide covers the main workflows for using spectraflex to identify transfer functions from OrcaFlex simulations and predict spectral responses.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [Workflow Overview](#workflow-overview)
5. [Working with Wave Spectra](#working-with-wave-spectra)
6. [Transfer Function Identification](#transfer-function-identification)
7. [Response Prediction](#response-prediction)
8. [Managing Transfer Function Libraries](#managing-transfer-function-libraries)
9. [OrcaFlex Integration](#orcaflex-integration)
10. [Command-Line Interface](#command-line-interface)
11. [Best Practices](#best-practices)

---

## Installation

```bash
pip install spectraflex

# For OrcaFlex integration (requires OrcaFlex licence)
pip install OrcFxAPI
```

**Dependencies:**
- numpy
- scipy
- xarray
- netcdf4

---

## Core Concepts

### Transfer Functions

A transfer function H(f) describes how a system transforms an input signal (wave elevation) to an output signal (structural response) in the frequency domain:

```
Response(f) = H(f) × Wave(f)
```

For power spectra:
```
S_yy(f) = |H(f)|² × S_xx(f)
```

### White Noise Identification

The key insight is that a **white noise input** (flat spectrum) allows direct identification of H(f) from input/output measurements:

```
H(f) = S_xy(f) / S_xx(f)
```

where S_xy is the cross-spectrum between wave elevation and response.

### Coherence

Coherence γ²(f) measures how linear the input-output relationship is at each frequency:

```
γ²(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))
```

- γ² ≈ 1: Strong linear relationship (reliable H(f))
- γ² ≈ 0: No linear relationship (unreliable H(f))

---

## Quick Start

```python
import numpy as np
from spectraflex import spectrum, identify, predict
from spectraflex.io import load_transfer_function, save_transfer_function

# 1. Identify transfer function from pre-computed spectra
tf = identify.from_spectra("simulation_spectra.npz")

# 2. Create a wave spectrum for prediction
f = tf.coords["frequency"].values
wave = spectrum.jonswap(hs=4.0, tp=12.0, f=f)

# 3. Predict response statistics
stats = predict.statistics(tf, wave, duration=10800)

for var, var_stats in stats.items():
    print(f"{var}:")
    print(f"  Hs: {var_stats['hs']:.2f}")
    print(f"  MPM: {var_stats['mpm']:.2f}")
```

---

## Workflow Overview

The typical spectraflex workflow consists of three phases:

### Phase 1: Simulation Setup (with OrcaFlex)

```
Template Model (.dat)
        ↓
   [generate YAML]
        ↓
White Noise Models (.yml)
        ↓
   [run simulations]
        ↓
Completed Sims (.sim)
```

### Phase 2: Transfer Function Identification

```
Simulation Results (.sim or .npz)
        ↓
     [identify]
        ↓
Transfer Functions (.nc)
        ↓
     [collect]
        ↓
TF Library (.nc)
```

### Phase 3: Response Prediction

```
TF Library + Wave Spectrum
        ↓
      [predict]
        ↓
Response Statistics / Time Series
```

---

## Working with Wave Spectra

### Creating Standard Spectra

```python
import numpy as np
from spectraflex import spectrum

# Create frequency array
f = spectrum.frequency_array(f_min=0.02, f_max=0.3, n_freq=256)

# JONSWAP spectrum (developing sea)
wave_jonswap = spectrum.jonswap(hs=3.0, tp=10.0, f=f, gamma=3.3)

# Pierson-Moskowitz spectrum (fully developed sea)
wave_pm = spectrum.pierson_moskowitz(hs=3.0, tp=10.0, f=f)

# White noise spectrum (for identification)
wave_white = spectrum.white_noise(hs=3.0, f=f, f_min=0.02, f_max=0.25)
```

### Custom Spectra

```python
# From arrays
measured_f = np.array([0.05, 0.08, 0.10, 0.12, 0.15])
measured_s = np.array([0.1, 0.5, 1.2, 0.8, 0.3])

wave = spectrum.from_array(measured_f, measured_s)

# Scale to different Hs
wave_scaled = spectrum.scale_to_hs(wave, hs=5.0)
```

### Spectrum Properties

```python
from spectraflex import statistics

# Get all statistics
stats = statistics.all_statistics(f, wave.values)
print(f"Hs: {stats['hs']:.2f} m")
print(f"Tp: {stats['tp']:.2f} s")
print(f"Tz: {stats['tz']:.2f} s")

# Individual functions
hs = statistics.hs_from_spectrum(f, wave.values)
tp = statistics.tp_from_spectrum(f, wave.values)
moments = statistics.spectral_moments(f, wave.values)
```

---

## Transfer Function Identification

### From Pre-computed Spectra (.npz)

This is the most common workflow when using post-calculation actions:

```python
from spectraflex import identify

# Basic identification
tf = identify.from_spectra("model_Hs2.0_Dir0_spectra.npz")

# With configuration metadata
tf = identify.from_spectra(
    "model_Hs2.0_Dir0_spectra.npz",
    config={"hs": 2.0, "heading": 0.0, "draft": 21.0}
)

# Filter to valid frequency range
tf = identify.from_spectra(
    "model_spectra.npz",
    freq_range=(0.02, 0.25)  # Only use frequencies with white noise energy
)
```

### From Time History Arrays

For custom analysis or testing:

```python
import numpy as np
from spectraflex import identify

# Load or generate time histories
dt = 0.1  # Sample interval [s]
wave = np.random.randn(10000)  # Wave elevation
roll = np.random.randn(10000)  # Roll response
pitch = np.random.randn(10000)  # Pitch response

# Identify transfer functions
tf = identify.from_time_histories(
    wave_elevation=wave,
    responses={
        "Roll": roll,
        "Pitch": pitch,
    },
    dt=dt,
    nperseg=1024,
    window="hann",
)

# Access results
print(f"Frequencies: {len(tf.coords['frequency'])} points")
print(f"Variables: {list(tf.coords['variable'].values)}")
```

### From OrcaFlex .sim Files

Requires OrcFxAPI and an OrcaFlex licence:

```python
from spectraflex import identify

results = [
    {"object": "Vessel", "variable": "Rotation 1", "label": "Roll"},
    {"object": "Vessel", "variable": "Rotation 2", "label": "Pitch"},
    {"object": "Riser", "variable": "Effective Tension", "arclength": 0.0, "label": "TopTension"},
]

tf = identify.from_sim(
    sim_path="completed_simulation.sim",
    results=results,
    nperseg=2048,
    config={"hs": 2.0, "heading": 45.0},
)
```

### Examining Transfer Functions

```python
# Check coherence (quality indicator)
print("Mean coherence per variable:")
for var in tf.coords["variable"].values:
    mean_coh = tf["coherence"].sel(variable=var).mean().values
    print(f"  {var}: {mean_coh:.3f}")

# Get complex transfer function
from spectraflex import transfer_function
H = transfer_function.complex_transfer_function(tf)

# Select frequency range
tf_low = transfer_function.select_frequency_range(tf, f_min=0.02, f_max=0.1)

# Select specific variables
tf_vessel = transfer_function.select_variables(tf, ["Roll", "Pitch"])
```

### Masking Low-Coherence Regions

```python
from spectraflex import identify

# Create mask for coherence > 0.5
mask = identify.coherence_mask(tf, threshold=0.5)

# Apply mask (set low-coherence values to 0)
tf_masked = identify.apply_coherence_mask(tf, threshold=0.5)
```

---

## Response Prediction

### Basic Prediction

```python
from spectraflex import spectrum, predict

# Load transfer function
from spectraflex.io import load_transfer_function
tf = load_transfer_function("transfer_function.nc")

# Create wave spectrum
f = tf.coords["frequency"].values
wave = spectrum.jonswap(hs=5.0, tp=14.0, f=f, gamma=2.5)

# Compute response spectrum
resp = predict.response_spectrum(tf, wave)
print(f"Response spectrum shape: {resp['Syy'].shape}")
```

### Response Statistics

```python
# Get statistics for 3-hour storm
stats = predict.statistics(tf, wave, duration=10800)

for var, s in stats.items():
    print(f"\n{var}:")
    print(f"  Hs (4√m₀):  {s['hs']:.3f}")
    print(f"  σ (√m₀):    {s['sigma']:.3f}")
    print(f"  Tz:         {s['tz']:.2f} s")
    print(f"  Tp:         {s['tp']:.2f} s")
    print(f"  MPM:        {s['mpm']:.3f}")
```

### Synthesize Time Series

```python
# Direct summation method
ts = predict.synthesize_timeseries(
    tf, wave,
    duration=600,  # 10 minutes
    dt=0.1,        # 10 Hz
    seed=42,       # For reproducibility
)

# Access results
time = ts.coords["time"].values
wave_ts = ts["wave"].values
roll_ts = ts["Roll"].values

# FFT method (faster for long durations)
ts_fft = predict.synthesize_timeseries_fft(
    tf, wave,
    n_samples=8192,  # Use power of 2
    dt=0.1,
    seed=42,
)
```

### Check Prediction Reliability

```python
# Get coherence-weighted reliability score
reliability = predict.cross_check_coherence(tf, wave)

for var in reliability.coords["variable"].values:
    score = reliability.sel(variable=var).values
    print(f"{var}: {score:.2f}")
```

---

## Managing Transfer Function Libraries

### Creating a Library

```python
from spectraflex import TransferFunctionLibrary, identify

lib = TransferFunctionLibrary()

# Add transfer functions from multiple conditions
for hs in [1.0, 2.0, 4.0]:
    for heading in [0.0, 45.0, 90.0]:
        tf = identify.from_spectra(
            f"spectra/model_Hs{hs}_Dir{heading}_spectra.npz",
            config={"hs": hs, "heading": heading}
        )
        lib.add(tf)

print(lib)  # TransferFunctionLibrary(n_configs=9, config_keys=['heading', 'hs'])
```

### Selecting from Library

```python
# Exact match
tf = lib.select(hs=2.0, heading=45.0)

# Nearest neighbor lookup
tf_nearest = lib.lookup(hs=2.5, heading=40.0, method="nearest")

# Interpolated lookup
tf_interp = lib.lookup(hs=2.5, heading=40.0, method="linear")
```

### Querying Library Contents

```python
# Get parameter ranges
hs_min, hs_max = lib.get_config_range("hs")
print(f"Hs range: {hs_min} to {hs_max}")

# Get unique values
headings = lib.get_unique_values("heading")
print(f"Headings: {headings}")

# Filter library
lib_stern = lib.filter(heading=180.0)
```

### Saving and Loading

```python
# Save
lib.save("tf_library.nc")

# Load
from spectraflex import TransferFunctionLibrary
lib = TransferFunctionLibrary.load("tf_library.nc")
```

---

## OrcaFlex Integration

### Generating White Noise Models

```python
from spectraflex.orcaflex import white_noise

# Single model
path = white_noise.generate(
    template="base_model.dat",
    hs=2.0,
    freq_range=(0.02, 0.25),
    duration=512,
    wave_direction=45.0,
    output_dir="./models",
)

# Batch generation
paths = white_noise.generate_batch(
    template="base_model.dat",
    matrix={
        "hs": [1.0, 2.0, 4.0],
        "wave_direction": [0.0, 45.0, 90.0, 135.0, 180.0],
    },
    freq_range=(0.02, 0.25),
    duration=512,
    output_dir="./models",
)
print(f"Generated {len(paths)} model files")
```

### Case Matrix Generation

```python
from spectraflex.orcaflex.batch import generate_case_matrix, CaseConfig

# Generate all combinations
cases = generate_case_matrix(
    hs=[1.0, 2.0, 4.0],
    wave_direction=[0.0, 45.0, 90.0],
    current_speed=[0.0, 0.5],
)

print(f"Total cases: {len(cases)}")  # 3 × 3 × 2 = 18

for case in cases[:3]:
    print(f"  {case.label}: Hs={case.hs}, Dir={case.wave_direction}")
```

### Post-Calculation Actions

Generate a Python script that OrcaFlex runs after each simulation:

```python
from spectraflex.orcaflex import post_calc

results = [
    {"object": "Vessel", "variable": "Rotation 1", "label": "Roll"},
    {"object": "Vessel", "variable": "Rotation 2", "label": "Pitch"},
    {"object": "Riser", "variable": "Effective Tension", "arclength": 0.0, "label": "TopTension"},
]

# Generate standalone script
post_calc.write_standalone_script(
    path="post_calc_action.py",
    results=results,
    nperseg=2048,
    window="hann",
)

# Or attach to model (requires OrcFxAPI)
import OrcFxAPI as ofx
model = ofx.Model("base_model.dat")
post_calc.attach_post_calc(model, results, nperseg=2048)
model.SaveData("model_with_postcalc.dat")
```

### Batch Processing Status

```python
from spectraflex.orcaflex.batch import (
    find_completed_sims,
    find_spectra_files,
    get_batch_status,
)

# Check progress
status = get_batch_status("./output", expected_cases=18)
print(f"Simulations completed: {status['n_sims']}")
print(f"Spectra extracted: {status['n_spectra']}")
print(f"Completion: {status['completion']*100:.0f}%")
```

---

## Command-Line Interface

### Generate White Noise Models

```bash
# Single case
spectraflex generate template.dat --hs 2.0 --direction 45 -o ./models/

# Batch
spectraflex generate template.dat \
    --hs 1.0 2.0 4.0 \
    --direction 0 45 90 135 180 \
    --freq-range 0.02 0.25 \
    --duration 512 \
    -o ./models/
```

### Identify Transfer Functions

```bash
spectraflex identify spectra.npz -o transfer_function.nc

# With config
spectraflex identify spectra.npz \
    --config '{"hs": 2.0, "heading": 45.0}' \
    -o tf.nc
```

### Predict Responses

```bash
spectraflex predict tf.nc \
    --hs 5.0 \
    --tp 14.0 \
    --gamma 2.5 \
    --duration 10800 \
    -o results.json
```

### Library Management

```bash
# Show library info
spectraflex library info library.nc

# Build from spectra files
spectraflex library build ./spectra_dir/ -o library.nc
```

---

## Best Practices

### Simulation Parameters

1. **Duration**: Use at least 512s for transfer function identification. Longer durations improve statistical reliability.

2. **Frequency Range**: Match the white noise frequency range to your frequencies of interest. Typical range: 0.02-0.25 Hz (4-50s periods).

3. **nperseg**: Choose based on frequency resolution needs:
   - Higher nperseg = better frequency resolution, but more averaging required
   - Typical values: 1024-4096

4. **Build-up Time**: Allow at least one full period of the lowest frequency for transients to decay.

### Quality Checks

1. **Check Coherence**: Low coherence indicates unreliable transfer function estimates.

```python
# Reject frequencies with coherence < 0.5
tf_good = identify.apply_coherence_mask(tf, threshold=0.5)
```

2. **Compare Across Hs**: For linear systems, H(f) should be similar regardless of Hs.

```python
from spectraflex import transfer_function

stats = transfer_function.compare(tf_hs2, tf_hs4)
if stats['correlation'] < 0.95:
    print("Warning: Non-linear behavior detected")
```

3. **Average Multiple Runs**: Reduce noise by averaging transfer functions from multiple simulations.

```python
tf_avg = transfer_function.average([tf1, tf2, tf3], weights="coherence")
```

### Memory Management

For large libraries, use the library's lazy loading:

```python
# Good: Select specific config
tf = lib.select(hs=2.0, heading=45.0)

# Avoid: Loading all into memory at once
all_tfs = lib.datasets  # Creates copies of all datasets
```

### File Organization

Recommended directory structure:

```
project/
├── models/
│   ├── template.dat
│   └── generated/
│       ├── model_Hs1.0_Dir0.yml
│       └── ...
├── simulations/
│   ├── model_Hs1.0_Dir0.sim
│   └── ...
├── spectra/
│   ├── model_Hs1.0_Dir0_spectra.npz
│   └── ...
└── results/
    └── library.nc
```
