# Tutorial 3: Managing Transfer Function Libraries

This tutorial covers how to organize, store, and query transfer functions across multiple operating conditions using the TransferFunctionLibrary class.

## What You'll Learn

- Creating and populating a transfer function library
- Querying by exact match and interpolation
- Saving and loading libraries
- Batch processing workflows
- Best practices for library organization

## Prerequisites

- Completed Tutorials 1 and 2

## Part 1: Why Use Libraries?

In practice, transfer functions vary with operating conditions:
- **Wave direction (heading)**
- **Vessel draft**
- **Current speed**
- **Mooring configuration**

A **library** stores transfer functions indexed by these configuration parameters, enabling:
- Fast lookup for specific conditions
- Interpolation between conditions
- Systematic organization of analysis results

## Part 2: Creating a Library

```python
import numpy as np
from spectraflex import TransferFunctionLibrary, transfer_function, identify, spectrum

# Create empty library
lib = TransferFunctionLibrary()
print(lib)  # TransferFunctionLibrary(empty)
```

### Adding Transfer Functions

Each transfer function must have a **configuration** that identifies its operating condition:

```python
# Create synthetic transfer functions for demonstration
f = np.linspace(0.02, 0.3, 100)

def create_sample_tf(hs, heading):
    """Create a sample TF with config-dependent characteristics."""
    # Magnitude varies with heading (simplified model)
    base_mag = 2.0 + np.sin(np.radians(heading)) * 0.5
    magnitude = base_mag * np.ones((len(f), 2))
    magnitude[:, 0] *= (1 + 0.3 * np.exp(-(f - 0.1)**2 / 0.01))  # Roll peak
    magnitude[:, 1] *= (1 + 0.2 * np.exp(-(f - 0.08)**2 / 0.01))  # Pitch peak

    phase = np.zeros((len(f), 2))
    coherence = 0.9 * np.ones((len(f), 2))

    return transfer_function.create(
        frequency=f,
        magnitude=magnitude,
        phase=phase,
        coherence=coherence,
        variable_names=["Roll", "Pitch"],
        config={"hs": hs, "heading": heading}
    )

# Populate library with a matrix of conditions
for hs in [2.0, 3.0, 4.0]:
    for heading in [0.0, 45.0, 90.0, 135.0, 180.0]:
        tf = create_sample_tf(hs, heading)
        lib.add(tf)
        print(f"Added: Hs={hs}, Heading={heading}")

print(f"\nLibrary now contains {len(lib)} transfer functions")
print(lib)
```

### Library Properties

```python
# Configuration keys (automatically inferred)
print(f"Config keys: {lib.config_keys}")

# Get all configurations
configs = lib.configs
print(f"\nFirst 3 configs:")
for c in configs[:3]:
    print(f"  {c}")

# Get parameter ranges
hs_min, hs_max = lib.get_config_range("hs")
print(f"\nHs range: {hs_min} to {hs_max}")

# Get unique values
headings = lib.get_unique_values("heading")
print(f"Headings: {headings}")
```

## Part 3: Selecting from Library

### Exact Match

```python
# Select by exact configuration match
tf = lib.select(hs=3.0, heading=90.0)

print(f"Selected TF:")
print(f"  Config: {tf.attrs.get('config')}")
print(f"  Variables: {list(tf.coords['variable'].values)}")
```

### Handling Missing Configurations

```python
# Trying to select non-existent config raises KeyError
try:
    tf = lib.select(hs=2.5, heading=60.0)
except KeyError as e:
    print(f"Not found: {e}")
```

### Nearest Neighbor Lookup

For conditions between stored values:

```python
# Find nearest available configuration
tf_nearest = lib.lookup(hs=2.5, heading=60.0, method="nearest")

print(f"Requested: Hs=2.5, Heading=60")
print(f"Found: {tf_nearest.attrs.get('config')}")
```

### Interpolated Lookup

For smoother variation between conditions:

```python
# Interpolate between nearby configurations
tf_interp = lib.lookup(hs=2.5, heading=60.0, method="linear")

print(f"Interpolated TF:")
print(f"  Config: {tf_interp.attrs.get('config')}")
print(f"  Interpolation method: {tf_interp.attrs.get('interpolation_method')}")
print(f"  Number of neighbors used: {tf_interp.attrs.get('interpolation_k')}")
```

## Part 4: Filtering Libraries

Create subsets based on conditions:

```python
# Get all transfer functions for heading = 90 degrees
lib_beam = lib.filter(heading=90.0)
print(f"Beam sea library: {len(lib_beam)} TFs")

# Get all for Hs = 4.0 m
lib_severe = lib.filter(hs=4.0)
print(f"Severe conditions: {len(lib_severe)} TFs")

# Multiple filters
lib_subset = lib.filter(hs=3.0, heading=45.0)
print(f"Specific condition: {len(lib_subset)} TFs")
```

## Part 5: Saving and Loading

### Save to NetCDF

```python
# Save the library
lib.save("vessel_tf_library.nc")
print("Library saved to vessel_tf_library.nc")
```

### Load from NetCDF

```python
# Load the library
lib_loaded = TransferFunctionLibrary.load("vessel_tf_library.nc")

print(f"Loaded library: {lib_loaded}")
print(f"Config keys: {lib_loaded.config_keys}")
print(f"Number of TFs: {len(lib_loaded)}")

# Verify contents
tf_check = lib_loaded.select(hs=3.0, heading=90.0)
print(f"Successfully retrieved TF for Hs=3.0, Heading=90")
```

## Part 6: Building Libraries from Simulation Results

### Batch Processing Workflow

```python
from pathlib import Path
from spectraflex import identify
from spectraflex.orcaflex.batch import config_from_filename

# Assume we have a directory of spectra files
spectra_dir = Path("./spectra_output")

# Build library from all spectra files
lib = TransferFunctionLibrary()

# In practice, you would glob the directory:
# for spectra_file in spectra_dir.glob("*_spectra.npz"):
#     config = config_from_filename(spectra_file)
#     tf = identify.from_spectra(spectra_file, config=config)
#     lib.add(tf)
#     print(f"Added: {spectra_file.name}")

# Example with simulated files
example_files = [
    ("model_Hs2.0_Dir0_spectra.npz", {"hs": 2.0, "heading": 0.0}),
    ("model_Hs2.0_Dir45_spectra.npz", {"hs": 2.0, "heading": 45.0}),
    ("model_Hs4.0_Dir0_spectra.npz", {"hs": 4.0, "heading": 0.0}),
    ("model_Hs4.0_Dir45_spectra.npz", {"hs": 4.0, "heading": 45.0}),
]

# Demonstrate with synthetic data
for filename, config in example_files:
    tf = create_sample_tf(config["hs"], config["heading"])
    lib.add(tf)
    print(f"Added: {filename}")

print(f"\nFinal library: {lib}")
```

### Using the CLI

```bash
# Build library from command line
spectraflex library build ./spectra_dir/ -o vessel_library.nc

# Check library info
spectraflex library info vessel_library.nc
```

## Part 7: Response Prediction with Libraries

### Single Condition

```python
from spectraflex import predict, spectrum

# Select transfer function for beam seas
tf = lib.select(hs=3.0, heading=90.0)

# Create wave spectrum
f = tf.coords["frequency"].values
wave = spectrum.jonswap(hs=5.0, tp=12.0, f=f, gamma=3.3)

# Predict response
stats = predict.statistics(tf, wave, duration=10800)

print("Response for beam seas (heading=90°):")
for var, s in stats.items():
    print(f"  {var}: Hs={s['hs']:.2f}, MPM={s['mpm']:.2f}")
```

### Comparing Across Conditions

```python
import matplotlib.pyplot as plt

# Compare roll response across all headings
headings = lib.get_unique_values("heading")
roll_hs = []
roll_mpm = []

for heading in headings:
    tf = lib.select(hs=3.0, heading=heading)
    wave = spectrum.jonswap(hs=5.0, tp=12.0, f=f, gamma=3.3)
    stats = predict.statistics(tf, wave, duration=10800)
    roll_hs.append(stats["Roll"]["hs"])
    roll_mpm.append(stats["Roll"]["mpm"])

# Plot polar diagram
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

headings_rad = np.radians(headings)
# Close the polar plot
headings_rad = np.append(headings_rad, headings_rad[0])
roll_hs = np.append(roll_hs, roll_hs[0])

ax.plot(headings_rad, roll_hs, 'b-o', linewidth=2)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Roll Hs [deg] vs Wave Direction", pad=20)
plt.show()
```

### Interpolated Predictions

```python
# Predict for conditions not in the library
heading_test = 75.0  # Not stored, will interpolate

tf_interp = lib.lookup(hs=3.0, heading=heading_test, method="linear")
wave = spectrum.jonswap(hs=5.0, tp=12.0, f=f, gamma=3.3)
stats = predict.statistics(tf_interp, wave, duration=10800)

print(f"Interpolated prediction for heading={heading_test}°:")
for var, s in stats.items():
    print(f"  {var}: Hs={s['hs']:.2f}, MPM={s['mpm']:.2f}")
```

## Part 8: Advanced Library Operations

### Combining Libraries

```python
# Create libraries for different drafts
lib_ballast = TransferFunctionLibrary()
lib_loaded = TransferFunctionLibrary()

# Add TFs with draft in config
for heading in [0.0, 90.0, 180.0]:
    tf = create_sample_tf(3.0, heading)
    tf.attrs["config"]["draft"] = 10.0
    lib_ballast.add(tf, config={"hs": 3.0, "heading": heading, "draft": 10.0})

    tf = create_sample_tf(3.0, heading)
    tf.attrs["config"]["draft"] = 20.0
    lib_loaded.add(tf, config={"hs": 3.0, "heading": heading, "draft": 20.0})

print(f"Ballast library: {lib_ballast}")
print(f"Loaded library: {lib_loaded}")
```

### Exporting to Combined Dataset

For analysis across all configurations:

```python
# Convert library to single xarray Dataset
ds = lib.to_dataset()

print(f"Combined dataset dimensions: {dict(ds.dims)}")
print(f"Data variables: {list(ds.data_vars)}")
print(f"Coordinates: {list(ds.coords)}")

# Access data across all configs
roll_mag = ds["magnitude"].sel(variable="Roll")
print(f"Roll magnitude shape: {roll_mag.shape}")  # (frequency, config)
```

## Part 9: Best Practices

### Library Organization

1. **Consistent config keys**: Use the same parameter names across all TFs
2. **Meaningful labels**: Include units in config values documentation
3. **Version control**: Include library version or creation date in filename

```python
# Good config structure
config = {
    "hs": 3.0,           # Significant wave height [m]
    "heading": 45.0,     # Wave heading [deg]
    "draft": 15.0,       # Vessel draft [m]
    "current": 0.5,      # Current speed [m/s]
}
```

### Recommended Directory Structure

```
project/
├── models/
│   └── template.dat
├── simulations/
│   ├── Hs2.0_Dir0.sim
│   └── ...
├── spectra/
│   ├── Hs2.0_Dir0_spectra.npz
│   └── ...
└── libraries/
    ├── vessel_v1.nc          # Version 1
    ├── vessel_v2.nc          # Version 2 (updated)
    └── vessel_draft15m.nc    # Specific loading condition
```

### Quality Checks Before Adding

```python
def validate_tf_for_library(tf, min_coherence=0.3):
    """Check if TF meets quality standards for library inclusion."""
    warnings = []

    # Check mean coherence
    for var in tf.coords["variable"].values:
        mean_coh = tf["coherence"].sel(variable=var).mean().values
        if mean_coh < min_coherence:
            warnings.append(f"{var}: low coherence ({mean_coh:.2f})")

    # Check for NaN values
    if tf["magnitude"].isnull().any():
        warnings.append("Contains NaN values in magnitude")

    # Check frequency coverage
    f = tf.coords["frequency"].values
    if f.max() < 0.2:
        warnings.append(f"Limited frequency range: max={f.max():.3f} Hz")

    return len(warnings) == 0, warnings

# Usage
is_valid, warnings = validate_tf_for_library(tf)
if not is_valid:
    print("Quality warnings:")
    for w in warnings:
        print(f"  - {w}")
```

## Summary

In this tutorial, you learned:

1. **Creating Libraries**: Organizing TFs by configuration parameters
2. **Querying**: Exact selection, nearest lookup, and interpolation
3. **Persistence**: Saving and loading to NetCDF format
4. **Batch Processing**: Building libraries from simulation results
5. **Analysis**: Comparing responses across conditions
6. **Best Practices**: Organization and quality control

## Key Takeaways

- **Libraries index by config**: Each TF has unique (hs, heading, draft, ...)
- **Three lookup modes**: exact (select), nearest, interpolated (linear)
- **Save to NetCDF**: Portable format preserving all data and metadata
- **Batch processing**: Automate library building from many simulations

## Next Steps

- **Tutorial 4**: Complete end-to-end workflow with OrcaFlex
