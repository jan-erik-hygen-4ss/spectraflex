# Tutorial 2: Transfer Function Identification

This tutorial covers how to identify transfer functions from simulation data, including synthetic examples and real OrcaFlex workflows.

## What You'll Learn

- The theory behind transfer function identification
- How to identify H(f) from time histories
- Understanding and using coherence
- Working with OrcaFlex simulations
- Quality assessment of identified transfer functions

## Prerequisites

- Completed Tutorial 1
- For OrcaFlex examples: OrcFxAPI installed with valid licence

## Part 1: The Identification Method

### Theory: Cross-Spectral Analysis

For a linear system with input x(t) and output y(t), the transfer function is:

```
H(f) = S_xy(f) / S_xx(f)
```

where:
- S_xx(f) is the input auto-spectrum (power spectral density)
- S_xy(f) is the cross-spectrum between input and output

The **coherence** measures how much of the output is explained by the input:

```
γ²(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))
```

- γ² = 1: Perfect linear relationship
- γ² = 0: No linear relationship

### Why White Noise?

White noise has constant spectral density across all frequencies. This means:
- Every frequency is equally excited
- H(f) can be identified at all frequencies simultaneously
- One simulation provides the complete transfer function

## Part 2: Synthetic Example

Let's create a known transfer function, generate input/output signals, and verify we can recover H(f).

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from spectraflex import identify, transfer_function

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
dt = 0.1       # Sample interval [s]
duration = 600  # 10 minutes
n_samples = int(duration / dt)
t = np.arange(n_samples) * dt

# Generate white noise input (wave elevation)
wave = np.random.randn(n_samples)

# Define a known transfer function: simple second-order system
# H(s) = ω²_n / (s² + 2ζω_n s + ω²_n)
f_n = 0.1  # Natural frequency [Hz]
zeta = 0.1  # Damping ratio

# Create digital filter
w_n = 2 * np.pi * f_n
b, a = signal.butter(2, f_n / (0.5 / dt), btype='low')

# Generate response by filtering wave through the system
# (This is a simplification - real H(f) would use frequency-domain filtering)
response = signal.lfilter(b, a, wave) * 5  # Scale for visibility

# Identify transfer function
tf = identify.from_time_histories(
    wave_elevation=wave,
    responses={"Response": response},
    dt=dt,
    nperseg=1024,
    window="hann"
)

print("Identified Transfer Function:")
print(f"  Frequency points: {len(tf.coords['frequency'])}")
print(f"  Frequency range: {tf.coords['frequency'].values[0]:.4f} - {tf.coords['frequency'].values[-1]:.4f} Hz")
```

### Visualizing the Results

```python
f = tf.coords["frequency"].values
mag = tf["magnitude"].sel(variable="Response").values
phase = tf["phase"].sel(variable="Response").values
coh = tf["coherence"].sel(variable="Response").values

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Magnitude (log scale)
axes[0].semilogy(f, mag)
axes[0].set_ylabel('|H(f)|')
axes[0].set_title('Identified Transfer Function')
axes[0].grid(True)
axes[0].axvline(f_n, color='r', linestyle='--', alpha=0.5, label=f'f_n = {f_n} Hz')
axes[0].legend()

# Phase
axes[1].plot(f, np.degrees(phase))
axes[1].set_ylabel('Phase [deg]')
axes[1].set_ylim([-180, 180])
axes[1].grid(True)

# Coherence
axes[2].plot(f, coh)
axes[2].set_ylabel('Coherence γ²')
axes[2].set_xlabel('Frequency [Hz]')
axes[2].set_ylim([0, 1.05])
axes[2].axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
```

## Part 3: Effect of Simulation Length

Longer simulations give more reliable transfer function estimates:

```python
durations = [60, 120, 300, 600, 1200]  # seconds
coherence_means = []

for dur in durations:
    n = int(dur / dt)
    wave_short = np.random.randn(n)
    response_short = signal.lfilter(b, a, wave_short) * 5

    nperseg = min(1024, n // 4)  # Adjust nperseg for short signals

    tf_short = identify.from_time_histories(
        wave_elevation=wave_short,
        responses={"Response": response_short},
        dt=dt,
        nperseg=nperseg,
    )

    mean_coh = tf_short["coherence"].mean().values
    coherence_means.append(mean_coh)
    print(f"Duration: {dur:4d}s, Mean coherence: {mean_coh:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(durations, coherence_means, 'o-')
plt.xlabel('Simulation Duration [s]')
plt.ylabel('Mean Coherence')
plt.title('Effect of Duration on Identification Quality')
plt.grid(True)
plt.show()
```

## Part 4: Multiple Response Variables

Real systems have multiple outputs. Let's identify transfer functions for several response variables:

```python
# Generate multiple responses with different characteristics
responses = {}

# Response 1: Low-pass (e.g., heave)
b1, a1 = signal.butter(2, 0.05 / (0.5 / dt), btype='low')
responses["Heave"] = signal.lfilter(b1, a1, wave) * 2

# Response 2: Band-pass (e.g., roll)
b2, a2 = signal.butter(2, [0.05, 0.15] / (0.5 / dt), btype='band')
responses["Roll"] = signal.lfilter(b2, a2, wave) * 10

# Response 3: High-pass (e.g., acceleration)
b3, a3 = signal.butter(2, 0.1 / (0.5 / dt), btype='high')
responses["Accel"] = signal.lfilter(b3, a3, wave) * 3

# Identify all transfer functions
tf_multi = identify.from_time_histories(
    wave_elevation=wave,
    responses=responses,
    dt=dt,
    nperseg=1024,
)

print(f"Variables: {list(tf_multi.coords['variable'].values)}")
```

### Comparing Multiple Transfer Functions

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
f = tf_multi.coords["frequency"].values

for var in tf_multi.coords["variable"].values:
    mag = tf_multi["magnitude"].sel(variable=var).values
    axes[0].semilogy(f, mag, label=var)

axes[0].set_ylabel('|H(f)|')
axes[0].set_title('Transfer Functions for Multiple Responses')
axes[0].legend()
axes[0].grid(True)

for var in tf_multi.coords["variable"].values:
    phase = tf_multi["phase"].sel(variable=var).values
    axes[1].plot(f, np.degrees(phase), label=var)

axes[1].set_ylabel('Phase [deg]')
axes[1].set_ylim([-180, 180])
axes[1].legend()
axes[1].grid(True)

for var in tf_multi.coords["variable"].values:
    coh = tf_multi["coherence"].sel(variable=var).values
    axes[2].plot(f, coh, label=var)

axes[2].set_ylabel('Coherence γ²')
axes[2].set_xlabel('Frequency [Hz]')
axes[2].set_ylim([0, 1.05])
axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

## Part 5: Working with OrcaFlex Data

### From Pre-computed Spectra Files

The most common workflow uses post-calculation actions to save spectra:

```python
from spectraflex import identify

# Load from spectra file (created by post-calc action)
tf = identify.from_spectra(
    "model_Hs2.0_Dir45_spectra.npz",
    config={"hs": 2.0, "heading": 45.0, "draft": 21.0},
    freq_range=(0.02, 0.25)  # Match white noise frequency range
)

# Check quality
print("Transfer Function Quality:")
for var in tf.coords["variable"].values:
    coh = tf["coherence"].sel(variable=var)
    print(f"  {var}:")
    print(f"    Mean coherence: {coh.mean().values:.3f}")
    print(f"    Min coherence:  {coh.min().values:.3f}")
```

### Directly from .sim Files (requires OrcFxAPI)

```python
from spectraflex import identify

# Define what to extract
results = [
    {"object": "Vessel", "variable": "Rotation 1", "label": "Roll"},
    {"object": "Vessel", "variable": "Rotation 2", "label": "Pitch"},
    {"object": "Vessel", "variable": "X", "label": "Surge"},
    {"object": "Riser", "variable": "Effective Tension", "arclength": 0.0, "label": "TopTension"},
    {"object": "Riser", "variable": "Bend Moment", "arclength": 0.0, "label": "TopBendMoment"},
]

# Identify (requires OrcFxAPI)
try:
    tf = identify.from_sim(
        sim_path="white_noise_simulation.sim",
        results=results,
        nperseg=2048,
        config={"hs": 2.0, "heading": 45.0},
        wave_position=(0.0, 0.0, 0.0),
    )
    print(f"Identified {len(tf.coords['variable'])} transfer functions")
except ImportError:
    print("OrcFxAPI not available - using pre-computed spectra instead")
```

## Part 6: Quality Assessment

### Coherence Analysis

Low coherence indicates unreliable transfer function values:

```python
from spectraflex import identify

# Get coherence mask
mask = identify.coherence_mask(tf, threshold=0.5)

# Count frequencies above threshold per variable
for var in tf.coords["variable"].values:
    n_good = mask.sel(variable=var).sum().values
    n_total = len(tf.coords["frequency"])
    pct = 100 * n_good / n_total
    print(f"{var}: {pct:.0f}% of frequencies have coherence > 0.5")
```

### Applying Coherence Masks

```python
# Set transfer function to zero where coherence is low
tf_masked = identify.apply_coherence_mask(tf, threshold=0.5)

# Compare
var = tf.coords["variable"].values[0]
f = tf.coords["frequency"].values

plt.figure(figsize=(10, 5))
plt.plot(f, tf["magnitude"].sel(variable=var), label='Original', alpha=0.7)
plt.plot(f, tf_masked["magnitude"].sel(variable=var), label='Masked (γ² < 0.5)', alpha=0.7)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|H(f)|')
plt.title(f'Effect of Coherence Masking - {var}')
plt.legend()
plt.grid(True)
plt.show()
```

### Comparing Transfer Functions

Check linearity by comparing H(f) from different Hs values:

```python
from spectraflex import transfer_function

# Load transfer functions from different wave heights
tf_hs2 = identify.from_spectra("model_Hs2.0_spectra.npz")
tf_hs4 = identify.from_spectra("model_Hs4.0_spectra.npz")

# Compare
comparison = transfer_function.compare(tf_hs2, tf_hs4, variable="Roll")

print("Comparison of H(f) at Hs=2m vs Hs=4m:")
print(f"  Correlation:        {comparison['correlation']:.4f}")
print(f"  Mean rel. diff:     {comparison['mean_rel_diff']:.4f}")
print(f"  Max rel. diff:      {comparison['max_rel_diff']:.4f}")
print(f"  RMS diff:           {comparison['rms_diff']:.4f}")

if comparison['correlation'] > 0.95:
    print("\n✓ Good linearity - H(f) is consistent across wave heights")
else:
    print("\n⚠ Possible non-linearity detected")
```

### Averaging Multiple Runs

Improve reliability by averaging transfer functions from multiple simulations:

```python
# Load multiple transfer functions (e.g., from different seeds)
tf_list = [
    identify.from_spectra("model_seed1_spectra.npz"),
    identify.from_spectra("model_seed2_spectra.npz"),
    identify.from_spectra("model_seed3_spectra.npz"),
]

# Average with coherence weighting
tf_avg = transfer_function.average(tf_list, weights="coherence")

print(f"Averaged from {tf_avg.attrs.get('averaged_from', len(tf_list))} simulations")
```

## Part 7: Saving and Loading Transfer Functions

```python
from spectraflex.io import save_transfer_function, load_transfer_function

# Save to NetCDF
save_transfer_function(tf, "identified_tf.nc")

# Load back
tf_loaded = load_transfer_function("identified_tf.nc")

# Verify
print(f"Loaded TF with {len(tf_loaded.coords['variable'])} variables")
print(f"Config: {tf_loaded.attrs.get('config', 'None')}")
```

## Summary

In this tutorial, you learned:

1. **Identification Theory**: H(f) = S_xy / S_xx and the role of coherence
2. **Synthetic Testing**: Verifying identification with known transfer functions
3. **Multiple Variables**: Identifying transfer functions for several response variables
4. **OrcaFlex Integration**: Working with .sim files and spectra files
5. **Quality Assessment**: Using coherence, masking, and comparison
6. **Reliability Improvement**: Averaging multiple simulations

## Key Takeaways

- **Longer simulations = higher coherence = more reliable H(f)**
- **White noise is essential** because it excites all frequencies equally
- **Check coherence** to know where your transfer function is reliable
- **Compare across Hs values** to verify system linearity
- **Average multiple runs** to reduce statistical uncertainty

## Next Steps

- **Tutorial 3**: Managing transfer function libraries
- **Tutorial 4**: Complete workflow from simulation to prediction
