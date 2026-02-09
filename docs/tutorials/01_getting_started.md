# Tutorial 1: Getting Started with spectraflex

This tutorial introduces the basic concepts and workflow of spectraflex for transfer function identification and spectral response prediction.

## What You'll Learn

- Core concepts: transfer functions, spectra, and coherence
- How to work with wave spectra
- How to compute spectral statistics
- Basic response prediction workflow

## Prerequisites

```bash
pip install spectraflex
```

## Part 1: Understanding Wave Spectra

Wave energy is distributed across frequencies. A **wave spectrum** S(f) describes how much energy exists at each frequency.

### Creating a JONSWAP Spectrum

The JONSWAP spectrum is the most common model for wind-generated waves:

```python
import numpy as np
import matplotlib.pyplot as plt
from spectraflex import spectrum

# Create frequency array from 0.02 to 0.3 Hz
f = spectrum.frequency_array(f_min=0.02, f_max=0.3, n_freq=200)

# Create JONSWAP spectrum
# Hs = 4m significant wave height
# Tp = 10s peak period
# gamma = 3.3 (standard JONSWAP peakedness)
wave = spectrum.jonswap(hs=4.0, tp=10.0, f=f, gamma=3.3)

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(f, wave.values)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectral Density [m²/Hz]')
plt.title('JONSWAP Wave Spectrum (Hs=4m, Tp=10s)')
plt.grid(True)
plt.show()
```

### Comparing Spectrum Types

```python
# Compare JONSWAP and Pierson-Moskowitz
wave_jonswap = spectrum.jonswap(hs=4.0, tp=10.0, f=f, gamma=3.3)
wave_pm = spectrum.pierson_moskowitz(hs=4.0, tp=10.0, f=f)

plt.figure(figsize=(10, 5))
plt.plot(f, wave_jonswap.values, label='JONSWAP (γ=3.3)')
plt.plot(f, wave_pm.values, label='Pierson-Moskowitz (γ=1.0)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectral Density [m²/Hz]')
plt.title('Comparison of Wave Spectra')
plt.legend()
plt.grid(True)
plt.show()
```

The JONSWAP spectrum has a sharper peak than Pierson-Moskowitz, representing developing seas where energy is concentrated near the peak frequency.

## Part 2: Computing Spectral Statistics

From a spectrum, we can derive important statistics without needing the actual time series.

```python
from spectraflex import statistics

# Compute comprehensive statistics
stats = statistics.all_statistics(f, wave.values, duration=10800)

print("Wave Statistics:")
print(f"  Hs (significant height): {stats['hs']:.2f} m")
print(f"  Tp (peak period):        {stats['tp']:.2f} s")
print(f"  Tz (zero-crossing):      {stats['tz']:.2f} s")
print(f"  σ  (std deviation):      {stats['sigma']:.2f} m")
print(f"  MPM (3-hour):            {stats['mpm']:.2f} m")
```

### Understanding Spectral Moments

The spectrum's shape is characterized by **spectral moments**:

```python
moments = statistics.spectral_moments(f, wave.values, orders=(0, 1, 2, 4))

print("\nSpectral Moments:")
print(f"  m₀ = {moments[0]:.4f}  (variance)")
print(f"  m₁ = {moments[1]:.6f}")
print(f"  m₂ = {moments[2]:.6f}")
print(f"  m₄ = {moments[4]:.8f}")

# Verify Hs = 4 * sqrt(m0)
hs_computed = 4 * np.sqrt(moments[0])
print(f"\nHs from m₀: {hs_computed:.2f} m")
```

### Most Probable Maximum (MPM)

The MPM is the expected maximum value during a storm. It depends on the number of cycles:

```python
# Compare MPM for different storm durations
durations = [1800, 3600, 10800, 21600]  # 30min, 1h, 3h, 6h
labels = ['30 min', '1 hour', '3 hours', '6 hours']

print("\nMost Probable Maximum vs Duration:")
for dur, label in zip(durations, labels):
    mpm = statistics.mpm_rayleigh(moments[0], dur, m2_val=moments[2])
    print(f"  {label}: MPM = {mpm:.2f} m")
```

## Part 3: Introduction to Transfer Functions

A **transfer function** H(f) describes how a system transforms input (waves) to output (structural response):

```
Response(f) = H(f) × Wave(f)
```

For power spectra:
```
S_response(f) = |H(f)|² × S_wave(f)
```

### Creating a Simple Transfer Function

Let's create a synthetic transfer function for a vessel roll motion:

```python
from spectraflex import transfer_function
import numpy as np

# Frequency array
f = np.linspace(0.02, 0.3, 100)

# Create a resonant transfer function
# Peak at f = 0.1 Hz (10 second roll period)
f0 = 0.1  # Natural frequency
zeta = 0.1  # Damping ratio

# Simple resonance model: H(f) = 1 / sqrt((1-(f/f0)²)² + (2*zeta*f/f0)²)
magnitude = 1.0 / np.sqrt((1 - (f/f0)**2)**2 + (2*zeta*f/f0)**2)
phase = np.arctan2(-2*zeta*f/f0, 1 - (f/f0)**2)
coherence = np.ones_like(f) * 0.95  # High coherence (synthetic data)

# Create transfer function dataset
tf = transfer_function.create(
    frequency=f,
    magnitude=magnitude,
    phase=phase,
    coherence=coherence,
    variable_names=["Roll"],
    config={"vessel": "Example", "draft": 20.0}
)

print(tf)
```

### Visualizing the Transfer Function

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Magnitude
axes[0].plot(f, tf["magnitude"].values)
axes[0].set_ylabel('|H(f)| [deg/m]')
axes[0].set_title('Transfer Function: Vessel Roll')
axes[0].grid(True)

# Phase
axes[1].plot(f, np.degrees(tf["phase"].values))
axes[1].set_ylabel('Phase [deg]')
axes[1].grid(True)

# Coherence
axes[2].plot(f, tf["coherence"].values)
axes[2].set_ylabel('Coherence γ²')
axes[2].set_xlabel('Frequency [Hz]')
axes[2].set_ylim([0, 1])
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

## Part 4: Response Prediction

Now let's predict the roll response to a wave spectrum:

```python
from spectraflex import predict

# Create wave spectrum on same frequency grid
wave = spectrum.jonswap(hs=4.0, tp=10.0, f=f, gamma=3.3)

# Compute response spectrum
response = predict.response_spectrum(tf, wave)

# Plot input and response spectra
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(f, wave.values)
axes[0].set_ylabel('Wave S(f) [m²/Hz]')
axes[0].set_title('Input Wave Spectrum')
axes[0].grid(True)

axes[1].plot(f, response["Syy"].sel(variable="Roll").values)
axes[1].set_ylabel('Roll S(f) [deg²/Hz]')
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_title('Response Spectrum')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### Computing Response Statistics

```python
# Get response statistics
roll_stats = predict.statistics(tf, wave, duration=10800)["Roll"]

print("\nRoll Response Statistics (3-hour storm):")
print(f"  Hs (significant):  {roll_stats['hs']:.2f} deg")
print(f"  σ (std deviation): {roll_stats['sigma']:.2f} deg")
print(f"  Tz:                {roll_stats['tz']:.2f} s")
print(f"  MPM:               {roll_stats['mpm']:.2f} deg")
```

### Varying Sea State

Let's see how roll response changes with wave height:

```python
hs_values = [2.0, 3.0, 4.0, 5.0, 6.0]
results = []

for hs in hs_values:
    wave = spectrum.jonswap(hs=hs, tp=10.0, f=f, gamma=3.3)
    stats = predict.statistics(tf, wave, duration=10800)["Roll"]
    results.append({
        "wave_hs": hs,
        "roll_hs": stats["hs"],
        "roll_mpm": stats["mpm"]
    })

print("\nRoll Response vs Wave Height:")
print("-" * 50)
print(f"{'Wave Hs [m]':>12} {'Roll Hs [deg]':>15} {'Roll MPM [deg]':>15}")
print("-" * 50)
for r in results:
    print(f"{r['wave_hs']:>12.1f} {r['roll_hs']:>15.2f} {r['roll_mpm']:>15.2f}")
```

## Part 5: Synthesizing Time Series

For time-domain analysis, we can synthesize random time series that match the predicted spectrum:

```python
# Synthesize 10-minute time series
wave = spectrum.jonswap(hs=4.0, tp=10.0, f=f, gamma=3.3)

ts = predict.synthesize_timeseries(
    tf, wave,
    duration=600,  # 10 minutes
    dt=0.5,        # 2 Hz sample rate
    seed=42        # For reproducibility
)

# Plot time series
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

t = ts.coords["time"].values

axes[0].plot(t, ts["wave"].values)
axes[0].set_ylabel('Wave Elevation [m]')
axes[0].set_title('Synthesized Time Series')
axes[0].grid(True)

axes[1].plot(t, ts["Roll"].values)
axes[1].set_ylabel('Roll [deg]')
axes[1].set_xlabel('Time [s]')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Verify statistics
from scipy import signal
f_ts, psd_wave = signal.welch(ts["wave"].values, fs=2.0, nperseg=256)
hs_ts = 4 * np.sqrt(np.trapz(psd_wave, f_ts))
print(f"\nTime series wave Hs: {hs_ts:.2f} m (target: 4.0 m)")
```

## Summary

In this tutorial, you learned:

1. **Wave Spectra**: How to create JONSWAP and PM spectra, and understand spectral shapes
2. **Spectral Statistics**: Computing Hs, Tp, Tz, and MPM from spectra
3. **Transfer Functions**: The frequency-domain relationship between input and response
4. **Response Prediction**: Computing response spectra and statistics from H(f) and S_wave(f)
5. **Time Series Synthesis**: Generating random realizations matching predicted spectra

## Next Steps

- **Tutorial 2**: Transfer function identification from OrcaFlex simulations
- **Tutorial 3**: Managing transfer function libraries for different operating conditions
- **Tutorial 4**: Complete workflow from simulation to response prediction
