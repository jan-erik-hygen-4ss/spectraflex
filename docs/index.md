# spectraflex Documentation

**Transfer function identification and spectral response prediction for OrcaFlex simulations**

spectraflex enables fast spectral analysis of offshore structures by:
1. Identifying transfer functions H(f) from white noise simulations
2. Predicting response spectra and statistics for arbitrary sea states
3. Managing libraries of transfer functions across operating conditions

## Quick Links

- [Theory](theory.md) - Mathematical foundations
- [Usage Guide](usage_guide.md) - How to use spectraflex
- [API Reference](api_reference.md) - Complete API documentation
- [Tutorials](tutorials/) - Step-by-step learning guides

## Installation

```bash
pip install spectraflex

# For OrcaFlex integration
pip install OrcFxAPI
```

## Quick Start

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

## Documentation Contents

### [Theory](theory.md)

Mathematical foundations covering:
- Random processes and power spectral density
- Wave spectra (JONSWAP, Pierson-Moskowitz, white noise)
- Spectral moments and derived statistics
- Transfer functions and RAOs
- Cross-spectral analysis and the H1 estimator
- Coherence function and interpretation
- Response spectrum prediction
- Extreme value statistics (Rayleigh, MPM)
- Practical considerations (resolution, duration, linearity)

### [Usage Guide](usage_guide.md)

Comprehensive guide covering:
- Core concepts (transfer functions, spectra, coherence)
- Working with wave spectra
- Transfer function identification
- Response prediction
- Managing transfer function libraries
- OrcaFlex integration
- Command-line interface
- Best practices

### [API Reference](api_reference.md)

Complete reference for all modules:
- `spectraflex.spectrum` - Wave spectrum definitions
- `spectraflex.statistics` - Spectral statistics calculations
- `spectraflex.transfer_function` - TransferFunction Dataset operations
- `spectraflex.identify` - Transfer function identification
- `spectraflex.predict` - Response prediction
- `spectraflex.library` - TransferFunctionLibrary class
- `spectraflex.io` - I/O utilities
- `spectraflex.orcaflex` - OrcaFlex integration modules

### Tutorials

Step-by-step guides for learning spectraflex:

1. **[Getting Started](tutorials/01_getting_started.md)**
   - Core concepts
   - Working with wave spectra
   - Computing spectral statistics
   - Basic response prediction

2. **[Transfer Function Identification](tutorials/02_transfer_function_identification.md)**
   - Cross-spectral analysis theory
   - Identifying H(f) from time histories
   - Understanding coherence
   - Quality assessment

3. **[Transfer Function Libraries](tutorials/03_transfer_function_libraries.md)**
   - Creating and populating libraries
   - Querying and interpolation
   - Saving and loading
   - Batch processing

4. **[Complete Workflow](tutorials/04_complete_workflow.md)**
   - End-to-end analysis with OrcaFlex
   - Model generation
   - Simulation and extraction
   - Prediction and reporting

## Core Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Simulation                                             │
│  Template → White Noise Models → Run Simulations → Spectra     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Identification                                         │
│  Spectra → Transfer Functions → Library                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Prediction                                             │
│  Library + Wave Spectrum → Response Statistics / Time Series   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Transfer Function

The frequency-domain relationship between wave input and structural response:

```
H(f) = Response(f) / Wave(f)
```

For power spectra:
```
S_yy(f) = |H(f)|² × S_xx(f)
```

### White Noise Identification

Using white noise (flat spectrum) waves allows identification of H(f) at all frequencies simultaneously from cross-spectral analysis:

```
H(f) = S_xy(f) / S_xx(f)
```

### Coherence

Measures reliability of the identified transfer function:
- γ² ≈ 1: Reliable (strong linear relationship)
- γ² ≈ 0: Unreliable (noise or nonlinearity)

## Command-Line Interface

```bash
# Generate white noise models
spectraflex generate template.dat --hs 2 3 4 --direction 0 45 90

# Identify transfer function
spectraflex identify spectra.npz -o tf.nc

# Predict response
spectraflex predict tf.nc --hs 5.0 --tp 12.0 --duration 10800

# Manage libraries
spectraflex library build ./spectra/ -o library.nc
spectraflex library info library.nc
```

## Support

- GitHub Issues: [github.com/spectraflex/spectraflex/issues](https://github.com/spectraflex/spectraflex/issues)
- OrcaFlex Documentation: [orcina.com/webhelp/OrcaFlex](https://www.orcina.com/webhelp/OrcaFlex/)
