# Tutorial 4: Complete Workflow with OrcaFlex

This tutorial walks through the complete spectraflex workflow from model setup to response prediction, demonstrating a realistic engineering analysis workflow.

## What You'll Learn

- Setting up OrcaFlex models for white noise analysis
- Generating batch simulation files
- Extracting and processing results
- Building transfer function libraries
- Predicting responses for design conditions

## Prerequisites

- Completed Tutorials 1-3
- OrcaFlex software and licence (for running simulations)
- OrcFxAPI Python package (for automation)

## Overview

The complete workflow consists of five phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Setup                                                  │
│  Template Model → Generate White Noise Variations               │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Simulation                                             │
│  Run Simulations → Extract Spectra (post-calc action)          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Identification                                         │
│  Spectra Files → Transfer Functions → Library                  │
├─────────────────────────────────────────────────────────────────┤
│  Phase 4: Prediction                                             │
│  Wave Spectrum + TF Library → Response Statistics               │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5: Reporting                                              │
│  Generate Tables, Plots, Design Values                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Setup

### Project Structure

```bash
mkdir -p project/{models,simulations,spectra,libraries,results}
```

```
project/
├── models/
│   ├── template.dat         # Base OrcaFlex model
│   └── generated/           # White noise variations
├── simulations/             # Completed .sim files
├── spectra/                 # Extracted spectra (.npz)
├── libraries/               # Transfer function libraries
└── results/                 # Predictions and reports
```

### Define Analysis Parameters

```python
from pathlib import Path

# Project paths
PROJECT_DIR = Path("./project")
TEMPLATE = PROJECT_DIR / "models" / "template.dat"
MODELS_DIR = PROJECT_DIR / "models" / "generated"
SIM_DIR = PROJECT_DIR / "simulations"
SPECTRA_DIR = PROJECT_DIR / "spectra"
LIBRARY_DIR = PROJECT_DIR / "libraries"
RESULTS_DIR = PROJECT_DIR / "results"

# Analysis parameters
FREQ_RANGE = (0.02, 0.25)  # White noise frequency range [Hz]
DURATION = 1024.0          # Simulation duration [s]
NPERSEG = 2048             # FFT segment length

# Operating conditions to analyze
HS_VALUES = [2.0, 3.0, 4.0]
HEADINGS = [0.0, 45.0, 90.0, 135.0, 180.0]

# Response variables to extract
RESULTS_SPEC = [
    {"object": "Vessel", "variable": "Rotation 1", "label": "Roll"},
    {"object": "Vessel", "variable": "Rotation 2", "label": "Pitch"},
    {"object": "Vessel", "variable": "Rotation 3", "label": "Yaw"},
    {"object": "Vessel", "variable": "X", "label": "Surge"},
    {"object": "Vessel", "variable": "Y", "label": "Sway"},
    {"object": "Vessel", "variable": "Z", "label": "Heave"},
    {"object": "Riser", "variable": "Effective Tension", "arclength": 0.0, "label": "TopTension"},
    {"object": "Riser", "variable": "Bend Moment", "arclength": 0.0, "label": "TopBendMoment"},
]
```

### Generate White Noise Models

```python
from spectraflex.orcaflex import white_noise, post_calc
from spectraflex.orcaflex.batch import generate_case_matrix

# Generate case matrix
cases = generate_case_matrix(
    hs=HS_VALUES,
    wave_direction=HEADINGS,
)
print(f"Total cases to simulate: {len(cases)}")

# Generate YAML variation files
paths = white_noise.generate_batch(
    template=TEMPLATE,
    matrix={
        "hs": HS_VALUES,
        "wave_direction": HEADINGS,
    },
    freq_range=FREQ_RANGE,
    duration=DURATION,
    output_dir=MODELS_DIR,
)

print(f"\nGenerated {len(paths)} model files:")
for p in paths[:5]:
    print(f"  {p.name}")
if len(paths) > 5:
    print(f"  ... and {len(paths) - 5} more")
```

### Create Post-Calculation Action Script

```python
# Generate the post-calc script that will extract spectra
script_path = MODELS_DIR / "extract_spectra.py"

post_calc.write_standalone_script(
    path=script_path,
    results=RESULTS_SPEC,
    nperseg=NPERSEG,
    window="hann",
    wave_position=(0.0, 0.0, 0.0),
)

print(f"Post-calc script written to: {script_path}")
print("\nTo use with OrcaFlex batch processing, add this script as a")
print("'Post-calculation action' in each model's General Data.")
```

### Attach Post-Calc Action to Models (Optional)

```python
# This requires OrcFxAPI
try:
    import OrcFxAPI as ofx

    for yml_path in MODELS_DIR.glob("*.yml"):
        # Load base model with variation
        model = ofx.Model(str(TEMPLATE))
        model.LoadDataMod(str(yml_path))

        # Attach post-calc action
        post_calc.attach_post_calc(
            model,
            RESULTS_SPEC,
            nperseg=NPERSEG,
        )

        # Save modified model
        dat_path = yml_path.with_suffix(".dat")
        model.SaveData(str(dat_path))
        print(f"Created: {dat_path.name}")

except ImportError:
    print("OrcFxAPI not available - attach post-calc manually in OrcaFlex GUI")
```

---

## Phase 2: Simulation

### Running Simulations

Simulations can be run via:

1. **OrcaFlex GUI Batch**: File → Batch Processing
2. **Command line**: `OrcFxAPI /batch models/*.dat`
3. **Python script**:

```python
# Example batch runner (requires OrcFxAPI)
try:
    import OrcFxAPI as ofx

    def run_simulation(model_path):
        """Run a single simulation."""
        model = ofx.Model(str(model_path))

        if model.state != ofx.ModelState.SimulationComplete:
            print(f"Running: {model_path.name}")
            model.RunSimulation()

            # Save results
            sim_path = SIM_DIR / model_path.with_suffix(".sim").name
            model.SaveSimulation(str(sim_path))
            print(f"  Saved: {sim_path.name}")
        else:
            print(f"Already complete: {model_path.name}")

    # Run all models
    for model_path in sorted(MODELS_DIR.glob("*.dat")):
        run_simulation(model_path)

except ImportError:
    print("Run simulations manually using OrcaFlex GUI or batch processor")
```

### Monitor Progress

```python
from spectraflex.orcaflex.batch import find_completed_sims, find_spectra_files, get_batch_status

# Check batch status
status = get_batch_status(SIM_DIR, expected_cases=len(cases))

print(f"Batch Progress:")
print(f"  Expected cases: {status.get('n_expected', len(cases))}")
print(f"  Completed sims: {status.get('n_sims', 0)}")
print(f"  Extracted spectra: {status.get('n_spectra', 0)}")
print(f"  Completion: {status.get('completion', 0) * 100:.0f}%")
```

---

## Phase 3: Identification

### Extract Transfer Functions from Spectra

```python
from spectraflex import identify, TransferFunctionLibrary
from spectraflex.orcaflex.batch import config_from_filename

# Find all spectra files
spectra_files = sorted(SPECTRA_DIR.glob("*_spectra.npz"))
print(f"Found {len(spectra_files)} spectra files")

# Build library
lib = TransferFunctionLibrary()

for spectra_file in spectra_files:
    # Parse config from filename
    config = config_from_filename(spectra_file)

    # Identify transfer function
    tf = identify.from_spectra(
        spectra_file,
        config=config,
        freq_range=FREQ_RANGE,
    )

    # Quality check
    mean_coherence = tf["coherence"].mean().values
    if mean_coherence < 0.3:
        print(f"Warning: Low coherence ({mean_coherence:.2f}) for {spectra_file.name}")

    # Add to library
    lib.add(tf)
    print(f"Added: {spectra_file.name} -> {config}")

print(f"\nLibrary complete: {lib}")
```

### Quality Assessment

```python
import numpy as np

# Check coherence across all conditions
print("\nCoherence Summary by Variable:")
print("-" * 60)

for var in RESULTS_SPEC:
    label = var["label"]
    coherences = []

    for config in lib.configs:
        tf = lib.select(**config)
        coh = tf["coherence"].sel(variable=label)
        coherences.append({
            "config": config,
            "mean": float(coh.mean()),
            "min": float(coh.min()),
        })

    mean_all = np.mean([c["mean"] for c in coherences])
    min_all = np.min([c["min"] for c in coherences])

    print(f"{label:20s}  Mean: {mean_all:.3f}  Min: {min_all:.3f}")
```

### Save Library

```python
# Save with descriptive name
library_path = LIBRARY_DIR / "vessel_tf_library.nc"
lib.save(library_path)
print(f"Saved library to: {library_path}")

# Also save summary
summary_path = LIBRARY_DIR / "library_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"Transfer Function Library Summary\n")
    f.write(f"=" * 50 + "\n\n")
    f.write(f"Number of configurations: {len(lib)}\n")
    f.write(f"Config keys: {lib.config_keys}\n\n")
    f.write(f"Parameter ranges:\n")
    for key in lib.config_keys:
        min_val, max_val = lib.get_config_range(key)
        unique = lib.get_unique_values(key)
        f.write(f"  {key}: {min_val} to {max_val} ({len(unique)} values)\n")
    f.write(f"\nVariables:\n")
    for var in RESULTS_SPEC:
        f.write(f"  - {var['label']}\n")

print(f"Saved summary to: {summary_path}")
```

---

## Phase 4: Prediction

### Define Design Conditions

```python
# Design sea states for prediction
DESIGN_CONDITIONS = [
    {"name": "Operational", "hs": 3.0, "tp": 9.0, "gamma": 3.3},
    {"name": "Design", "hs": 6.0, "tp": 12.0, "gamma": 2.5},
    {"name": "Survival", "hs": 10.0, "tp": 15.0, "gamma": 2.0},
]

# Headings to analyze (use library headings or custom set)
PRED_HEADINGS = lib.get_unique_values("heading")

# Storm duration for MPM
STORM_DURATION = 10800  # 3 hours
```

### Compute Predictions

```python
from spectraflex import spectrum, predict
import pandas as pd

results = []

for condition in DESIGN_CONDITIONS:
    print(f"\nProcessing: {condition['name']}")

    for heading in PRED_HEADINGS:
        # Get transfer function (use nearest Hs from library)
        tf = lib.lookup(hs=3.0, heading=heading, method="nearest")

        # Create wave spectrum
        f = tf.coords["frequency"].values
        wave = spectrum.jonswap(
            hs=condition["hs"],
            tp=condition["tp"],
            f=f,
            gamma=condition["gamma"],
        )

        # Compute statistics
        stats = predict.statistics(tf, wave, duration=STORM_DURATION)

        # Store results
        for var_name, var_stats in stats.items():
            results.append({
                "condition": condition["name"],
                "hs_wave": condition["hs"],
                "tp": condition["tp"],
                "heading": heading,
                "variable": var_name,
                "hs_response": var_stats["hs"],
                "sigma": var_stats["sigma"],
                "tz": var_stats["tz"],
                "mpm": var_stats["mpm"],
            })

# Create DataFrame
df = pd.DataFrame(results)
print(f"\nGenerated {len(df)} predictions")
```

### Generate Summary Tables

```python
# Pivot table: MPM by variable and heading for each condition
for condition in DESIGN_CONDITIONS:
    print(f"\n{condition['name']} Condition - MPM Values")
    print("=" * 70)

    df_cond = df[df["condition"] == condition["name"]]
    pivot = df_cond.pivot_table(
        values="mpm",
        index="variable",
        columns="heading",
        aggfunc="first"
    )

    print(pivot.round(3).to_string())
```

### Find Maximum Values

```python
# Find worst heading for each variable and condition
print("\nMaximum Response Summary")
print("=" * 80)
print(f"{'Condition':<12} {'Variable':<15} {'Max MPM':>10} {'Heading':>10} {'Hs_resp':>10}")
print("-" * 80)

for condition in DESIGN_CONDITIONS:
    df_cond = df[df["condition"] == condition["name"]]

    for var in df_cond["variable"].unique():
        df_var = df_cond[df_cond["variable"] == var]
        idx_max = df_var["mpm"].idxmax()
        row = df_var.loc[idx_max]

        print(f"{condition['name']:<12} {var:<15} {row['mpm']:>10.3f} "
              f"{row['heading']:>10.0f}° {row['hs_response']:>10.3f}")
```

### Save Results

```python
# Save to CSV
csv_path = RESULTS_DIR / "response_predictions.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved predictions to: {csv_path}")

# Save summary report
report_path = RESULTS_DIR / "prediction_summary.txt"
with open(report_path, "w") as f:
    f.write("Response Prediction Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
    f.write(f"Storm Duration: {STORM_DURATION/3600:.1f} hours\n\n")

    for condition in DESIGN_CONDITIONS:
        f.write(f"\n{condition['name']} ({condition['hs']}m, {condition['tp']}s)\n")
        f.write("-" * 40 + "\n")

        df_cond = df[df["condition"] == condition["name"]]
        for var in df_cond["variable"].unique():
            df_var = df_cond[df_cond["variable"] == var]
            max_mpm = df_var["mpm"].max()
            worst_heading = df_var.loc[df_var["mpm"].idxmax(), "heading"]
            f.write(f"  {var:<15} MPM={max_mpm:>8.3f} (worst at {worst_heading}°)\n")

print(f"Saved report to: {report_path}")
```

---

## Phase 5: Visualization

### Response Polar Plots

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_polar_response(df, variable, value_col="mpm"):
    """Create polar plot of response vs heading."""
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"},
                              figsize=(15, 5))

    for ax, condition in zip(axes, DESIGN_CONDITIONS):
        df_cond = df[(df["condition"] == condition["name"]) &
                     (df["variable"] == variable)]

        headings = np.radians(df_cond["heading"].values)
        values = df_cond[value_col].values

        # Close the polar plot
        headings = np.append(headings, headings[0])
        values = np.append(values, values[0])

        ax.plot(headings, values, 'b-o', linewidth=2)
        ax.fill(headings, values, alpha=0.3)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"{condition['name']}\n(Hs={condition['hs']}m)")

    fig.suptitle(f"{variable} - {value_col.upper()} vs Wave Direction", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"polar_{variable}_{value_col}.png", dpi=150)
    plt.show()

# Generate polar plots for key variables
for var in ["Roll", "Pitch", "TopTension"]:
    plot_polar_response(df, var, "mpm")
```

### Transfer Function Comparison

```python
def plot_tf_comparison(lib, variable, config_key="heading"):
    """Plot transfer functions across one configuration dimension."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    values = lib.get_unique_values(config_key)
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

    for val, color in zip(values, colors):
        tf = lib.select(**{config_key: val, "hs": 3.0})
        f = tf.coords["frequency"].values

        mag = tf["magnitude"].sel(variable=variable).values
        coh = tf["coherence"].sel(variable=variable).values

        axes[0].semilogy(f, mag, color=color, label=f"{config_key}={val}")
        axes[1].plot(f, coh, color=color)

    axes[0].set_ylabel("|H(f)|")
    axes[0].set_title(f"{variable} Transfer Function")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].set_ylabel("Coherence γ²")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylim([0, 1.05])
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"tf_comparison_{variable}.png", dpi=150)
    plt.show()

# Plot TF comparison for key variables
plot_tf_comparison(lib, "Roll")
plot_tf_comparison(lib, "TopTension")
```

---

## CLI Quick Reference

```bash
# Generate models
spectraflex generate template.dat \
    --hs 2.0 3.0 4.0 \
    --direction 0 45 90 135 180 \
    --freq-range 0.02 0.25 \
    --duration 1024 \
    -o ./models/generated/

# Build library from spectra
spectraflex library build ./spectra/ -o ./libraries/vessel.nc

# Check library
spectraflex library info ./libraries/vessel.nc

# Predict for single condition
spectraflex predict ./libraries/vessel.nc \
    --hs 6.0 --tp 12.0 --gamma 2.5 \
    --duration 10800 \
    -o ./results/design_condition.json
```

---

## Summary

This tutorial covered the complete spectraflex workflow:

1. **Setup**: Generating white noise model files and post-calc scripts
2. **Simulation**: Running OrcaFlex batch simulations
3. **Identification**: Building transfer function libraries from spectra
4. **Prediction**: Computing response statistics for design conditions
5. **Reporting**: Generating tables, polar plots, and comparisons

## Key Takeaways

- **Automation is key**: Use scripts to manage large parameter matrices
- **Quality checks matter**: Always verify coherence before trusting H(f)
- **Libraries enable rapid analysis**: Once built, predictions are instantaneous
- **Document everything**: Save configs, parameters, and quality metrics

## Further Reading

- [API Reference](../api_reference.md)
- [Usage Guide](../usage_guide.md)
- [OrcaFlex Documentation](https://www.orcina.com/webhelp/OrcaFlex/)
