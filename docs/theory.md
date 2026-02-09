# Theoretical Background

This document covers the mathematical foundations of spectral analysis, transfer function identification, and response prediction used in spectraflex.

## Table of Contents

1. [Random Processes and Spectra](#random-processes-and-spectra)
2. [Wave Spectra](#wave-spectra)
3. [Spectral Moments and Statistics](#spectral-moments-and-statistics)
4. [Transfer Functions](#transfer-functions)
5. [Cross-Spectral Analysis](#cross-spectral-analysis)
6. [Coherence](#coherence)
7. [Response Prediction](#response-prediction)
8. [Extreme Value Statistics](#extreme-value-statistics)
9. [Practical Considerations](#practical-considerations)

---

## Random Processes and Spectra

### Stationary Random Processes

Ocean waves and structural responses are modeled as **stationary random processes** - their statistical properties don't change with time. A stationary process x(t) is characterized by:

- **Mean**: μ = E[x(t)]
- **Variance**: σ² = E[(x(t) - μ)²]
- **Autocorrelation**: R_xx(τ) = E[x(t)x(t+τ)]

### Power Spectral Density

The **power spectral density** (PSD) S(f) describes how variance is distributed across frequencies. It is the Fourier transform of the autocorrelation function:

$$S_{xx}(f) = \int_{-\infty}^{\infty} R_{xx}(\tau) e^{-j2\pi f\tau} d\tau$$

Key properties:
- S(f) ≥ 0 for all f (non-negative)
- Units: [x]²/Hz
- Area under spectrum equals variance:

$$\sigma^2 = \int_0^{\infty} S_{xx}(f) df = m_0$$

### Welch's Method

In practice, we estimate S(f) from finite time series using **Welch's method**:

1. Divide signal into overlapping segments
2. Apply window function to each segment
3. Compute FFT of each windowed segment
4. Average the squared magnitudes

Parameters:
- **nperseg**: Segment length (frequency resolution = fs/nperseg)
- **noverlap**: Overlap between segments (typically 50%)
- **window**: Taper function (e.g., Hann) to reduce spectral leakage

Trade-offs:
- Longer segments → better frequency resolution, more variance
- More segments → less variance, worse frequency resolution

---

## Wave Spectra

### JONSWAP Spectrum

The JONSWAP (Joint North Sea Wave Project) spectrum models developing wind seas:

$$S(f) = \alpha \frac{g^2}{(2\pi)^4 f^5} \exp\left[-\frac{5}{4}\left(\frac{f_p}{f}\right)^4\right] \gamma^r$$

where:
- f_p = 1/T_p is the peak frequency
- γ is the peak enhancement factor (typically 3.3)
- r = exp[-(f - f_p)² / (2σ²f_p²)]
- σ = 0.07 for f < f_p, σ = 0.09 for f > f_p
- α is scaled to achieve the target H_s

The spectrum is normalized so that:

$$H_s = 4\sqrt{m_0} = 4\sqrt{\int_0^{\infty} S(f) df}$$

### Pierson-Moskowitz Spectrum

The Pierson-Moskowitz spectrum represents a **fully developed sea** (equilibrium with wind):

$$S(f) = \alpha \frac{g^2}{(2\pi)^4 f^5} \exp\left[-\frac{5}{4}\left(\frac{f_p}{f}\right)^4\right]$$

This is equivalent to JONSWAP with γ = 1 (no peak enhancement).

### White Noise Spectrum

A **white noise spectrum** has constant spectral density:

$$S(f) = S_0 \quad \text{for } f_{min} \leq f \leq f_{max}$$

where S_0 is chosen to achieve the target variance:

$$S_0 = \frac{m_0}{f_{max} - f_{min}} = \frac{(H_s/4)^2}{f_{max} - f_{min}}$$

White noise is essential for transfer function identification because it excites all frequencies equally.

---

## Spectral Moments and Statistics

### Spectral Moments

The **n-th spectral moment** is defined as:

$$m_n = \int_0^{\infty} f^n S(f) df$$

Key moments:
- m_0: Variance (zeroth moment)
- m_1: First moment (used for mean frequency)
- m_2: Second moment (related to zero-crossing rate)
- m_4: Fourth moment (related to peak rate)

### Derived Statistics

From spectral moments, we derive important statistics:

**Significant Height:**
$$H_s = 4\sqrt{m_0}$$

This equals the average height of the highest 1/3 of waves.

**Zero-Crossing Period:**
$$T_z = \sqrt{\frac{m_0}{m_2}}$$

Average time between upward zero crossings.

**Mean Period:**
$$T_1 = \frac{m_0}{m_1}$$

**Spectral Bandwidth:**
$$\varepsilon = \sqrt{1 - \frac{m_2^2}{m_0 m_4}}$$

- ε → 0: Narrow-banded (regular, nearly sinusoidal)
- ε → 1: Broad-banded (irregular)

---

## Transfer Functions

### Linear System Theory

For a **linear time-invariant (LTI) system**, the relationship between input x(t) and output y(t) in the frequency domain is:

$$Y(f) = H(f) \cdot X(f)$$

where H(f) is the **complex transfer function** (or frequency response function).

### Magnitude and Phase

The transfer function is complex-valued:

$$H(f) = |H(f)| e^{j\phi(f)}$$

- **Magnitude** |H(f)|: Amplification factor at frequency f
- **Phase** φ(f): Phase shift at frequency f

### Physical Interpretation

For a vessel in waves:
- H(f) for roll describes how wave elevation at frequency f produces roll motion
- |H(f)| peaks near the natural roll frequency (resonance)
- φ(f) describes the phase lag between wave and response

### Response Amplitude Operator (RAO)

In naval architecture, the transfer function magnitude is often called the **Response Amplitude Operator (RAO)**:

$$RAO(f) = |H(f)| = \frac{\text{Response amplitude}}{\text{Wave amplitude}}$$

Units depend on the response variable (e.g., deg/m for rotations, m/m for translations).

---

## Cross-Spectral Analysis

### Cross-Spectral Density

The **cross-spectral density** S_xy(f) between input x(t) and output y(t) is:

$$S_{xy}(f) = \int_{-\infty}^{\infty} R_{xy}(\tau) e^{-j2\pi f\tau} d\tau$$

where R_xy(τ) = E[x(t)y(t+τ)] is the cross-correlation.

S_xy(f) is complex-valued:
- Magnitude |S_xy| indicates correlation strength at frequency f
- Phase arg(S_xy) indicates phase relationship

### Transfer Function Estimation

For a linear system, the transfer function can be estimated as:

$$\hat{H}(f) = \frac{S_{xy}(f)}{S_{xx}(f)}$$

This is the **H1 estimator**, optimal when there is noise on the output only.

Alternative estimators:
- **H2**: Ĥ(f) = S_yy(f) / S_yx(f) - optimal for noise on input only
- **Hv**: Geometric mean of H1 and H2

### Why White Noise Input?

With white noise input, S_xx(f) = constant, so:

$$\hat{H}(f) = \frac{S_{xy}(f)}{S_0}$$

This means:
1. All frequencies are equally excited
2. H(f) is identified at all frequencies simultaneously
3. No need to run multiple single-frequency tests

---

## Coherence

### Definition

The **coherence function** γ²(f) measures the linear correlation between input and output at each frequency:

$$\gamma^2(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}$$

Properties:
- 0 ≤ γ²(f) ≤ 1
- γ² = 1: Perfect linear relationship
- γ² = 0: No linear correlation

### Interpretation

Low coherence can indicate:
1. **Noise**: Measurement noise corrupts the signals
2. **Nonlinearity**: System behavior is nonlinear at that frequency
3. **Multiple inputs**: Unmeasured inputs affect the output
4. **Insufficient data**: Too few averages in spectral estimation

### Coherence and Transfer Function Reliability

The variance of the transfer function estimate is inversely related to coherence:

$$\text{Var}[\hat{H}(f)] \propto \frac{1 - \gamma^2(f)}{\gamma^2(f) \cdot n_d}$$

where n_d is the number of spectral averages.

**Rule of thumb**: Trust H(f) where γ² > 0.5

---

## Response Prediction

### Spectral Prediction

Given transfer function H(f) and input spectrum S_xx(f), the response spectrum is:

$$S_{yy}(f) = |H(f)|^2 S_{xx}(f)$$

This follows from:
- For a single frequency: |Y|² = |H|² |X|²
- For a spectrum: spectral densities multiply as |H|²

### Response Statistics

From the response spectrum, we compute statistics:

$$m_0^{(y)} = \int_0^{\infty} |H(f)|^2 S_{xx}(f) df$$

$$H_s^{(y)} = 4\sqrt{m_0^{(y)}}$$

$$T_z^{(y)} = \sqrt{\frac{m_0^{(y)}}{m_2^{(y)}}}$$

### Time Series Synthesis

To generate a random time series matching a spectrum, use **spectral synthesis**:

$$x(t) = \sum_{k=1}^{N} A_k \cos(2\pi f_k t + \phi_k)$$

where:
- A_k = √(2 S(f_k) Δf) is the amplitude at frequency f_k
- φ_k ~ Uniform(0, 2π) are random phases
- Δf is the frequency spacing

For response including transfer function phase:

$$y(t) = \sum_{k=1}^{N} A_k |H(f_k)| \cos(2\pi f_k t + \phi_k + \arg H(f_k))$$

---

## Extreme Value Statistics

### Rayleigh Distribution

For a **narrow-banded Gaussian process**, the peak amplitudes follow a **Rayleigh distribution**:

$$P(A > a) = \exp\left(-\frac{a^2}{2\sigma^2}\right)$$

where σ = √m_0 is the standard deviation.

### Most Probable Maximum (MPM)

The **most probable maximum** (median of the maximum) in N cycles is:

$$MPM = \sigma \sqrt{2 \ln N}$$

where:
- σ = √m_0
- N = T / T_z (number of zero-crossing cycles in duration T)

For a 3-hour storm with T_z = 10s:
- N = 10800 / 10 = 1080 cycles
- MPM = σ × √(2 × ln(1080)) = σ × 3.74

### Expected Maximum

The **expected maximum** (mean of the maximum distribution) is:

$$E[\max] = \sigma \left(\sqrt{2 \ln N} + \frac{\gamma_E}{\sqrt{2 \ln N}}\right)$$

where γ_E ≈ 0.5772 is the Euler-Mascheroni constant.

For large N, E[max] ≈ 1.25 × MPM.

### Design Values

Common design quantiles:

| Exceedance Probability | Factor × σ (N=1000) |
|------------------------|---------------------|
| 50% (MPM)              | 3.72                |
| 36.8% (1/e)            | 3.87                |
| 10%                    | 4.29                |
| 1%                     | 4.80                |

---

## Practical Considerations

### Frequency Resolution vs. Statistical Reliability

The spectral estimate variance depends on:

$$\text{Relative variance} = \frac{1}{n_d}$$

where n_d ≈ 2 × (signal length / nperseg) × (1 - overlap ratio) is the number of independent averages.

Trade-off:
- Longer nperseg → finer frequency resolution, fewer averages, more variance
- Shorter nperseg → coarser resolution, more averages, less variance

**Recommendation**: Choose nperseg so that n_d ≥ 20 for reasonable estimates.

### White Noise Frequency Range

The white noise spectrum should cover all frequencies of interest:

- **Lower limit** (f_min): Set by lowest natural frequency of system
  - Typical: 0.02 Hz (50s period) for floating systems

- **Upper limit** (f_max): Set by highest response frequency
  - Typical: 0.25 Hz (4s period) for wave-frequency responses
  - Higher for springing/ringing responses

### Simulation Duration

Longer simulations improve:
1. Spectral estimate reliability (more averages)
2. Coherence values
3. Low-frequency resolution

**Recommendation**: At least 512s, preferably 1024s or longer for low-frequency responses.

### Build-up Time

Allow transients to decay before the analysis period:

$$T_{buildup} \geq \frac{1}{f_{min}} \times n_{cycles}$$

where n_cycles ≈ 3-5 for lightly damped systems.

### Linearity Verification

Transfer functions assume linearity. Verify by:

1. **Comparing H(f) across different H_s values**: For linear systems, H(f) should be independent of wave height

2. **Checking coherence**: High coherence indicates linear behavior

3. **Computing H1 vs H2 estimators**: They should agree for linear systems

If nonlinearity is significant:
- Consider amplitude-dependent H(f)
- Use higher-order spectral analysis
- Employ time-domain simulations for extreme conditions

---

## References

1. Bendat, J.S. and Piersol, A.G. (2010). *Random Data: Analysis and Measurement Procedures*, 4th ed. Wiley.

2. DNV-RP-C205 (2021). *Environmental Conditions and Environmental Loads*. DNV.

3. Faltinsen, O.M. (1990). *Sea Loads on Ships and Offshore Structures*. Cambridge University Press.

4. Naess, A. and Moan, T. (2012). *Stochastic Dynamics of Marine Structures*. Cambridge University Press.

5. Orcina (2024). *OrcaFlex Documentation*. https://www.orcina.com/webhelp/OrcaFlex/

6. Tucker, M.J. and Pitt, E.G. (2001). *Waves in Ocean Engineering*. Elsevier.
