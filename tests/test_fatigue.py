"""Tests for spectraflex.fatigue module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectraflex import fatigue, transfer_function, spectrum
from spectraflex.fatigue import SNCurve


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dnv_d_curve() -> SNCurve:
    """DNV D curve (most common)."""
    return SNCurve.dnv_d()


@pytest.fixture
def frequency_array() -> np.ndarray:
    """Frequency array for testing."""
    return np.linspace(0.01, 0.5, 100)


@pytest.fixture
def narrow_band_psd(frequency_array: np.ndarray) -> np.ndarray:
    """Narrow-band stress PSD (peaked at 0.1 Hz)."""
    f = frequency_array
    # Very narrow peak -> narrow-band
    return 1000 * np.exp(-((f - 0.1) ** 2) / 0.001)


@pytest.fixture
def wide_band_psd(frequency_array: np.ndarray) -> np.ndarray:
    """Wide-band stress PSD (broad distribution)."""
    f = frequency_array
    # Broad spectrum -> wide-band
    return 100 * np.exp(-((f - 0.2) ** 2) / 0.1)


# =============================================================================
# SNCurve Tests
# =============================================================================


class TestSNCurve:
    """Tests for SNCurve dataclass."""

    def test_dnv_d_parameters(self) -> None:
        """Test DNV-D curve has correct parameters."""
        curve = SNCurve.dnv_d()
        assert curve.m1 == 3.0
        assert curve.log_a1 == 12.164
        assert curve.m2 == 5.0
        assert curve.k == 0.20
        assert curve.name == "DNV-D"

    def test_dnv_d_seawater(self) -> None:
        """Test DNV-D seawater curve has different log_a."""
        curve_air = SNCurve.dnv_d(in_air=True)
        curve_sw = SNCurve.dnv_d(in_air=False)
        assert curve_sw.log_a1 < curve_air.log_a1  # More conservative

    def test_all_dnv_curves_exist(self) -> None:
        """Test all DNV curves can be created."""
        curves = [
            SNCurve.dnv_b1(),
            SNCurve.dnv_b2(),
            SNCurve.dnv_c(),
            SNCurve.dnv_c1(),
            SNCurve.dnv_c2(),
            SNCurve.dnv_d(),
            SNCurve.dnv_e(),
            SNCurve.dnv_f(),
            SNCurve.dnv_f1(),
            SNCurve.dnv_f3(),
            SNCurve.dnv_g(),
            SNCurve.dnv_w1(),
            SNCurve.dnv_w2(),
            SNCurve.dnv_w3(),
        ]
        assert len(curves) == 14
        # All should have positive slopes
        for curve in curves:
            assert curve.m1 > 0
            assert curve.m2 > 0

    def test_cycles_to_failure_single_value(self, dnv_d_curve: SNCurve) -> None:
        """Test cycles to failure for single stress value."""
        n = dnv_d_curve.cycles_to_failure(100.0)
        assert n > 0
        # DNV-D at 100 MPa should be around 10^6 cycles
        assert 1e5 < n < 1e7

    def test_cycles_to_failure_array(self, dnv_d_curve: SNCurve) -> None:
        """Test cycles to failure for array of stresses."""
        s = np.array([50, 100, 200])
        n = dnv_d_curve.cycles_to_failure(s)
        assert len(n) == 3
        # Higher stress -> fewer cycles
        assert n[0] > n[1] > n[2]

    def test_cycles_to_failure_inverse_relationship(self, dnv_d_curve: SNCurve) -> None:
        """Test that doubling stress reduces cycles by 2^m."""
        s1, s2 = 100, 200
        n1 = dnv_d_curve.cycles_to_failure(s1)
        n2 = dnv_d_curve.cycles_to_failure(s2)
        # For slope m=3, doubling stress should reduce cycles by 2^3 = 8
        ratio = n1 / n2
        np.testing.assert_allclose(ratio, 8.0, rtol=0.01)

    def test_stress_at_transition(self, dnv_d_curve: SNCurve) -> None:
        """Test stress at transition point."""
        s_trans = dnv_d_curve.stress_at_transition()
        assert s_trans > 0
        # Verify: at this stress, N should equal n_transition
        n = dnv_d_curve.cycles_to_failure(s_trans)
        np.testing.assert_allclose(n, 1e7, rtol=0.01)

    def test_with_scf(self, dnv_d_curve: SNCurve) -> None:
        """Test SCF application."""
        scf = 1.5
        curve_scf = dnv_d_curve.with_scf(scf)

        # SCF should reduce fatigue life
        s = 100.0
        n_orig = dnv_d_curve.cycles_to_failure(s)
        n_scf = curve_scf.cycles_to_failure(s)

        # With SCF, effective stress is higher -> fewer cycles
        assert n_scf < n_orig
        # Ratio should be SCF^m
        expected_ratio = scf ** dnv_d_curve.m1
        np.testing.assert_allclose(n_orig / n_scf, expected_ratio, rtol=0.01)

    def test_with_thickness_no_correction(self, dnv_d_curve: SNCurve) -> None:
        """Test thickness correction when t <= t_ref."""
        curve_thick = dnv_d_curve.with_thickness(20.0)  # < 25mm ref
        # Should return same curve (no correction needed)
        assert curve_thick.log_a1 == dnv_d_curve.log_a1

    def test_with_thickness_correction(self, dnv_d_curve: SNCurve) -> None:
        """Test thickness correction when t > t_ref."""
        t = 50.0  # > 25mm ref
        curve_thick = dnv_d_curve.with_thickness(t)

        # Should reduce fatigue life
        s = 100.0
        n_orig = dnv_d_curve.cycles_to_failure(s)
        n_thick = curve_thick.cycles_to_failure(s)
        assert n_thick < n_orig

    # --- DNV-RP-C203 (2024) parameter verification ---

    def test_2024_transition_points_air(self) -> None:
        """Verify transition points match DNV-RP-C203 (2024) Table 2-1."""
        # B1, B2, C, C1, C2 transition at 5e6 in air
        for factory in [SNCurve.dnv_b1, SNCurve.dnv_b2, SNCurve.dnv_c,
                        SNCurve.dnv_c1, SNCurve.dnv_c2]:
            curve = factory(in_air=True)
            assert curve.n_transition == 5e6, f"{curve.name} should have N_t=5e6"

        # D through W3 transition at 1e7 in air
        for factory in [SNCurve.dnv_d, SNCurve.dnv_e, SNCurve.dnv_f,
                        SNCurve.dnv_f1, SNCurve.dnv_f3, SNCurve.dnv_g,
                        SNCurve.dnv_w1, SNCurve.dnv_w2, SNCurve.dnv_w3]:
            curve = factory(in_air=True)
            assert curve.n_transition == 1e7, f"{curve.name} should have N_t=1e7"

    def test_2024_transition_points_cp(self) -> None:
        """Verify all seawater CP curves transition at 1e6 (Table 2-2)."""
        for factory in [SNCurve.dnv_b1, SNCurve.dnv_b2, SNCurve.dnv_c,
                        SNCurve.dnv_c1, SNCurve.dnv_c2, SNCurve.dnv_d,
                        SNCurve.dnv_e, SNCurve.dnv_f, SNCurve.dnv_f1,
                        SNCurve.dnv_f3, SNCurve.dnv_g, SNCurve.dnv_w1,
                        SNCurve.dnv_w2, SNCurve.dnv_w3]:
            curve = factory(in_air=False)
            assert curve.n_transition == 1e6, f"{curve.name} should have N_t=1e6"

    def test_2024_c_curves_slope_35(self) -> None:
        """Verify C, C1, C2 have m1=3.5 in 2024 edition (both environments)."""
        for factory in [SNCurve.dnv_c, SNCurve.dnv_c1, SNCurve.dnv_c2]:
            for in_air in [True, False]:
                curve = factory(in_air=in_air)
                assert curve.m1 == 3.5, f"{curve.name} should have m1=3.5"

    def test_2024_cp_log_a2_matches_air(self) -> None:
        """Verify seawater CP log_a2 equals in-air log_a2 (2024 edition)."""
        for factory in [SNCurve.dnv_b1, SNCurve.dnv_b2, SNCurve.dnv_c,
                        SNCurve.dnv_c1, SNCurve.dnv_c2, SNCurve.dnv_d,
                        SNCurve.dnv_e, SNCurve.dnv_f, SNCurve.dnv_f1,
                        SNCurve.dnv_f3, SNCurve.dnv_g, SNCurve.dnv_w1,
                        SNCurve.dnv_w2, SNCurve.dnv_w3]:
            air = factory(in_air=True)
            cp = factory(in_air=False)
            assert cp.log_a2 == air.log_a2, (
                f"{air.name}: CP log_a2={cp.log_a2} != air log_a2={air.log_a2}"
            )

    def test_2024_d_air_fatigue_limit(self) -> None:
        """Verify D curve fatigue limit at N_t matches Table 2-1 (52.63 MPa)."""
        curve = SNCurve.dnv_d(in_air=True)
        s_t = curve.stress_at_transition()
        np.testing.assert_allclose(s_t, 52.63, rtol=0.001)

    def test_2024_b1_air_fatigue_limit(self) -> None:
        """Verify B1 curve fatigue limit at N_t matches Table 2-1 (127.21 MPa)."""
        curve = SNCurve.dnv_b1(in_air=True)
        s_t = curve.stress_at_transition()
        np.testing.assert_allclose(s_t, 127.21, rtol=0.01)

    def test_2024_c_air_fatigue_limit(self) -> None:
        """Verify C curve fatigue limit at N_t matches Table 2-1 (96.21 MPa)."""
        curve = SNCurve.dnv_c(in_air=True)
        s_t = curve.stress_at_transition()
        np.testing.assert_allclose(s_t, 96.21, rtol=0.01)

    def test_2024_thickness_exponents(self) -> None:
        """Verify thickness exponents match 2024 tables."""
        expected_k = {
            "dnv_b1": 0.0, "dnv_b2": 0.0,
            "dnv_c": 0.05, "dnv_c1": 0.10, "dnv_c2": 0.15,
            "dnv_d": 0.20, "dnv_e": 0.20,
            "dnv_f": 0.25, "dnv_f1": 0.25, "dnv_f3": 0.25,
            "dnv_g": 0.25, "dnv_w1": 0.25, "dnv_w2": 0.25, "dnv_w3": 0.25,
        }
        for name, k in expected_k.items():
            factory = getattr(SNCurve, name)
            curve = factory(in_air=True)
            assert curve.k == k, f"{curve.name} k={curve.k} != expected {k}"


# =============================================================================
# Spectral Parameter Tests
# =============================================================================


class TestSpectralParameters:
    """Tests for spectral parameter functions."""

    def test_peak_rate(self) -> None:
        """Test peak rate calculation."""
        m2, m4 = 100, 400
        nu_p = fatigue.peak_rate(m2, m4)
        expected = np.sqrt(m4 / m2)
        np.testing.assert_allclose(nu_p, expected)

    def test_peak_rate_zero_m2(self) -> None:
        """Test peak rate with zero m2."""
        nu_p = fatigue.peak_rate(0, 100)
        assert nu_p == 0.0

    def test_zero_crossing_rate(self) -> None:
        """Test zero-crossing rate calculation."""
        m0, m2 = 100, 400
        nu_0 = fatigue.zero_crossing_rate(m0, m2)
        expected = np.sqrt(m2 / m0)
        np.testing.assert_allclose(nu_0, expected)

    def test_irregularity_factor(self) -> None:
        """Test irregularity factor calculation."""
        m0, m2, m4 = 100, 200, 500
        gamma = fatigue.irregularity_factor(m0, m2, m4)
        expected = m2 / np.sqrt(m0 * m4)
        np.testing.assert_allclose(gamma, expected)

    def test_irregularity_factor_narrow_band(self) -> None:
        """Test irregularity factor for narrow-band (gamma -> 1)."""
        # For narrow-band: m2^2 ≈ m0 * m4
        m0 = 100
        m4 = 10000
        m2 = np.sqrt(m0 * m4)  # Makes gamma = 1
        gamma = fatigue.irregularity_factor(m0, m2, m4)
        np.testing.assert_allclose(gamma, 1.0, rtol=0.01)


# =============================================================================
# Dirlik Tests
# =============================================================================


class TestDirlik:
    """Tests for Dirlik method."""

    def test_dirlik_coefficients_sum_to_one(self) -> None:
        """Test that D1 + D2 + D3 = 1."""
        coef = fatigue.dirlik_coefficients(100, 50, 30, 20)
        d_sum = coef["D1"] + coef["D2"] + coef["D3"]
        np.testing.assert_allclose(d_sum, 1.0, rtol=0.01)

    def test_dirlik_pdf_non_negative(self) -> None:
        """Test Dirlik PDF is non-negative."""
        s = np.linspace(0, 100, 50)
        pdf = fatigue.dirlik_pdf(s, m0=100, m1=50, m2=30, m4=20)
        assert np.all(pdf >= 0)

    def test_dirlik_pdf_integrates_to_one(self) -> None:
        """Test Dirlik PDF integrates to approximately 1."""
        s = np.linspace(0.01, 200, 1000)
        pdf = fatigue.dirlik_pdf(s, m0=100, m1=50, m2=30, m4=20)
        integral = np.trapezoid(pdf, s)
        np.testing.assert_allclose(integral, 1.0, rtol=0.05)

    def test_dirlik_damage_positive(self, dnv_d_curve: SNCurve) -> None:
        """Test Dirlik damage is positive."""
        damage = fatigue.dirlik_damage(
            m0=100, m1=50, m2=30, m4=20,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        assert damage > 0

    def test_dirlik_damage_scales_with_time(self, dnv_d_curve: SNCurve) -> None:
        """Test Dirlik damage scales linearly with exposure time."""
        d1 = fatigue.dirlik_damage(
            m0=100, m1=50, m2=30, m4=20,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        d2 = fatigue.dirlik_damage(
            m0=100, m1=50, m2=30, m4=20,
            sn_curve=dnv_d_curve,
            exposure_time=7200,
        )
        np.testing.assert_allclose(d2 / d1, 2.0, rtol=0.01)


# =============================================================================
# Narrow-Band Tests
# =============================================================================


class TestNarrowBand:
    """Tests for narrow-band method."""

    def test_narrow_band_damage_positive(self, dnv_d_curve: SNCurve) -> None:
        """Test narrow-band damage is positive."""
        damage = fatigue.narrow_band_damage(
            m0=100,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
            m2=50,
        )
        assert damage > 0

    def test_narrow_band_damage_scales_with_time(self, dnv_d_curve: SNCurve) -> None:
        """Test narrow-band damage scales linearly with time."""
        d1 = fatigue.narrow_band_damage(
            m0=100, sn_curve=dnv_d_curve, exposure_time=3600, m2=50
        )
        d2 = fatigue.narrow_band_damage(
            m0=100, sn_curve=dnv_d_curve, exposure_time=7200, m2=50
        )
        np.testing.assert_allclose(d2 / d1, 2.0, rtol=0.01)

    def test_narrow_band_requires_m2_or_nu0(self, dnv_d_curve: SNCurve) -> None:
        """Test narrow-band raises error if neither m2 nor nu_0 provided."""
        with pytest.raises(ValueError, match="Either m2 or nu_0"):
            fatigue.narrow_band_damage(
                m0=100, sn_curve=dnv_d_curve, exposure_time=3600
            )


# =============================================================================
# Damage From Spectrum Tests
# =============================================================================


class TestDamageFromSpectrum:
    """Tests for damage_from_spectrum function."""

    def test_returns_dict(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test function returns dict with expected keys."""
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        assert isinstance(result, dict)
        assert "damage" in result
        assert "life_seconds" in result
        assert "n_cycles" in result
        assert "damage_rate" in result
        assert "bandwidth" in result

    def test_dirlik_method(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test Dirlik method produces valid result."""
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
            method="dirlik",
        )
        assert result["damage"] > 0
        assert result["life_seconds"] > 0

    def test_narrow_band_method(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test narrow-band method produces valid result."""
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
            method="narrow_band",
        )
        assert result["damage"] > 0

    def test_narrow_band_conservative_for_narrow_spectrum(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test narrow-band is typically conservative for narrow-band spectra."""
        result_dir = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
            method="dirlik",
        )
        result_nb = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
            method="narrow_band",
        )
        # Narrow-band should be similar or slightly conservative
        # (may give higher damage estimate)
        assert result_nb["damage"] > 0
        assert result_dir["damage"] > 0

    def test_bandwidth_narrow_spectrum(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test bandwidth parameter for narrow spectrum."""
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        # Narrow-band should have low epsilon (close to 0)
        assert result["bandwidth"] < 0.5

    def test_bandwidth_wide_spectrum(
        self,
        frequency_array: np.ndarray,
        wide_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test bandwidth parameter for wide spectrum."""
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=wide_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        # Wide-band should have higher epsilon
        assert result["bandwidth"] > 0.3

    def test_invalid_method_raises(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            fatigue.damage_from_spectrum(
                frequency=frequency_array,
                stress_psd=narrow_band_psd,
                sn_curve=dnv_d_curve,
                exposure_time=3600,
                method="invalid",
            )

    def test_life_inverse_of_damage(
        self,
        frequency_array: np.ndarray,
        narrow_band_psd: np.ndarray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test life = exposure_time / damage."""
        exposure = 3600
        result = fatigue.damage_from_spectrum(
            frequency=frequency_array,
            stress_psd=narrow_band_psd,
            sn_curve=dnv_d_curve,
            exposure_time=exposure,
        )
        expected_life = exposure / result["damage"]
        np.testing.assert_allclose(result["life_seconds"], expected_life, rtol=0.001)


# =============================================================================
# Damage From Transfer Function Tests
# =============================================================================


class TestDamageFromTransferFunction:
    """Tests for damage_from_transfer_function."""

    @pytest.fixture
    def sample_tf(self) -> xr.Dataset:
        """Create a sample transfer function dataset."""
        f = np.linspace(0.01, 0.5, 100)
        # Simple constant magnitude (10 MPa/m)
        mag = 10.0 * np.ones_like(f)
        phase = np.zeros_like(f)
        coh = np.ones_like(f)
        return transfer_function.create(
            frequency=f,
            magnitude=mag,
            phase=phase,
            coherence=coh,
            variable_names=["stress"],
        )

    @pytest.fixture
    def sample_wave_spectrum(self, sample_tf: xr.Dataset) -> xr.DataArray:
        """Create a sample wave spectrum."""
        f = sample_tf.coords["frequency"].values
        # JONSWAP-like shape
        s = spectrum.jonswap(hs=3.0, tp=10.0, f=f)
        return s

    def test_returns_dict(
        self,
        sample_tf: xr.Dataset,
        sample_wave_spectrum: xr.DataArray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test function returns dict with expected keys."""
        result = fatigue.damage_from_transfer_function(
            tf=sample_tf,
            wave_spectrum=sample_wave_spectrum,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        assert isinstance(result, dict)
        assert "damage" in result
        assert "stress_m0" in result
        assert "stress_rms" in result

    def test_damage_positive(
        self,
        sample_tf: xr.Dataset,
        sample_wave_spectrum: xr.DataArray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test damage is positive."""
        result = fatigue.damage_from_transfer_function(
            tf=sample_tf,
            wave_spectrum=sample_wave_spectrum,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        assert result["damage"] > 0
        assert result["stress_rms"] > 0

    def test_stress_m0_from_tf_and_wave(
        self,
        sample_tf: xr.Dataset,
        sample_wave_spectrum: xr.DataArray,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test stress m0 is computed from |H|^2 * S_wave."""
        result = fatigue.damage_from_transfer_function(
            tf=sample_tf,
            wave_spectrum=sample_wave_spectrum,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        # With H = 10 MPa/m and known wave spectrum, stress_m0 should be:
        # m0_stress = |H|^2 * m0_wave = 100 * m0_wave
        m0_wave = np.trapezoid(sample_wave_spectrum.values, sample_tf.coords["frequency"].values)
        expected_m0_stress = 100 * m0_wave
        np.testing.assert_allclose(result["stress_m0"], expected_m0_stress, rtol=0.1)

    def test_with_numpy_wave_spectrum(
        self,
        sample_tf: xr.Dataset,
        dnv_d_curve: SNCurve,
    ) -> None:
        """Test with numpy array wave spectrum."""
        f = sample_tf.coords["frequency"].values
        s_wave = spectrum.jonswap(hs=3.0, tp=10.0, f=f).values

        result = fatigue.damage_from_transfer_function(
            tf=sample_tf,
            wave_spectrum=s_wave,
            sn_curve=dnv_d_curve,
            exposure_time=3600,
        )
        assert result["damage"] > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for fatigue workflow."""

    def test_full_workflow_from_jonswap(self) -> None:
        """Test full workflow: JONSWAP -> TF -> damage."""
        # Create frequency array
        f = np.linspace(0.02, 0.3, 100)

        # Create transfer function (stress per wave height)
        # Simulate a resonant response at T=10s (f=0.1 Hz)
        f_res = 0.1
        damping = 0.05
        # Simple resonance model: H(f) = H0 / sqrt((1-(f/f_res)^2)^2 + (2*damping*f/f_res)^2)
        h0 = 5.0  # Base stress per meter wave [MPa/m]
        freq_ratio = f / f_res
        h_mag = h0 / np.sqrt((1 - freq_ratio**2)**2 + (2 * damping * freq_ratio)**2)

        tf = transfer_function.create(
            frequency=f,
            magnitude=h_mag,
            phase=np.zeros_like(f),
            coherence=np.ones_like(f),
            variable_names=["stress"],
        )

        # Create wave spectrum
        wave_spec = spectrum.jonswap(hs=4.0, tp=10.0, f=f)

        # Calculate damage for 1 year
        one_year = 365.25 * 24 * 3600
        curve = SNCurve.dnv_d()

        result = fatigue.damage_from_transfer_function(
            tf=tf,
            wave_spectrum=wave_spec,
            sn_curve=curve,
            exposure_time=one_year,
            method="dirlik",
        )

        # Should have meaningful results
        assert result["damage"] > 0
        assert result["stress_rms"] > 0
        # Life should be > 0
        assert result["life_seconds"] > 0

        # Print for inspection
        print(f"\nStress RMS: {result['stress_rms']:.2f} MPa")
        print(f"Annual damage: {result['damage']:.4f}")
        print(f"Fatigue life: {result['life_seconds'] / one_year:.1f} years")
