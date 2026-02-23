"""Tests for spectraflex.orcaflex.fatigue module.

Pure Python tests run without OrcFxAPI.
Integration tests are guarded with pytest.importorskip("OrcFxAPI").
"""

from __future__ import annotations

import numpy as np
import pytest

from spectraflex.fatigue import SNCurve
from spectraflex.orcaflex.fatigue import (
    ComparisonResult,
    OrcaFlexFatigueConfig,
    OrcaFlexFatigueResult,
    SpectralLoadCase,
    compare_results,
    sn_curve_to_orcaflex,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dnv_d_curve() -> SNCurve:
    """DNV D curve (most common)."""
    return SNCurve.dnv_d()


@pytest.fixture
def sample_load_case() -> SpectralLoadCase:
    """A sample spectral load case."""
    return SpectralLoadCase(
        sim_file="test.sim",
        line_name="Riser",
        exposure_time=3600.0,
        hs=3.0,
        tz=8.0,
        tp=10.0,
        gamma=3.3,
    )


@pytest.fixture
def sample_config(
    dnv_d_curve: SNCurve, sample_load_case: SpectralLoadCase
) -> OrcaFlexFatigueConfig:
    """A sample fatigue config."""
    return OrcaFlexFatigueConfig(
        load_cases=[sample_load_case],
        sn_curve=dnv_d_curve,
        arclengths=[(0.0, 100.0)],
    )


# =============================================================================
# SpectralLoadCase Tests
# =============================================================================


class TestSpectralLoadCase:
    """Tests for SpectralLoadCase dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        lc = SpectralLoadCase(
            sim_file="test.sim",
            line_name="Riser",
            exposure_time=3600,
            hs=3.0,
            tz=8.0,
        )
        assert lc.tp is None
        assert lc.gamma is None

    def test_all_fields(self) -> None:
        """Test all fields set."""
        lc = SpectralLoadCase(
            sim_file="test.sim",
            line_name="Riser",
            exposure_time=7200,
            hs=4.0,
            tz=9.0,
            tp=12.0,
            gamma=2.5,
        )
        assert lc.hs == 4.0
        assert lc.tp == 12.0
        assert lc.gamma == 2.5


# =============================================================================
# OrcaFlexFatigueConfig Tests
# =============================================================================


class TestOrcaFlexFatigueConfig:
    """Tests for OrcaFlexFatigueConfig dataclass."""

    def test_defaults(self, dnv_d_curve: SNCurve) -> None:
        """Test default values."""
        lc = SpectralLoadCase(
            sim_file="test.sim",
            line_name="Riser",
            exposure_time=3600,
            hs=3.0,
            tz=8.0,
        )
        config = OrcaFlexFatigueConfig(
            load_cases=[lc],
            sn_curve=dnv_d_curve,
            arclengths=[(0.0, 50.0)],
        )
        assert config.theta_count == 16
        assert config.scf == 1.0
        assert config.thickness_correction == 1.0
        assert config.radial_position == "Outer"

    def test_custom_values(self, dnv_d_curve: SNCurve) -> None:
        """Test custom configuration values."""
        lc = SpectralLoadCase(
            sim_file="test.sim",
            line_name="Riser",
            exposure_time=3600,
            hs=3.0,
            tz=8.0,
        )
        config = OrcaFlexFatigueConfig(
            load_cases=[lc],
            sn_curve=dnv_d_curve,
            arclengths=[(0.0, 50.0), (50.0, 100.0)],
            theta_count=8,
            scf=1.5,
            thickness_correction=1.2,
            radial_position="Inner",
        )
        assert config.theta_count == 8
        assert config.scf == 1.5
        assert len(config.arclengths) == 2


# =============================================================================
# sn_curve_to_orcaflex Tests
# =============================================================================


class TestSNCurveToOrcaflex:
    """Tests for sn_curve_to_orcaflex mapping function."""

    def test_dnv_d_mapping(self, dnv_d_curve: SNCurve) -> None:
        """Test DNV-D curve maps correctly with default kPa conversion."""
        props = sn_curve_to_orcaflex(dnv_d_curve)

        assert props["SNCurvem1"] == 3.0
        # log_a1_kPa = 12.164 + 3.0 * log10(1000) = 12.164 + 9.0 = 21.164
        np.testing.assert_allclose(props["SNCurveLogA1"], 21.164)
        assert props["SNCurveRegionBoundary"] == 1e7
        assert props["SNCurvem2"] == 5.0

    def test_returns_correct_keys(self, dnv_d_curve: SNCurve) -> None:
        """Test return dict has exactly the expected keys."""
        props = sn_curve_to_orcaflex(dnv_d_curve)
        expected_keys = {
            "SNCurvem1",
            "SNCurveLogA1",
            "SNCurveRegionBoundary",
            "SNCurvem2",
        }
        assert set(props.keys()) == expected_keys

    def test_key_order(self, dnv_d_curve: SNCurve) -> None:
        """Test keys are in the correct order for OrcaFlex property setting."""
        props = sn_curve_to_orcaflex(dnv_d_curve)
        keys = list(props.keys())
        # Must set m1, LogA1 before RegionBoundary before m2
        assert keys.index("SNCurvem1") < keys.index("SNCurveRegionBoundary")
        assert keys.index("SNCurveLogA1") < keys.index("SNCurveRegionBoundary")
        assert keys.index("SNCurveRegionBoundary") < keys.index("SNCurvem2")

    def test_all_dnv_curves(self) -> None:
        """Test mapping for all DNV curves applies kPa conversion."""
        factories = [
            SNCurve.dnv_b1,
            SNCurve.dnv_b2,
            SNCurve.dnv_c,
            SNCurve.dnv_c1,
            SNCurve.dnv_c2,
            SNCurve.dnv_d,
            SNCurve.dnv_e,
            SNCurve.dnv_f,
            SNCurve.dnv_f1,
            SNCurve.dnv_f3,
            SNCurve.dnv_g,
            SNCurve.dnv_w1,
            SNCurve.dnv_w2,
            SNCurve.dnv_w3,
        ]
        for factory in factories:
            curve = factory()
            props = sn_curve_to_orcaflex(curve)
            assert props["SNCurvem1"] == curve.m1
            expected_log_a1 = curve.log_a1 + curve.m1 * 3.0  # MPa → kPa
            np.testing.assert_allclose(props["SNCurveLogA1"], expected_log_a1)
            assert props["SNCurveRegionBoundary"] == curve.n_transition
            assert props["SNCurvem2"] == curve.m2

    def test_boundary_is_float(self) -> None:
        """Test that SNCurveRegionBoundary is a float (not int)."""
        curve = SNCurve.dnv_d()
        props = sn_curve_to_orcaflex(curve)
        assert isinstance(props["SNCurveRegionBoundary"], float)

    def test_seawater_curve(self) -> None:
        """Test mapping of seawater (CP) curve with different transition."""
        curve = SNCurve.dnv_d(in_air=False)
        props = sn_curve_to_orcaflex(curve)
        assert props["SNCurveRegionBoundary"] == 1e6
        # 11.764 + 3.0 * 3 = 20.764
        np.testing.assert_allclose(props["SNCurveLogA1"], 20.764)

    def test_stress_factor_unity(self) -> None:
        """Test stress_factor=1.0 passes MPa values unchanged."""
        curve = SNCurve.dnv_d()
        props = sn_curve_to_orcaflex(curve, stress_factor=1.0)
        assert props["SNCurveLogA1"] == curve.log_a1

    def test_stress_factor_custom(self) -> None:
        """Test custom stress factor (e.g. Pa = 1e6)."""
        curve = SNCurve.dnv_d()
        props = sn_curve_to_orcaflex(curve, stress_factor=1e6)
        # log_a1_Pa = 12.164 + 3.0 * 6 = 30.164
        np.testing.assert_allclose(props["SNCurveLogA1"], 30.164)

    def test_does_not_include_log_a2(self, dnv_d_curve: SNCurve) -> None:
        """Test that LogA2 is NOT in output (it's read-only in OrcaFlex)."""
        props = sn_curve_to_orcaflex(dnv_d_curve)
        assert "SNCurveLogA2" not in props


# =============================================================================
# compare_results Tests
# =============================================================================


class TestCompareResults:
    """Tests for compare_results function."""

    def test_equal_damage(self) -> None:
        """Test comparison when both damages are equal."""
        result = compare_results(ofx_damage=0.01, sfx_damage=0.01)
        assert result.ratio == 1.0
        assert result.abs_diff == 0.0
        assert result.rel_diff == 0.0

    def test_ofx_higher(self) -> None:
        """Test comparison when OrcaFlex damage is higher."""
        result = compare_results(ofx_damage=0.012, sfx_damage=0.01)
        np.testing.assert_allclose(result.ratio, 1.2)
        np.testing.assert_allclose(result.abs_diff, 0.002)
        np.testing.assert_allclose(result.rel_diff, 0.2)

    def test_sfx_higher(self) -> None:
        """Test comparison when spectraflex damage is higher."""
        result = compare_results(ofx_damage=0.008, sfx_damage=0.01)
        np.testing.assert_allclose(result.ratio, 0.8)
        np.testing.assert_allclose(result.abs_diff, -0.002)
        np.testing.assert_allclose(result.rel_diff, -0.2)

    def test_zero_sfx_damage_with_nonzero_ofx(self) -> None:
        """Test comparison when spectraflex damage is zero."""
        result = compare_results(ofx_damage=0.01, sfx_damage=0.0)
        assert result.ratio == float("inf")
        assert result.rel_diff == float("inf")
        assert result.abs_diff == 0.01

    def test_both_zero(self) -> None:
        """Test comparison when both damages are zero."""
        result = compare_results(ofx_damage=0.0, sfx_damage=0.0)
        assert result.ratio == 1.0
        assert result.rel_diff == 0.0
        assert result.abs_diff == 0.0

    def test_returns_comparison_result(self) -> None:
        """Test return type is ComparisonResult."""
        result = compare_results(ofx_damage=0.01, sfx_damage=0.01)
        assert isinstance(result, ComparisonResult)

    def test_stores_input_values(self) -> None:
        """Test input values are stored in result."""
        result = compare_results(ofx_damage=0.015, sfx_damage=0.012)
        assert result.ofx_damage == 0.015
        assert result.sfx_damage == 0.012


# =============================================================================
# OrcaFlexFatigueResult Tests
# =============================================================================


class TestOrcaFlexFatigueResult:
    """Tests for OrcaFlexFatigueResult dataclass."""

    def test_create_result(self, sample_config: OrcaFlexFatigueConfig) -> None:
        """Test creating a result dataclass."""
        n_arclengths = 5
        n_theta = 16
        n_cases = 1

        overall = np.random.default_rng(42).random((n_arclengths, n_theta))
        lc_damage = np.random.default_rng(42).random((n_cases, n_arclengths, n_theta))
        theta = np.linspace(0, 360, n_theta, endpoint=False)
        arclengths = np.linspace(0, 100, n_arclengths)

        max_idx = np.unravel_index(np.argmax(overall), overall.shape)

        result = OrcaFlexFatigueResult(
            overall_damage=overall,
            load_case_damage=lc_damage,
            theta=theta,
            arclengths=arclengths,
            max_damage=float(overall[max_idx]),
            max_damage_arclength=float(arclengths[max_idx[0]]),
            max_damage_theta=float(theta[max_idx[1]]),
            config=sample_config,
        )

        assert result.overall_damage.shape == (n_arclengths, n_theta)
        assert result.load_case_damage.shape == (n_cases, n_arclengths, n_theta)
        assert result.max_damage > 0
        assert result.config is sample_config


# =============================================================================
# Integration Tests (require OrcFxAPI)
# =============================================================================


class TestOrcFxAPIIntegration:
    """Integration tests that require OrcFxAPI.

    These tests are skipped if OrcFxAPI is not installed.
    """

    @pytest.fixture(autouse=True)
    def _require_orcfxapi(self) -> None:
        pytest.importorskip("OrcFxAPI")

    def test_create_fatigue_analysis_import(self) -> None:
        """Test that create_fatigue_analysis can be imported with OrcFxAPI."""
        from spectraflex.orcaflex.fatigue import create_fatigue_analysis

        assert callable(create_fatigue_analysis)

    def test_extract_results_import(self) -> None:
        """Test that extract_results can be imported with OrcFxAPI."""
        from spectraflex.orcaflex.fatigue import extract_results

        assert callable(extract_results)

    def test_run_spectral_fatigue_import(self) -> None:
        """Test that run_spectral_fatigue can be imported with OrcFxAPI."""
        from spectraflex.orcaflex.fatigue import run_spectral_fatigue

        assert callable(run_spectral_fatigue)
