"""Pytest configuration and shared fixtures for spectraflex tests."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "orcaflex: marks tests as requiring OrcFxAPI (deselect with '-m \"not orcaflex\"')"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def frequency_array() -> np.ndarray:
    """Standard frequency array for testing (0.01 to 0.5 Hz, 256 points)."""
    return np.linspace(0.01, 0.5, 256)


@pytest.fixture
def dt() -> float:
    """Standard sample interval for time series tests."""
    return 0.1


@pytest.fixture
def sample_duration() -> float:
    """Standard simulation duration for tests (seconds)."""
    return 512.0
