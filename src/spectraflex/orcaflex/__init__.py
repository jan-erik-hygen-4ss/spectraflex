"""OrcaFlex integration for spectraflex.

Provides utilities for:
- Generating white noise model files (.yml variation files)
- Attaching post-calculation actions for automatic spectrum extraction
- Extracting time histories from .sim files
- Batch parameter sweep generation
"""

from __future__ import annotations

from spectraflex.orcaflex.batch import generate_case_matrix
from spectraflex.orcaflex.extract import (
    extract_time_histories,
    extract_wave_elevation,
    get_analysis_period,
    get_sample_interval,
)
from spectraflex.orcaflex.post_calc import (
    attach_post_calc,
    get_post_calc_script,
)
from spectraflex.orcaflex.white_noise import (
    generate,
    generate_batch,
)

__all__ = [
    # white_noise
    "generate",
    "generate_batch",
    # post_calc
    "attach_post_calc",
    "get_post_calc_script",
    # extract
    "extract_time_histories",
    "extract_wave_elevation",
    "get_analysis_period",
    "get_sample_interval",
    # batch
    "generate_case_matrix",
]
