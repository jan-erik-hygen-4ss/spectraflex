"""spectraflex - Transfer function identification and spectral response prediction.

Identify transfer functions from OrcaFlex white noise simulations and use them
for fast spectral response prediction.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export main functions for convenient access
from spectraflex import identify, predict, spectrum, statistics, transfer_function

__all__ = [
    "__version__",
    "identify",
    "predict",
    "spectrum",
    "statistics",
    "transfer_function",
]
