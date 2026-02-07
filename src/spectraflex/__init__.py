"""spectraflex - Transfer function identification and spectral response prediction.

Identify transfer functions from OrcaFlex white noise simulations and use them
for fast spectral response prediction.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export main modules for convenient access
from spectraflex import (
    identify,
    io,
    library,
    predict,
    spectrum,
    statistics,
    transfer_function,
)
from spectraflex.library import TransferFunctionLibrary

__all__ = [
    "__version__",
    "identify",
    "io",
    "library",
    "predict",
    "spectrum",
    "statistics",
    "transfer_function",
    "TransferFunctionLibrary",
]
