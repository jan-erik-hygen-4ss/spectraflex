"""I/O utilities for spectraflex.

Provides functions for saving and loading TransferFunction datasets
and TransferFunctionLibrary objects to various formats.
"""

from __future__ import annotations

from spectraflex.io.netcdf import (
    load_library,
    load_transfer_function,
    save_library,
    save_transfer_function,
)

__all__ = [
    "load_library",
    "load_transfer_function",
    "save_library",
    "save_transfer_function",
]
