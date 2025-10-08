"""Compatibility shim exposing commutator utilities to top-level imports.

This ensures modules and tests that expect ``commutator_common`` at the
repository root can import the shared helpers regardless of the execution
context. All functionality lives in :mod:`scripts.commutator_common`.
"""

from scripts.commutator_common import (
    MapBundle,
    build_mask,
    effective_f_sky,
    load_bins_from_prereg,
    nm_bandpowers,
    nm_bins_from_config,
    nm_bins_from_params,
    nm_field_from_scalar,
    read_map,
    save_json,
    save_npy,
    summary_line,
)

__all__ = [
    "MapBundle",
    "build_mask",
    "effective_f_sky",
    "load_bins_from_prereg",
    "nm_bandpowers",
    "nm_bins_from_config",
    "nm_bins_from_params",
    "nm_field_from_scalar",
    "read_map",
    "save_json",
    "save_npy",
    "summary_line",
]
