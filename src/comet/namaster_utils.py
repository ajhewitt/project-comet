"""Helper utilities for working with NaMaster (pymaster)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

try:  # pragma: no cover - dependency is optional in CI
    import healpy as hp
except Exception as exc:  # pragma: no cover
    hp = None  # type: ignore[assignment]
    _HP_ERROR = exc
else:  # pragma: no cover
    _HP_ERROR = None

try:  # pragma: no cover - dependency is optional in CI
    import pymaster as nmt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pymaster (NaMaster) is required for namaster_utils. "
        "Install it from conda-forge as 'namaster'."
    ) from exc


def _require_healpy() -> Any:
    if hp is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_HP_ERROR")
        if isinstance(error, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "healpy is required for pixel/beam window corrections. "
                "Install it from conda-forge as 'healpy'."
            ) from error
        raise RuntimeError("Failed to import healpy") from error
    return hp


def _as_1d_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, received shape {arr.shape}")
    return arr


@cache
def pixel_window(nside: int, lmax: int | None = None) -> np.ndarray:
    """Return the scalar HEALPix pixel window function for ``nside`` up to ``lmax``."""

    if nside <= 0:
        raise ValueError("nside must be positive")
    healpy = _require_healpy()
    window = healpy.sphtfunc.pixwin(nside, pol=False, lmax=lmax)
    return np.asarray(window, dtype=float)


@cache
def gaussian_beam_window(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """Return a Gaussian beam transfer function sampled up to ``lmax``."""

    if lmax <= 0:
        raise ValueError("lmax must be positive")
    if fwhm_arcmin <= 0.0:
        raise ValueError("fwhm_arcmin must be positive")
    healpy = _require_healpy()
    fwhm_rad = math.radians(float(fwhm_arcmin) / 60.0)
    window = healpy.gauss_beam(fwhm=fwhm_rad, lmax=lmax)
    return np.asarray(window, dtype=float)


def _window_response(window: np.ndarray | None, ells: np.ndarray) -> np.ndarray:
    if window is None:
        return np.ones_like(ells, dtype=float)
    base_ell = np.arange(window.size, dtype=float)
    response = np.interp(ells, base_ell, window, left=window[0], right=window[-1])
    return response


def apply_window_corrections(
    cl: Sequence[float] | np.ndarray,
    ells: Sequence[float] | np.ndarray,
    *,
    pixel_windows: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    beam_windows: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Deconvolve the provided bandpowers by pixel and/or beam windows."""

    cl_arr = _as_1d_array(cl)
    ell_arr = _as_1d_array(ells)
    if cl_arr.size != ell_arr.size:
        raise ValueError("bandpowers and ell arrays must share the same length")

    correction = np.ones_like(ell_arr, dtype=float)

    if pixel_windows is not None:
        pw1, pw2 = pixel_windows
        correction *= _window_response(pw1, ell_arr)
        correction *= _window_response(pw2, ell_arr)

    if beam_windows is not None:
        bw1, bw2 = beam_windows
        correction *= _window_response(bw1, ell_arr)
        correction *= _window_response(bw2, ell_arr)

    correction = np.clip(correction, eps, None)
    return cl_arr / correction


@dataclass(frozen=True)
class WindowConfig:
    """Configuration flags controlling NaMaster window deconvolutions."""

    apply_pixel_window: bool = False
    deconvolve_beam: bool = False
    beam_fwhm_arcmin: Mapping[str, float] = field(default_factory=dict)

    def beam_for(self, field: str) -> float | None:
        if field in self.beam_fwhm_arcmin:
            return float(self.beam_fwhm_arcmin[field])
        if "default" in self.beam_fwhm_arcmin:
            return float(self.beam_fwhm_arcmin["default"])
        return None

    def to_metadata(self) -> dict[str, Any]:
        data = {
            "apply_pixel_window": self.apply_pixel_window,
            "deconvolve_beam": self.deconvolve_beam,
        }
        if self.beam_fwhm_arcmin:
            data["beam_fwhm_arcmin"] = {
                key: float(value) for key, value in sorted(self.beam_fwhm_arcmin.items())
            }
        else:
            data["beam_fwhm_arcmin"] = {}
        return data


def parse_window_config(cfg: Mapping[str, Any] | None) -> WindowConfig:
    if cfg is None:
        return WindowConfig()

    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, Mapping):
            for key in ("apply", "enabled", "deconvolve"):
                if key in value:
                    return _as_bool(value[key], default)
        return default

    apply_pixel = False
    pixel_cfg = cfg.get("pixel") if isinstance(cfg, Mapping) else None
    if pixel_cfg is None and isinstance(cfg, Mapping):
        pixel_cfg = cfg.get("pixel_window")
    if pixel_cfg is not None:
        apply_pixel = _as_bool(pixel_cfg)

    deconvolve_beam = False
    beam_map: dict[str, float] = {}
    beam_cfg = cfg.get("beam") if isinstance(cfg, Mapping) else None
    if beam_cfg is None and isinstance(cfg, Mapping):
        beam_cfg = cfg.get("beam_window")

    if beam_cfg is not None:
        deconvolve_beam = _as_bool(beam_cfg)
        if isinstance(beam_cfg, Mapping):
            fwhm_cfg = beam_cfg.get("fwhm_arcmin")
            if isinstance(fwhm_cfg, Mapping):
                for key, value in fwhm_cfg.items():
                    if value is None:
                        continue
                    beam_map[str(key)] = float(value)
            elif fwhm_cfg is not None:
                beam_map["default"] = float(fwhm_cfg)
            else:
                default_val = beam_cfg.get("default")
                if default_val is not None:
                    beam_map["default"] = float(default_val)
        elif isinstance(beam_cfg, (int, float)) and not isinstance(beam_cfg, bool):
            beam_map["default"] = float(beam_cfg)

    return WindowConfig(
        apply_pixel_window=apply_pixel,
        deconvolve_beam=deconvolve_beam,
        beam_fwhm_arcmin=beam_map,
    )


def make_bins(lmax: int, nlb: int) -> nmt.NmtBin:
    """Construct a simple linear-ℓ binning up to ``lmax`` with width ``nlb``."""

    if lmax <= 0 or nlb <= 0:
        raise ValueError("lmax and nlb must be positive")
    return nmt.NmtBin.from_lmax(lmax=lmax, nlb=nlb)


def field_from_map(m: np.ndarray, mask: np.ndarray | None = None) -> nmt.NmtField:
    """Build a spin-0 NaMaster field from a scalar map and an optional mask."""

    if mask is None:
        mask = np.isfinite(m).astype(float)
    return nmt.NmtField(mask, [m])


def bandpowers(
    f1: nmt.NmtField,
    f2: nmt.NmtField,
    b: nmt.NmtBin,
    *,
    window_config: WindowConfig | None = None,
    field_names: tuple[str, str] = ("field_1", "field_2"),
) -> np.ndarray:
    """Compute decoupled pseudo-C_ell bandpowers with optional window corrections."""

    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(f1, f2, b)
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = workspace.decouple_cell(cl_coupled)[0]

    if window_config is None:
        return cl_decoupled

    needs_pixel = window_config.apply_pixel_window
    needs_beam = window_config.deconvolve_beam and bool(window_config.beam_fwhm_arcmin)
    if not needs_pixel and not needs_beam:
        return cl_decoupled

    ell_eff = np.asarray(b.get_effective_ells(), dtype=float)
    lmax = int(math.ceil(float(ell_eff.max()))) if ell_eff.size else 0

    pixel_windows: tuple[np.ndarray | None, np.ndarray | None] | None = None
    beam_windows: tuple[np.ndarray | None, np.ndarray | None] | None = None

    def _field_nside(field: nmt.NmtField) -> int:
        nside_attr = getattr(field, "nside", None)
        if nside_attr is not None:
            return int(nside_attr)
        mask = getattr(field, "mask", None)
        if mask is not None:
            healpy = _require_healpy()
            return int(healpy.npix2nside(np.asarray(mask).size))
        raise AttributeError("Could not infer nside for NaMaster field")

    if needs_pixel:
        if lmax <= 0:
            raise ValueError("Cannot apply pixel window without valid ell range")
        pixel_windows = (
            pixel_window(_field_nside(f1), lmax=lmax),
            pixel_window(_field_nside(f2), lmax=lmax),
        )

    if needs_beam:
        if lmax <= 0:
            raise ValueError("Cannot deconvolve beam without valid ell range")
        beam_windows = []
        for idx, field in enumerate(field_names[:2]):
            fwhm = window_config.beam_for(field)
            if fwhm is None:
                beam_windows.append(None)
            else:
                beam_windows.append(gaussian_beam_window(fwhm, lmax))
        beam_windows = tuple(beam_windows)  # type: ignore[assignment]

    return apply_window_corrections(
        cl_decoupled,
        ell_eff,
        pixel_windows=pixel_windows,
        beam_windows=beam_windows,
    )


__all__ = [
    "WindowConfig",
    "apply_window_corrections",
    "bandpowers",
    "field_from_map",
    "gaussian_beam_window",
    "make_bins",
    "parse_window_config",
    "pixel_window",
]
