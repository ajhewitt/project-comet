"""Helper utilities for working with NaMaster (pymaster)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - dependency is optional in CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NP_ERROR = exc
else:  # pragma: no cover
    _NP_ERROR = None

try:  # pragma: no cover - dependency is optional in CI
    import healpy as hp
except Exception as exc:  # pragma: no cover
    hp = None  # type: ignore[assignment]
    _HP_ERROR = exc
else:  # pragma: no cover
    _HP_ERROR = None

try:  # pragma: no cover - dependency is optional in CI
    import pymaster as _nmt
except Exception as exc:  # pragma: no cover
    _nmt = None  # type: ignore[assignment]
    _NMT_ERROR = exc
else:  # pragma: no cover
    _NMT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - hints only
    import pymaster as nmt  # type: ignore
else:  # pragma: no cover - runtime attribute filled via _require_nmt
    nmt = _nmt  # type: ignore[assignment]


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


def _require_nmt() -> Any:
    module = globals().get("_nmt")
    if module is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_NMT_ERROR")
        raise RuntimeError(
            "pymaster (NaMaster) is required for NaMaster bandpower helpers. "
            "Install it from conda-forge as 'namaster'."
        ) from error
    return module


def _require_numpy() -> Any:
    module = globals().get("np")
    if module is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_NP_ERROR")
        raise ModuleNotFoundError(
            "numpy is required for NaMaster helpers. Install it from conda-forge as 'numpy'."
        ) from error
    return module


def _workspace_from_fields(module: Any, field_1: Any, field_2: Any, bins: Any) -> Any:
    workspace_cls = module.NmtWorkspace
    from_fields = getattr(workspace_cls, "from_fields", None)
    if callable(from_fields):
        return from_fields(field_1, field_2, bins)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The bare constructor for `NmtWorkspace` objects is deprecated",
            category=DeprecationWarning,
        )
        try:
            return workspace_cls(field_1, field_2, bins)
        except TypeError:
            workspace = workspace_cls()

    workspace.compute_coupling_matrix(field_1, field_2, bins)
    return workspace


def _as_1d_array(values: Sequence[float] | Any) -> list[float]:
    if np is not None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D array, received shape {arr.shape}")
        return arr.astype(float).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        out: list[float] = []
        for item in values:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                raise ValueError("Expected 1-D sequence")
            out.append(float(item))
        return out
    return [float(values)]


@cache
def pixel_window(nside: int, lmax: int | None = None) -> np.ndarray | list[float]:
    """Return the scalar HEALPix pixel window function for ``nside`` up to ``lmax``."""

    if nside <= 0:
        raise ValueError("nside must be positive")
    numpy = _require_numpy()
    healpy = _require_healpy()
    window = healpy.sphtfunc.pixwin(nside, pol=False, lmax=lmax)
    return numpy.asarray(window, dtype=float)


@cache
def gaussian_beam_window(fwhm_arcmin: float, lmax: int) -> np.ndarray | list[float]:
    """Return a Gaussian beam transfer function sampled up to ``lmax``."""

    if lmax <= 0:
        raise ValueError("lmax must be positive")
    if fwhm_arcmin <= 0.0:
        raise ValueError("fwhm_arcmin must be positive")
    sigma = math.radians(float(fwhm_arcmin) / 60.0) / math.sqrt(8.0 * math.log(2.0))
    values = [math.exp(-0.5 * ell * (ell + 1.0) * sigma**2) for ell in range(lmax + 1)]
    if np is not None:
        return np.asarray(values, dtype=float)
    return values


def _window_response(window: Sequence[float] | None, ells: Sequence[float]) -> list[float]:
    if window is None:
        return [1.0 for _ in ells]
    samples = _as_1d_array(window)
    if not samples:
        return [1.0 for _ in ells]
    base = [float(i) for i in range(len(samples))]
    left = samples[0]
    right = samples[-1]

    def _interp(x: float) -> float:
        if x <= base[0]:
            return left
        if x >= base[-1]:
            return right
        for idx in range(1, len(base)):
            if x <= base[idx]:
                x0 = base[idx - 1]
                x1 = base[idx]
                y0 = samples[idx - 1]
                y1 = samples[idx]
                if x1 == x0:
                    return y1
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return right

    return [_interp(float(val)) for val in ells]


def apply_window_corrections(
    cl: Sequence[float] | Any,
    ells: Sequence[float] | Any,
    *,
    pixel_windows: tuple[Sequence[float] | None, Sequence[float] | None] | None = None,
    beam_windows: tuple[Sequence[float] | None, Sequence[float] | None] | None = None,
    eps: float = 1e-12,
) -> list[float] | np.ndarray:
    """Deconvolve the provided bandpowers by pixel and/or beam windows."""

    cl_arr = _as_1d_array(cl)
    ell_arr = _as_1d_array(ells)
    if len(cl_arr) != len(ell_arr):
        raise ValueError("bandpowers and ell arrays must share the same length")

    correction = [1.0 for _ in ell_arr]

    if pixel_windows is not None:
        pw1, pw2 = pixel_windows
        for response in (_window_response(pw1, ell_arr), _window_response(pw2, ell_arr)):
            correction = [c * r for c, r in zip(correction, response)]

    if beam_windows is not None:
        bw1, bw2 = beam_windows
        for response in (_window_response(bw1, ell_arr), _window_response(bw2, ell_arr)):
            correction = [c * r for c, r in zip(correction, response)]

    safe_correction = [max(c, eps) for c in correction]
    result = [val / corr for val, corr in zip(cl_arr, safe_correction)]
    if np is not None:
        return np.asarray(result, dtype=float)
    return result


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
    module = _require_nmt()
    return module.NmtBin.from_lmax(lmax=lmax, nlb=nlb)


def field_from_map(
    m: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    lmax: int | None = None,
) -> nmt.NmtField:
    """Build a spin-0 NaMaster field from a scalar map and an optional mask.

    Parameters
    ----------
    m
        The scalar map that should be wrapped in a :class:`~pymaster.NmtField`.
    mask
        Optional mask to apply when creating the field. If omitted a mask is
        constructed from the finite pixels of ``m``.
    lmax
        Optional harmonic band-limit to apply. When provided it is forwarded to
        :class:`~pymaster.NmtField` so that the field and any accompanying
        :class:`~pymaster.NmtBin` instances share the same maximum multipole.
    """

    numpy = _require_numpy()
    if mask is None:
        mask = numpy.isfinite(m).astype(float)
    module = _require_nmt()
    return module.NmtField(mask, [m], lmax=lmax)


def bandpowers(
    f1: nmt.NmtField,
    f2: nmt.NmtField,
    b: nmt.NmtBin,
    *,
    window_config: WindowConfig | None = None,
    field_names: tuple[str, str] = ("field_1", "field_2"),
) -> np.ndarray:
    """Compute decoupled pseudo-C_ell bandpowers with optional window corrections."""

    module = _require_nmt()
    numpy = _require_numpy()
    workspace = _workspace_from_fields(module, f1, f2, b)
    cl_coupled = module.compute_coupled_cell(f1, f2)
    cl_decoupled = workspace.decouple_cell(cl_coupled)[0]

    if window_config is None:
        return cl_decoupled

    needs_pixel = window_config.apply_pixel_window
    needs_beam = window_config.deconvolve_beam and bool(window_config.beam_fwhm_arcmin)
    if not needs_pixel and not needs_beam:
        return cl_decoupled

    ell_eff = numpy.asarray(b.get_effective_ells(), dtype=float)
    lmax = int(math.ceil(float(ell_eff.max()))) if ell_eff.size else 0

    pixel_windows: tuple[Sequence[float] | None, Sequence[float] | None] | None = None
    beam_windows: tuple[Sequence[float] | None, Sequence[float] | None] | None = None

    def _field_nside(field: nmt.NmtField) -> int:
        nside_attr = getattr(field, "nside", None)
        if nside_attr is not None:
            return int(nside_attr)
        mask = getattr(field, "mask", None)
        if mask is not None:
            healpy = _require_healpy()
            return int(healpy.npix2nside(numpy.asarray(mask).size))
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
