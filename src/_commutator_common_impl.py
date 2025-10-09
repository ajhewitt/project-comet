from __future__ import annotations

import json
from collections.abc import Mapping
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - dependency is optional in CI
    import healpy as hp
except Exception as exc:  # pragma: no cover
    hp = None  # type: ignore[assignment]
    _HP_ERROR = exc
else:  # pragma: no cover
    _HP_ERROR = None

try:  # pragma: no cover - dependency is optional in CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NP_ERROR = exc
else:  # pragma: no cover
    _NP_ERROR = None

from comet.config import load_prereg

try:  # pragma: no cover - dependency is optional in CI
    from comet.masking import build_mask as _build_mask
    from comet.masking import effective_f_sky as _effective_f_sky
except ModuleNotFoundError as exc:  # pragma: no cover
    _build_mask = None  # type: ignore[assignment]
    _effective_f_sky = None  # type: ignore[assignment]
    _MASKING_ERROR = exc
else:  # pragma: no cover
    _MASKING_ERROR = None
from comet.namaster_utils import WindowConfig, parse_window_config
from comet.namaster_utils import bandpowers as _bandpowers
from comet.namaster_utils import field_from_map as _field_from_map

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


def _require_nmt() -> Any:
    module = globals().get("_nmt")
    if module is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_NMT_ERROR")
        raise RuntimeError(
            "pymaster (NaMaster) is required for commutator helpers. "
            "Install it from conda-forge as 'namaster'."
        ) from error
    return module


def _require_healpy() -> Any:
    module = globals().get("hp")
    if module is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_HP_ERROR")
        raise RuntimeError(
            "healpy is required for commutator map helpers. "
            "Install it from conda-forge as 'healpy'."
        ) from error
    return module


__all__ = [
    "MapBundle",
    "build_mask",
    "effective_f_sky",
    "infer_bin_lmax",
    "load_bins_from_prereg",
    "load_windows_from_prereg",
    "nm_bandpowers",
    "nm_bins_from_config",
    "nm_bins_from_params",
    "nm_field_from_scalar",
    "WindowConfig",
    "read_map",
    "save_json",
    "save_npy",
    "summary_line",
]


@dataclass
class MapBundle:
    name: str
    map: np.ndarray
    mask: np.ndarray
    nside: int


def read_map(path: Path, quick_nside: int | None = None) -> np.ndarray:
    numpy = _require_numpy()
    healpy = _require_healpy()
    m = healpy.read_map(path.as_posix())
    if m.ndim != 1:
        arr = numpy.asarray(m)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            m = arr[0]
        else:
            raise ValueError(f"Unexpected FITS shape for {path}: {arr.shape}")
    if quick_nside is not None and healpy.get_nside(m) != quick_nside:
        # Power = -2 for temperature-like scalar field when downgrading
        m = healpy.ud_grade(m, nside_out=quick_nside, power=-2)
    return m


def build_mask(
    m: np.ndarray,
    threshold_sigma: float = 10.0,
    apod_arcmin: float | None = None,
) -> np.ndarray:
    build, _ = _require_masking()
    return build(m, threshold_sigma=threshold_sigma, apod_arcmin=apod_arcmin)


def effective_f_sky(mask: np.ndarray) -> float:
    _, eff = _require_masking()
    return eff(mask)


def nm_bins_from_params(
    nside: int,
    lmax: int | None = None,
    nlb: int = 50,
    lmin: int | None = None,
) -> nmt.NmtBin:
    """
    Construct simple linear-ℓ binning based on map nside using NaMaster's helper.
    Uses NmtBin.from_nside_linear, which chooses lmax ~ 3*nside-1 internally.
    """
    if nlb <= 0:
        raise ValueError("nlb must be positive")
    kwargs: dict[str, Any] = {"nside": nside, "nlb": nlb}
    lmin_val: int | None = None
    if lmin is not None:
        if lmin < 0:
            raise ValueError("lmin must be non-negative")
        lmin_val = int(lmin)
        kwargs["lmin"] = lmin_val
    lmax_val: int | None = None
    if lmax is not None:
        if lmax <= 0:
            raise ValueError("lmax must be positive")
        lmax_val = int(lmax)
        kwargs["lmax"] = lmax_val

    module = _require_nmt()
    numpy = _require_numpy()

    try:
        return module.NmtBin.from_nside_linear(**kwargs)
    except TypeError:
        if lmin_val is None:
            raise
        # Older NaMaster releases do not accept lmin in from_nside_linear; fall back to
        # constructing explicit edges that honour the requested lower bound.
        from_edges = getattr(module.NmtBin, "from_edges", None)
        if from_edges is None:
            raise RuntimeError("NaMaster does not expose NmtBin.from_edges; cannot enforce lmin")
        if lmax_val is None:
            lmax_val = 3 * nside - 1
        ell_min = numpy.arange(lmin_val, lmax_val + 1, nlb, dtype=int)
        ell_max = numpy.minimum(ell_min + nlb - 1, lmax_val)
        return from_edges(ell_min, ell_max)


def infer_bin_lmax(
    bins: nmt.NmtBin,
    *,
    bins_meta: Mapping[str, Any] | None = None,
    fallbacks: tuple[int | None, ...] | list[int | None] = (),
) -> int | None:
    """Infer the maximum multipole represented by ``bins``.

    Parameters
    ----------
    bins
        NaMaster binning object.
    bins_meta
        Optional metadata associated with the bins, typically sourced from the
        preregistration configuration. When provided, the function will prefer a
        positive ``lmax`` entry from this mapping.
    fallbacks
        Additional candidate values to consider when ``bins`` does not expose an
        explicit maximum multipole. Any positive entries are returned in the
        order supplied after consulting ``bins`` and ``bins_meta``.
    """

    candidates: list[int | None] = []

    lmax_attr = getattr(bins, "lmax", None)
    if lmax_attr is not None:
        candidates.append(int(lmax_attr))

    if bins_meta is not None and isinstance(bins_meta, Mapping):
        meta_lmax = bins_meta.get("lmax")
        if meta_lmax is not None:
            candidates.append(int(meta_lmax))

    get_ell_list = getattr(bins, "get_ell_list", None)
    if callable(get_ell_list):
        try:
            ell_list = get_ell_list()
        except TypeError:
            ell_list = None
        if ell_list is not None:
            ell_max: Any | None
            if isinstance(ell_list, tuple):
                if len(ell_list) >= 2:
                    ell_max = ell_list[1]
                elif len(ell_list) == 1:
                    ell_max = ell_list[0]
                else:
                    ell_max = None
            else:
                ell_max = ell_list
            if ell_max is not None:
                try:
                    candidates.append(int(ell_max[-1]))
                except (TypeError, IndexError):
                    pass

    get_effective_ells = getattr(bins, "get_effective_ells", None)
    if callable(get_effective_ells):
        try:
            effective = get_effective_ells()
        except TypeError:
            effective = None
        if effective is not None:
            try:
                value = float(max(effective))
            except ValueError:
                value = float("nan")
            if value > 0 and math.isfinite(value):
                candidates.append(int(math.ceil(value)))

    for candidate in candidates:
        if candidate is not None and int(candidate) > 0:
            return int(candidate)

    for fallback in fallbacks:
        if fallback is not None and int(fallback) > 0:
            return int(fallback)

    return None


def nm_bins_from_config(nside: int, bins_cfg: Mapping[str, Any]) -> nmt.NmtBin:
    """Create NaMaster bins from a preregistration ``bins`` configuration block."""

    if "nlb" not in bins_cfg:
        raise KeyError("bins configuration requires 'nlb'")

    nlb = int(bins_cfg["nlb"])
    lmin = bins_cfg.get("lmin")
    lmax = bins_cfg.get("lmax")
    return nm_bins_from_params(
        nside=nside,
        nlb=nlb,
        lmin=None if lmin is None else int(lmin),
        lmax=None if lmax is None else int(lmax),
    )


def load_bins_from_prereg(prereg_path: Path, nside: int) -> tuple[nmt.NmtBin, Mapping[str, Any]]:
    """Load the prereg YAML and construct NaMaster bins from its ``ells.bins`` block."""

    prereg = load_prereg(prereg_path)
    bins_cfg = prereg.get("ells", {}).get("bins")
    if bins_cfg is None:
        raise KeyError("prereg configuration is missing 'ells.bins'")
    bins = nm_bins_from_config(nside=nside, bins_cfg=bins_cfg)
    return bins, bins_cfg


def load_windows_from_prereg(prereg_path: Path) -> WindowConfig:
    """Load window/deconvolution settings from the preregistration YAML."""

    prereg = load_prereg(prereg_path)
    windows_cfg = prereg.get("windows") if isinstance(prereg, Mapping) else None
    return parse_window_config(windows_cfg)


def nm_field_from_scalar(
    m: np.ndarray,
    mask: np.ndarray,
    *,
    lmax: int | None = None,
) -> nmt.NmtField:
    """Create a NaMaster scalar field with an optional harmonic band-limit."""

    return _field_from_map(m, mask, lmax=lmax)


def nm_bandpowers(
    f1: nmt.NmtField,
    f2: nmt.NmtField,
    b: nmt.NmtBin,
    *,
    window_config: WindowConfig | None = None,
    field_names: tuple[str, str] = ("field_1", "field_2"),
) -> np.ndarray:
    return _bandpowers(f1, f2, b, window_config=window_config, field_names=field_names)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy = _require_numpy()
    numpy.save(path.as_posix(), arr)


def summary_line(msg: str) -> None:
    print(json.dumps({"msg": msg}))


def _require_numpy() -> Any:
    module = globals().get("np")
    if module is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_NP_ERROR")
        raise ModuleNotFoundError(
            "numpy is required for commutator helpers. Install it from conda-forge as 'numpy'."
        ) from error
    return module


def _require_masking() -> tuple[Any, Any]:
    build = globals().get("_build_mask")
    eff = globals().get("_effective_f_sky")
    if build is None or eff is None:  # pragma: no cover - exercised when dependency missing
        error = globals().get("_MASKING_ERROR")
        raise ModuleNotFoundError(
            "numpy is required for masking helpers. Install it from conda-forge as 'numpy'."
        ) from error
    return build, eff
