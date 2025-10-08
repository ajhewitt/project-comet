from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - exercised via integration tests when dependencies exist
    import healpy as _hp
except ModuleNotFoundError:  # pragma: no cover - fallback path used in CI
    _hp = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised via integration tests when dependencies exist
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - fallback path used in CI
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    import healpy as hp  # type: ignore
    import numpy as np
else:  # pragma: no cover - executed at runtime
    hp = _hp  # type: ignore[assignment]
    np = _np  # type: ignore[assignment]


@dataclass(slots=True)
class MapMetadata:
    """Structured metadata extracted from a HEALPix FITS map."""

    path: Path
    unit: str
    ordering: str
    coord_system: str
    nside: int
    pixel_window: str | None
    beam_fwhm_arcmin: float | None
    header: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        meta = {
            "path": self.path.as_posix(),
            "unit": self.unit,
            "ordering": self.ordering,
            "coord_system": self.coord_system,
            "nside": self.nside,
            "pixel_window": self.pixel_window,
            "beam_fwhm_arcmin": self.beam_fwhm_arcmin,
        }
        meta["header"] = {
            key: _jsonify(value) for key, value in self.header.items() if isinstance(key, str)
        }
        return meta


@dataclass(slots=True)
class MapData:
    """Container bundling map pixels with structured metadata."""

    pixels: Any
    metadata: MapMetadata

    def write_metadata_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.metadata.to_json_dict()
        payload["npix"] = int(self.pixels.size)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def _normalise_unit(raw: Any) -> str:
    if raw is None:
        return ""
    unit = str(raw).strip()
    return unit.upper()


def _jsonify(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("ascii", "ignore")
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _header_to_dict(header: Any) -> dict[str, Any]:
    if isinstance(header, dict):
        return {k: header[k] for k in header}
    header_dict: dict[str, Any] = {}
    for entry in header or []:
        if not entry:
            continue
        if isinstance(entry, (list, tuple)) and entry:
            key = entry[0]
            if not key:
                continue
            value = entry[1] if len(entry) > 1 else None
        else:
            continue
        if isinstance(key, bytes):
            key = key.decode("ascii", "ignore")
        if isinstance(value, bytes):
            value = value.decode("ascii", "ignore")
        if isinstance(key, str):
            header_dict.setdefault(key, value)
    return header_dict


def _extract_coord(header: dict[str, Any]) -> str:
    for key in ("COORDSYS", "COORDTYPE", "COORD", "COORD1"):
        value = header.get(key)
        if value is not None:
            coord = str(value).strip()
            if coord:
                return coord.upper()
    return ""


def _extract_pixel_window(header: dict[str, Any]) -> str | None:
    for key in ("PIXWIN", "PIXTYPE", "PIXFILE"):
        value = header.get(key)
        if value in (None, ""):
            continue
        return str(value).strip()
    return None


def _extract_beam_fwhm(header: dict[str, Any]) -> float | None:
    for key in ("FWHM", "BEAMFWHM", "FWHM_ARCMIN"):
        value = header.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _require_numpy() -> Any:
    if np is None:
        msg = (
            "numpy is required for FITS operations in comet.io_maps; "
            "install the optional 'maps' dependencies to enable this feature"
        )
        raise ModuleNotFoundError(msg)
    return np


def _require_healpy() -> Any:
    if hp is None:
        msg = (
            "healpy is required for FITS operations in comet.io_maps; "
            "install the optional 'maps' dependencies to enable this feature"
        )
        raise ModuleNotFoundError(msg)
    return hp


def _normalise_pixels(pixels: Any, path: Path) -> Any:
    numpy = _require_numpy()
    arr = numpy.asarray(pixels)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] >= 1:
        return arr[0]
    msg = f"Unexpected FITS shape for {path}: {arr.shape}"
    raise ValueError(msg)


def read_fits_map(path: Path) -> Any:
    return read_fits_map_with_meta(path).pixels


def read_fits_map_with_meta(path: Path) -> MapData:
    if not path.exists():
        raise FileNotFoundError(path)
    healpy = _require_healpy()
    numpy = _require_numpy()
    pixels, header = healpy.read_map(path.as_posix(), h=True)
    pixels = _normalise_pixels(pixels, path)
    header_dict = _header_to_dict(header)
    nside = int(header_dict.get("NSIDE", healpy.get_nside(pixels)))
    ordering = str(header_dict.get("ORDERING", "")).upper()
    meta = MapMetadata(
        path=path,
        unit=_normalise_unit(header_dict.get("TUNIT1")),
        ordering=ordering,
        coord_system=_extract_coord(header_dict),
        nside=nside,
        pixel_window=_extract_pixel_window(header_dict),
        beam_fwhm_arcmin=_extract_beam_fwhm(header_dict),
        header=header_dict,
    )
    return MapData(pixels=numpy.asarray(pixels), metadata=meta)


def get_nside(m: Any) -> int:
    healpy = _require_healpy()
    return int(healpy.get_nside(m))


def map_info(m: Any) -> dict:
    numpy = _require_numpy()
    arr = numpy.asarray(m)
    return {
        "nside": get_nside(arr),
        "npix": int(arr.size),
        "f_sky": float(numpy.isfinite(arr).mean()),
    }
