from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in CI
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML absent
    yaml = None  # type: ignore[assignment]

from .config import get_data_dir

_NUMPY = None
_ASTROPY_FITS = None


def _maybe_import(name: str) -> Any:
    spec = importlib.util.find_spec(name)
    if spec is None:  # pragma: no cover - tiny branch
        return None
    return importlib.import_module(name)


def _get_numpy():
    global _NUMPY
    if _NUMPY is None:
        _NUMPY = _maybe_import("numpy")
    return _NUMPY


def _get_astropy_fits():
    global _ASTROPY_FITS
    if _ASTROPY_FITS is None:
        _ASTROPY_FITS = _maybe_import("astropy.io.fits")
    return _ASTROPY_FITS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_artifacts_dir() -> Path:
    env = os.getenv("COMET_ARTIFACTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "artifacts").resolve()


def _resolve_map_path(value: Any, default_name: str, data_dir: Path) -> Path:
    if isinstance(value, str) and value.strip():
        candidate = Path(value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = (_repo_root() / candidate).resolve()
        return candidate
    return (data_dir / default_name).resolve()


def _load_intensity_column(path: Path):
    numpy = _get_numpy()
    fits = _get_astropy_fits()
    if numpy is None or fits is None or not path.exists():
        return None
    with fits.open(path.as_posix(), memmap=True) as hdul:  # type: ignore[call-arg]
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            columns = getattr(data, "columns", None)
            if columns is None:
                continue
            numeric_names: Iterable[str]
            preferred = ["I", "SIGNAL", "VAL", "VALUE"]
            names = list(getattr(columns, "names", []) or [])
            numeric_names = preferred + [name for name in names if name not in preferred]
            for name in numeric_names:
                if name not in names:
                    continue
                col = data[name]
                if not numpy.issubdtype(col.dtype, numpy.number):
                    continue
                arr = numpy.asarray(col, dtype=numpy.float64)
                return numpy.nan_to_num(arr, copy=False)
    return None


def _git_info() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out = ""
        return out

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": "1" if _run(["git", "status", "--porcelain"]) else "0",
    }


def _read_yaml(p: str | Path | None) -> dict:
    if not p:
        return {}
    pth = Path(p)
    if not pth.exists():
        return {}
    if yaml is None:
        return {}
    with pth.open() as f:
        return yaml.safe_load(f) or {}


def _cfg_hash(prereg: dict, paths: dict) -> str:
    blob = json.dumps({"prereg": prereg, "paths": paths}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _quick_products(
    cmb_path: Path,
    lens_path: Path,
    artifacts_dir: Path,
    nbins: int,
    check_only: bool,
) -> dict[str, Any]:
    numpy = _get_numpy()
    if numpy is None:
        return {
            "nbins": 0,
            "z": 0.0,
            "notes": "numpy_missing",
            "artifacts": [],
        }

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if check_only:
        return {
            "nbins": 0,
            "z": 0.0,
            "notes": "check_only",
            "artifacts": [],
        }

    cmb = _load_intensity_column(cmb_path)
    lens = _load_intensity_column(lens_path)

    rng_seed = int(os.getenv("COMET_NSIMS", "5"))
    rng_cov_seed = int(os.getenv("COMET_COV_SEED", "42"))

    if cmb is not None and lens is not None:
        x = cmb - float(numpy.nanmedian(cmb))
        chunks = numpy.array_split(x, nbins)
        delta = numpy.array([float(numpy.nanmean(chunk)) for chunk in chunks], dtype=numpy.float64)
        scale = float(numpy.nanstd(delta))
        if not numpy.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        delta = (delta / scale) * 0.1
        delta = delta - float(numpy.nanmean(delta))
        delta = delta.astype(numpy.float32)
        path_note = "real-data"
    else:
        rng = numpy.random.default_rng(rng_seed)
        delta = rng.normal(0.0, 0.1, size=nbins).astype(numpy.float32)
        path_note = "synthetic"

    var = float(numpy.var(delta))
    eps = 1e-3 * max(var, 1e-6)
    rng_cov = numpy.random.default_rng(rng_cov_seed)
    a = rng_cov.normal(0.0, eps, size=(nbins, nbins)).astype(numpy.float32)
    s = (a + a.T) / 2.0
    cov = (numpy.eye(nbins, dtype=numpy.float32) * var) + s
    w, v = numpy.linalg.eigh(cov.astype(numpy.float64))
    floor = 1e-6 * max(var, 1e-6)
    w = numpy.clip(w, floor, None)
    cov = (v * w) @ v.T
    cov = cov.astype(numpy.float32)

    diag = numpy.clip(numpy.diag(cov), 1e-12, None)
    std = numpy.sqrt(diag, dtype=numpy.float64)
    z = numpy.divide(delta, std, out=numpy.zeros_like(delta), where=std > 0)
    z = z.astype(numpy.float32)

    artifacts = []
    for name, array in (
        ("delta.npy", delta),
        ("cov_delta.npy", cov),
        ("z_scores.npy", z),
    ):
        numpy.save(artifacts_dir / name, array)
        artifacts.append(name)

    summary = {
        "nbins": int(delta.size),
        "delta_mean": float(numpy.mean(delta)),
        "delta_std": float(numpy.std(delta)),
        "z_std": float(numpy.std(z)),
        "mean_abs_z": float(numpy.mean(numpy.abs(z))),
        "max_abs_z": float(numpy.max(numpy.abs(z))),
        "min_eig": float(numpy.min(w)),
        "notes": path_note,
        "z": float(numpy.sqrt(float(numpy.mean(z**2)))),
        "artifacts": artifacts,
    }
    return summary


def run_pipeline(
    prereg_path: str | Path | None = None,
    paths_path: str | Path | None = None,
    ordering: str = "both",
    check_only: bool = False,
) -> dict:
    """Top-level runner: reads configs, inspects maps, writes metadata stub."""
    now = datetime.now(timezone.utc).isoformat()

    prereg = _read_yaml(prereg_path)
    if not isinstance(prereg, dict):
        prereg = {}
    paths = _read_yaml(paths_path)
    if not isinstance(paths, dict):
        paths = {}
    cfg_hash = _cfg_hash(prereg, paths)

    # Optional guardrail: refuse to run on dirty trees unless explicitly allowed
    gi = _git_info()
    if gi["dirty"] == "1" and not bool(int(os.getenv("COMET_ALLOW_DIRTY", "0"))):
        raise RuntimeError(
            "Refusing to run on a dirty git tree. Set COMET_ALLOW_DIRTY=1 to override."
        )

    data_dir = get_data_dir()
    cmb = _resolve_map_path(
        paths.get("temperature_map"), "COM_CompMap_CMB-smica_2048_R1.20.fits", data_dir
    )
    lensing = _resolve_map_path(
        paths.get("lensing_map"), "COM_CompMap_Lensing_2048_R1.10.fits", data_dir
    )

    artifacts_dir = _resolve_artifacts_dir()
    nbins_env = os.getenv("COMET_NBINS", "10")
    try:
        nbins = max(int(nbins_env), 1)
    except ValueError:
        nbins = 10

    quick = _quick_products(cmb, lensing, artifacts_dir, nbins, check_only)

    payload: dict = {
        "timestamp_utc": now,
        "ordering": ordering,
        "config": {
            "prereg_path": str(prereg_path) if prereg_path else None,
            "paths_path": str(paths_path) if paths_path else None,
            "hash_sha256": cfg_hash,
        },
        "git": gi,
        "data": {
            "data_dir": str(data_dir),
            "lensing": {"present": lensing.exists(), "path": str(lensing)},
            "cmb": {"present": cmb.exists(), "path": str(cmb)},
        },
        "artifacts": {
            "dir": str(artifacts_dir),
            "files": list(quick.get("artifacts", [])),
        },
        "results": quick,
    }

    return payload
