"""Simulation helpers for null covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from commutator_common import infer_bin_lmax, nm_bandpowers, nm_field_from_scalar

from .theory import TheoryCls

try:  # pragma: no cover - optional dependency in CI
    import healpy as _hp
except ModuleNotFoundError as exc:  # pragma: no cover
    _hp = None
    _HP_ERROR = exc
else:  # pragma: no cover
    _HP_ERROR = None


@dataclass(slots=True)
class SimulationGeometry:
    mask: np.ndarray
    bins: object  # NaMaster bins; stored as object to avoid heavy typing dependency
    nside: int
    lmax: int
    field_lmax: int


def _positive_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    return candidate


def resolve_simulation_bandlimits(
    bins: object,
    *,
    requested_lmax: int | None,
    theory_lmax: int | None,
    nside: int,
) -> tuple[int, int]:
    """Determine consistent simulation and field band-limits for NaMaster runs."""

    if nside <= 0:
        raise ValueError("nside must be positive")

    default_lmax = 3 * nside - 1
    bin_lmax = infer_bin_lmax(
        bins,
        fallbacks=(requested_lmax, default_lmax, theory_lmax),
    )

    candidates = [
        requested_lmax,
        theory_lmax,
        bin_lmax,
        default_lmax,
    ]
    positive_candidates = [_positive_int(value) for value in candidates]
    filtered = [value for value in positive_candidates if value is not None]
    if not filtered:
        raise ValueError("Unable to determine a positive simulation band-limit")

    sim_lmax = min(filtered)
    bin_limit = _positive_int(bin_lmax)
    field_lmax = min(sim_lmax, bin_limit) if bin_limit is not None else sim_lmax
    return sim_lmax, field_lmax


def _require_healpy():  # pragma: no cover - small wrapper
    module = globals().get("_hp")
    if module is None:
        error = globals().get("_HP_ERROR")
        raise ModuleNotFoundError(
            "healpy is required for simulation helpers. Install it from conda-forge as 'healpy'."
        ) from error
    return module


def draw_correlated_maps(
    theory: TheoryCls,
    *,
    nside: int,
    lmax: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw correlated temperature and lensing maps from theory spectra."""

    if lmax <= 0:
        raise ValueError("lmax must be positive")
    if nside <= 0:
        raise ValueError("nside must be positive")

    hp = _require_healpy()
    truncated = theory.truncate(lmax)
    cls = truncated.as_synalm_array(lmax=lmax)
    seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    legacy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        alms = hp.synalm(cls, lmax=lmax, new=True)
    finally:
        np.random.set_state(legacy_state)

    t_map = hp.alm2map(alms[0], nside=nside, lmax=lmax)
    k_map = hp.alm2map(alms[1], nside=nside, lmax=lmax)
    return np.asarray(t_map, dtype=float), np.asarray(k_map, dtype=float)


def estimate_delta_covariance(
    theory: TheoryCls,
    geometry: SimulationGeometry,
    *,
    nsims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate the covariance of Δ bandpowers using simulations."""

    if nsims <= 1:
        raise ValueError("nsims must be greater than 1 for covariance estimation")

    if geometry.lmax <= 0:
        raise ValueError("geometry.lmax must be positive")
    if geometry.field_lmax <= 0:
        raise ValueError("geometry.field_lmax must be positive")

    field_lmax = infer_bin_lmax(
        geometry.bins,
        fallbacks=(geometry.field_lmax, geometry.lmax, 3 * geometry.nside - 1),
    )
    resolved_field_lmax = _positive_int(field_lmax)
    if resolved_field_lmax is None:
        raise ValueError("Unable to determine a positive field band-limit")
    field_lmax = min(resolved_field_lmax, geometry.field_lmax, geometry.lmax)

    mask = np.asarray(geometry.mask, dtype=float)
    if mask.ndim != 1:
        raise ValueError("Mask must be a 1-D HEALPix vector")
    expected_size = 12 * geometry.nside**2
    if mask.size != expected_size:
        raise ValueError(f"Mask size {mask.size} does not match nside {geometry.nside}")

    deltas: list[np.ndarray] = []
    for _ in range(nsims):
        t_map, k_map = draw_correlated_maps(
            theory, nside=geometry.nside, lmax=geometry.lmax, rng=rng
        )
        f_t = nm_field_from_scalar(t_map, mask, lmax=field_lmax)
        f_k = nm_field_from_scalar(k_map, mask, lmax=field_lmax)
        cl_tk = nm_bandpowers(f_t, f_k, geometry.bins)
        cl_kt = nm_bandpowers(f_k, f_t, geometry.bins)
        deltas.append(cl_tk - cl_kt)

    delta_stack = np.vstack(deltas)
    cov = np.cov(delta_stack, rowvar=False)
    trace = float(np.trace(cov))
    eps = 1e-12 if trace == 0 else 1e-10 * (trace / max(cov.shape[0], 1))
    return cov + eps * np.eye(cov.shape[0])


__all__ = [
    "SimulationGeometry",
    "draw_correlated_maps",
    "estimate_delta_covariance",
    "resolve_simulation_bandlimits",
]
