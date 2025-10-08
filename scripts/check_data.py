#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

from comet.io_maps import MapData, MapMetadata, read_fits_map_with_meta

EXPECTED_UNITS = {
    "temperature_map": {"K_CMB"},
    "lensing_map": {"", "1", "DIMENSIONLESS", "UNITLESS"},
}


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_map_path(raw: str, data_dir: Path | None, repo_root: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if data_dir is not None:
        relative = candidate
        if relative.parts and relative.parts[0] == data_dir.name:
            relative = Path(*relative.parts[1:]) if len(relative.parts) > 1 else Path()
        resolved = (data_dir / relative).resolve()
        return resolved
    return (repo_root / candidate).resolve()


def _load_paths_config(path: Path) -> dict[str, str]:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected mapping in {path}, got {type(data)!r}"
        raise TypeError(msg)
    return {str(key): str(value) for key, value in data.items()}


def _expected_units_for(name: str) -> set[str]:
    return EXPECTED_UNITS.get(name, set())


def _validate_units(name: str, metadata: MapMetadata) -> None:
    unit = metadata.unit
    allowed = {value.upper() for value in _expected_units_for(name)}
    if allowed and unit.upper() not in allowed:
        unit_display = unit or "<none>"
        msg = f"{name}: expected units {sorted(allowed)}, found '{unit_display}'"
        raise ValueError(msg)


def _validate_geometry(name: str, metadata: MapMetadata) -> None:
    if metadata.ordering and metadata.ordering.upper() != "RING":
        msg = f"{name}: expected RING ordering, found {metadata.ordering!r}"
        raise ValueError(msg)
    if metadata.nside <= 0:
        msg = f"{name}: invalid NSIDE {metadata.nside}"
        raise ValueError(msg)


def _metadata_output_path(out_dir: Path, map_path: Path) -> Path:
    return out_dir / f"{map_path.stem}.metadata.json"


def _process_map(name: str, path: Path, out_dir: Path) -> MapData:
    map_data = read_fits_map_with_meta(path)
    _validate_units(name, map_data.metadata)
    _validate_geometry(name, map_data.metadata)
    meta_path = _metadata_output_path(out_dir, path)
    map_data.write_metadata_json(meta_path)
    return map_data


def _write_summary(out: Path, entries: Iterable[MapData]) -> None:
    records = []
    for entry in entries:
        payload = entry.metadata.to_json_dict()
        payload["npix"] = int(entry.pixels.size)
        records.append(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"maps": records}, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate Planck FITS metadata")
    repo_root = _resolve_repo_root()
    ap.add_argument("--paths", default=repo_root / "config" / "paths.yaml", type=Path)
    ap.add_argument("--data-dir", default=None, type=Path)
    ap.add_argument("--metadata-dir", default=repo_root / "artifacts" / "metadata", type=Path)
    ap.add_argument(
        "--summary", default=repo_root / "artifacts" / "metadata" / "summary.json", type=Path
    )
    args = ap.parse_args(argv)

    paths_cfg = _load_paths_config(args.paths)
    entries: list[MapData] = []
    errors: list[str] = []
    for name, raw_path in paths_cfg.items():
        resolved = _resolve_map_path(raw_path, args.data_dir, repo_root)
        try:
            data = _process_map(name, resolved, args.metadata_dir)
        except Exception as exc:  # noqa: BLE001 - surface validation error
            errors.append(f"{name}: {exc}")
            continue
        entries.append(data)

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    _write_summary(args.summary, entries)
    print(json.dumps({"validated": [entry.metadata.path.as_posix() for entry in entries]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
