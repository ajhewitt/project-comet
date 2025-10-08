from __future__ import annotations

import importlib
import importlib.util

import pytest

_REQUIRED = ("numpy", "astropy.io.fits", "healpy", "pymaster")


def _is_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


_MISSING = [name for name in _REQUIRED if _is_missing(name)]

pytestmark = pytest.mark.skipif(
    _MISSING,
    reason=f"missing optional compiled dependencies: {', '.join(_MISSING)}",
)


def test_core_libs_import_smoke():
    # Ensure compiled deps and core libs are importable in the env when present
    for mod in _REQUIRED:
        assert importlib.import_module(mod)
