from __future__ import annotations

import importlib


def test_core_libs_import_smoke():
    # Ensure compiled deps and core libs are importable in the env
    for mod in ("numpy", "astropy.io.fits", "healpy", "pymaster"):
        assert importlib.import_module(mod)
