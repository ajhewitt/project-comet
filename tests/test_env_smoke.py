import importlib


def test_core_libs_import_smoke():
    # Make sure the environment is sane
    for mod in ("numpy", "astropy.io.fits", "healpy", "pymaster"):
        assert importlib.import_module(mod)
