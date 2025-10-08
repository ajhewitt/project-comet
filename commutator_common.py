from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_impl = importlib.import_module("_commutator_common_impl")

__all__ = getattr(_impl, "__all__", tuple())
__doc__ = getattr(_impl, "__doc__", None)

globals().update({name: getattr(_impl, name) for name in __all__})
