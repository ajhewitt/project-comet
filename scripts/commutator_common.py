from __future__ import annotations

"""Compatibility wrapper for the shared commutator utilities.

The canonical implementation now lives in ``commutator_common`` (installed from the
``src`` tree).  This stub keeps existing script entry-points working when executed
directly via ``python scripts/...`` by deferring to the shared module at import time.
"""

# ruff: noqa: E402

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import _commutator_common_impl as _impl

__all__ = _impl.__all__

globals().update({name: getattr(_impl, name) for name in __all__})
