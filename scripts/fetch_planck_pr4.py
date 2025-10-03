#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILES = [
    "COM_CompMap_Lensing_2048_R1.10.fits",
    "COM_CompMap_CMB-smica_2048_R1.20.fits",
]


def main():
    ap = argparse.ArgumentParser(description="Validate presence of Planck PR4 maps.")
    ap.add_argument("--out", default="data", help="Directory where maps should live.")
    args = ap.parse_args()

    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)

    status = {}
    for f in FILES:
        p = d / f
        status[f] = p.exists()

    print(json.dumps({"data_dir": str(d), "files": status}))
    missing = [k for k, v in status.items() if not v]
    if missing:
        print(json.dumps({"error": f"missing files: {missing}"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
