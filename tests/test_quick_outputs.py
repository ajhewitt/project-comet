import os
import subprocess
from pathlib import Path

import numpy as np


def test_quick_outputs(tmp_path):
    env = os.environ.copy()
    env["COMET_NSIDE"] = "256"
    env["COMET_NSIMS"] = "5"
    subprocess.run(
        ["bash", "scripts/pipeline_quick.sh", str(tmp_path / "data")], check=True, env=env
    )
    for fname in ["delta.npy", "cov_delta.npy", "z_scores.npy"]:
        p = Path("artifacts") / fname
        assert p.exists(), f"missing {fname}"
        arr = np.load(p)
        assert np.isfinite(arr).all()
