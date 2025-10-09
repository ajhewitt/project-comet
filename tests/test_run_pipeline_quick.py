import importlib
import importlib.util

import pytest

from comet.run import run_pipeline

_NUMPY_SPEC = importlib.util.find_spec("numpy")
if _NUMPY_SPEC is not None:
    np = importlib.import_module("numpy")
else:  # pragma: no cover - exercised when numpy is unavailable
    np = None


@pytest.mark.usefixtures("monkeypatch")
def test_run_pipeline_handles_missing_numpy(tmp_path, monkeypatch):
    if np is not None:
        pytest.skip("numpy is available; fallback path not exercised")

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("COMET_ARTIFACTS_DIR", artifacts_dir.as_posix())
    monkeypatch.setenv("COMET_ALLOW_DIRTY", "1")

    payload = run_pipeline()
    results = payload["results"]

    assert results["notes"] == "numpy_missing"
    assert results["nbins"] == 0
    assert results["artifacts"] == []
    assert payload["artifacts"]["files"] == []


@pytest.mark.skipif(np is None, reason="quick pipeline summary requires numpy")
@pytest.mark.usefixtures("monkeypatch")
def test_run_pipeline_produces_quick_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("COMET_DATA_DIR", data_dir.as_posix())
    monkeypatch.setenv("COMET_ARTIFACTS_DIR", artifacts_dir.as_posix())
    monkeypatch.setenv("COMET_NBINS", "6")
    monkeypatch.setenv("COMET_ALLOW_DIRTY", "1")

    payload = run_pipeline()
    results = payload["results"]

    assert results["nbins"] == 6
    assert results["notes"] in {"synthetic", "real-data"}

    expected = {"delta.npy", "cov_delta.npy", "z_scores.npy"}
    assert expected <= set(results["artifacts"])
    assert payload["artifacts"]["dir"] == artifacts_dir.as_posix()
    assert expected <= set(payload["artifacts"]["files"])

    for name in expected:
        arr_path = artifacts_dir / name
        assert arr_path.exists(), f"missing artifact {name}"
        arr = np.load(arr_path)
        assert arr.shape[0] == results["nbins"], name

    assert results["max_abs_z"] >= 0.0
    assert results["mean_abs_z"] >= 0.0
