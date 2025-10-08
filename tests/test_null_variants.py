from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="null variant tests require numpy")

import numpy as np

from comet.nulls import (
    curl_null_field,
    evaluate_null_tests,
    hemisphere_jackknife,
    rotation_null,
)


def test_rotation_null_zero_for_symmetric_map():
    arr = np.ones(8)
    residual = rotation_null(arr)
    assert residual.shape == arr.shape
    assert np.allclose(residual, 0.0)


def test_hemisphere_jackknife_sign_flip_symmetry():
    arr = np.linspace(-1.0, 1.0, 8)
    residual = hemisphere_jackknife(arr)
    assert residual.shape == arr.shape
    assert np.isclose(residual.sum(), 0.0)
    # North/south halves should have opposite means
    half = residual.size // 2
    assert np.isclose(residual[:half].mean(), -residual[half:].mean())


def test_curl_null_field_annihilates_constant_modes():
    arr = np.full(6, 3.14)
    residual = curl_null_field(arr)
    assert residual.shape == arr.shape
    assert np.allclose(residual, 0.0)


def test_evaluate_null_tests_reports_small_z_for_synthetic_skies():
    delta = np.zeros(10, dtype=float)
    cov = np.eye(10, dtype=float)

    results = evaluate_null_tests(delta, cov)
    assert set(results.keys()) == {"rotation_90", "hemisphere_jackknife", "curl_null"}

    for result in results.values():
        assert result.residual.shape == delta.shape
        assert np.allclose(result.residual, 0.0)
        assert result.z == pytest.approx(0.0, abs=1e-6)
