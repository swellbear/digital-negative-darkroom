"""Smoke tests for the technical spike pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile, modify_curve
from digital_negative.development import develop, linear_to_relative_log_exposure
from digital_negative.ingest import create_synthetic_scene, ingest_path
from digital_negative.pipeline import run_spike_pipeline


def test_hp5_profile_loads_and_interpolates():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    assert profile.id == "hp5-plus-v1"
    dens = profile.density_from_log_exposure(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert dens[0] < dens[1] < dens[2]
    assert dens[0] > profile.base_plus_fog


def test_push_increases_midtone_density():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    normal = modify_curve(profile, relative_time=1.0)
    pushed = modify_curve(profile, relative_time=1.6)
    mid = np.array([2.3])
    assert float(pushed.density_from_log_exposure(mid)[0]) > float(
        normal.density_from_log_exposure(mid)[0]
    )


def test_ingest_synthetic_and_develop():
    dn = ingest_path(None)
    assert dn.image.ndim == 3
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    result = develop(dn, profile)
    assert result.density.shape == dn.to_luminance().shape
    assert 0.05 < float(result.density.mean()) < 2.5


def test_log_exposure_mapping_places_midtones():
    scene = create_synthetic_scene(128, 96)
    luma = 0.2126 * scene[..., 0] + 0.7152 * scene[..., 1] + 0.0722 * scene[..., 2]
    log_e = linear_to_relative_log_exposure(luma, mid_log_e=2.2)
    med = float(np.median(log_e))
    assert 1.8 < med < 2.6


def test_full_spike_writes_artifacts(tmp_path: Path):
    artifacts = run_spike_pipeline(output_dir=tmp_path)
    assert artifacts.comparison.exists()
    assert artifacts.dn_tiff.exists()
    assert artifacts.dn_json.exists()
    with artifacts.dn_json.open() as f:
        meta = json.load(f)
    assert meta["film_profile"]["id"] == "hp5-plus-v1"
    assert "develop" in {h.get("op") for h in meta["history"]}
