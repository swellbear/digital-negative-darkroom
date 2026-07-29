"""Smoke tests for the darkroom pipeline."""

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
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import run_darkroom_pipeline, run_spike_pipeline
from digital_negative.print_engine import print_negative


def test_hp5_profile_loads_and_interpolates():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    assert profile.id == "hp5-plus-v1"
    dens = profile.density_from_log_exposure(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert dens[0] < dens[1] < dens[2]
    assert dens[0] > profile.base_plus_fog


def test_fp4_is_finer_grain_than_hp5():
    hp5 = json.loads((ROOT / "profiles" / "films" / "hp5-plus-v1.json").read_text())
    fp4 = json.loads((ROOT / "profiles" / "films" / "fp4-plus-v1.json").read_text())
    assert fp4["grain_scale"] < hp5["grain_scale"]
    assert fp4["iso"] < hp5["iso"]


def test_push_increases_midtone_density():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    normal = modify_curve(profile, relative_time=1.0)
    pushed = modify_curve(profile, relative_time=1.6)
    mid = np.array([2.3])
    assert float(pushed.density_from_log_exposure(mid)[0]) > float(
        normal.density_from_log_exposure(mid)[0]
    )


def test_high_energy_raises_contrast_vs_standard():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    std = modify_curve(profile, developer_id="standard")
    energy = modify_curve(profile, developer_id="high_energy")
    lo = np.array([1.6])
    hi = np.array([3.0])
    std_span = float(std.density_from_log_exposure(hi)[0] - std.density_from_log_exposure(lo)[0])
    energy_span = float(
        energy.density_from_log_exposure(hi)[0] - energy.density_from_log_exposure(lo)[0]
    )
    assert energy_span > std_span


def test_ingest_synthetic_and_develop():
    dn = ingest_path(None)
    assert dn.image.ndim == 3
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    result = develop(dn, profile, grain_strength=0.5)
    assert result.density.shape == dn.to_luminance().shape
    assert 0.05 < float(result.density.mean()) < 2.5


def test_log_exposure_mapping_places_midtones():
    scene = create_synthetic_scene(128, 96)
    luma = 0.2126 * scene[..., 0] + 0.7152 * scene[..., 1] + 0.0722 * scene[..., 2]
    log_e = linear_to_relative_log_exposure(luma, mid_log_e=2.2)
    med = float(np.median(log_e))
    assert 1.8 < med < 2.6


def test_print_stage_darkens_with_more_exposure():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "fp4-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, profile, grain_strength=0.0)
    soft = print_negative(developed.transmittance, dn, paper, overall_exposure=-0.7, grade=2.0)
    hard = print_negative(developed.transmittance, dn, paper, overall_exposure=0.7, grade=2.0)
    assert float(hard.preview.mean()) < float(soft.preview.mean())


def test_full_pipeline_writes_print_artifacts(tmp_path: Path):
    artifacts = run_darkroom_pipeline(
        output_dir=tmp_path,
        film_id="fp4-plus-v1",
        print_grade=3.0,
        grain_strength=0.8,
    )
    assert artifacts.comparison.exists()
    assert artifacts.print_preview is not None and artifacts.print_preview.exists()
    with artifacts.dn_json.open() as f:
        meta = json.load(f)
    assert meta["film_profile"]["id"] == "fp4-plus-v1"
    assert meta["print"]["enabled"] is True
    ops = {h.get("op") for h in meta["history"]}
    assert {"develop", "print"} <= ops


def test_spike_alias_skips_print(tmp_path: Path):
    artifacts = run_spike_pipeline(output_dir=tmp_path)
    assert artifacts.print_result is None
    assert artifacts.developed_preview.exists()
