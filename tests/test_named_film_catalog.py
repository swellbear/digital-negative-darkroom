"""Named Kodak / Ilford / Fuji film catalog smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.ingest import ingest_path
from digital_negative.pipeline import list_film_profiles

ROOT = Path(__file__).resolve().parents[1]

NEW_BW = (
    "tmax-400-v1",
    "delta-400-v1",
    "acros-100-ii-v1",
)
NEW_COLOR = (
    "portra-160-spectral-v1",
    "portra-400-spectral-v1",
    "portra-800-spectral-v1",
    "ektachrome-100-spectral-v1",
    "provia-100f-spectral-v1",
    "velvia-50-spectral-v1",
)

EXPECTED_NAMES = {
    "tmax-400-v1": "Kodak T-Max 400",
    "delta-400-v1": "Ilford Delta 400",
    "acros-100-ii-v1": "Fujifilm Neopan Acros 100 II",
    "portra-160-spectral-v1": "Kodak Portra 160",
    "portra-400-spectral-v1": "Kodak Portra 400",
    "portra-800-spectral-v1": "Kodak Portra 800",
    "ektachrome-100-spectral-v1": "Kodak Ektachrome E100",
    "provia-100f-spectral-v1": "Fujifilm Provia 100F",
    "velvia-50-spectral-v1": "Fujifilm Velvia 50",
}


def test_named_batch_appears_in_correct_chemistry_lists():
    bw_ids = {p.stem for p in list_film_profiles(chemistry_mode="bw")}
    color_ids = {p.stem for p in list_film_profiles(chemistry_mode="color")}
    for fid in NEW_BW:
        assert fid in bw_ids
        assert fid not in color_ids
    for fid in NEW_COLOR:
        assert fid in color_ids
        assert fid not in bw_ids


def test_named_batch_profiles_load_with_source_attribution():
    for fid, expected_name in EXPECTED_NAMES.items():
        path = ROOT / "profiles" / "films" / f"{fid}.json"
        data = json.loads(path.read_text())
        assert data["id"] == fid
        assert data["name"] == expected_name
        assert data.get("manufacturer")
        source = data["source"]
        assert source.get("document")
        assert source.get("digitization_notes")
        profile = load_film_profile(path)
        assert profile.name == expected_name
        assert profile.iso == data["iso"]


def test_named_batch_develop_defaults_smoke():
    dn = ingest_path(None)
    means = {}
    for fid in (*NEW_BW, *NEW_COLOR):
        profile = load_film_profile(ROOT / "profiles" / "films" / f"{fid}.json")
        defaults = profile.raw["defaults"]
        result = develop(
            dn,
            profile,
            developer_id=defaults["developer_id"],
            development_minutes=defaults["development_minutes"],
            commit=False,
        )
        assert result.density is not None
        means[fid] = float(np.asarray(result.density).mean())
        if fid.startswith("portra") or fid.startswith("ektachrome") or fid.startswith("provia") or fid.startswith("velvia"):
            assert result.color_process in ("c41", "e6")
        else:
            assert result.color_process is None

    # Portra 160 / 400 / 800 should not collapse to the same mid density.
    assert abs(means["portra-160-spectral-v1"] - means["portra-800-spectral-v1"]) > 0.05
    # Velvia should read denser/higher contrast mid-scale than Provia on the same scene.
    assert means["velvia-50-spectral-v1"] > means["provia-100f-spectral-v1"]


def test_portra_family_grain_scales_ordered():
    g160 = json.loads((ROOT / "profiles/films/portra-160-spectral-v1.json").read_text())["grain_scale"]
    g400 = json.loads((ROOT / "profiles/films/portra-400-spectral-v1.json").read_text())["grain_scale"]
    g800 = json.loads((ROOT / "profiles/films/portra-800-spectral-v1.json").read_text())["grain_scale"]
    assert g160 < g400 < g800
