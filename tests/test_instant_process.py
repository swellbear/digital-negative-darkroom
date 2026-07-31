"""Integral Polaroid / Instant film process smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.instant_process import auto_process_minutes, process_instant
from digital_negative.pipeline import list_film_profiles
from digital_negative.spectral import chemistry_mode_for_film_type, is_instant_film_type

INSTANT_IDS = (
    "polaroid-600-instant-v1",
    "polaroid-sx70-instant-v1",
    "polaroid-itype-instant-v1",
    "polaroid-bw600-instant-v1",
)


def _scene():
    img = np.clip(np.linspace(0.04, 1.3, 48 * 64, dtype=np.float32).reshape(48, 64), 0, None)
    rgb = np.stack([img, img * 0.92, img * 0.8], axis=-1)
    return DigitalNegative(image=rgb, metadata=default_metadata())


def test_instant_catalog_split():
    instant = {p.stem for p in list_film_profiles(chemistry_mode="instant")}
    bw = {p.stem for p in list_film_profiles(chemistry_mode="bw")}
    color = {p.stem for p in list_film_profiles(chemistry_mode="color")}
    for fid in INSTANT_IDS:
        assert fid in instant
        assert fid not in bw
        assert fid not in color


def test_instant_film_type_helpers():
    assert chemistry_mode_for_film_type("instant_integral_color") == "instant"
    assert chemistry_mode_for_film_type("instant_integral_bw") == "instant"
    assert is_instant_film_type("instant_integral_color")
    assert not is_instant_film_type("color_negative")


def test_process_instant_returns_bordered_card():
    dn = _scene()
    profile = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    result = process_instant(dn, profile, process_temp_c=21.0, commit=False)
    assert result.card_rgb.ndim == 3 and result.card_rgb.shape[2] == 3
    assert result.card_rgb.shape[0] > dn.image.shape[0]  # border grows the frame
    assert result.preview.max() > 0.2
    assert result.process == "instant_integral"


def test_cold_process_auto_minutes_longer():
    profile = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    warm = auto_process_minutes(profile, 21.0)
    cold = auto_process_minutes(profile, 13.0)
    assert cold > warm


def test_develop_routes_instant_films():
    dn = _scene()
    for fid in INSTANT_IDS:
        profile = load_film_profile(ROOT / "profiles" / "films" / f"{fid}.json")
        dn.metadata.setdefault("development", {})["process_temp_c"] = 21.0
        out = develop(
            dn,
            profile,
            development_minutes=float(profile.defaults["development_minutes"]),
            developer_id="pod",
            commit=False,
        )
        assert out.color_process == "instant_integral"
        assert out.positive_preview.ndim == 3
