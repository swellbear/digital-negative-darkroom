"""Capture realism: EI, filters, reciprocity, halation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.capture import (
    apply_halation,
    ei_log_shift,
    filtered_luminance,
    reciprocity_factors,
)
from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
def test_ei_shift_direction():
    # Rating faster than box underexposes → negative log-E shift.
    assert ei_log_shift(400, 1600) < 0
    assert abs(ei_log_shift(400, 400)) < 1e-9


def test_reciprocity_kicks_in_for_long_exposures():
    shift, contrast = reciprocity_factors(30.0)
    assert shift < 0
    assert contrast > 0
    assert reciprocity_factors(0.01) == (0.0, 0.0)


def test_halation_adds_density_near_highlights():
    d = np.full((64, 64), 0.4, dtype=np.float32)
    d[28:36, 28:36] = 1.8
    out = apply_halation(d, strength=1.0, fog=0.1)
    # Neighbours of the hot square pick up bloom.
    assert float(out[20, 32]) > float(d[20, 32])


def test_filter_changes_xyz_luminance():
    xyz = np.zeros((8, 8, 3), dtype=np.float32)
    xyz[..., 0] = 0.4  # X
    xyz[..., 1] = 0.2  # Y
    xyz[..., 2] = 0.8  # Z (blue-heavy)
    meta = default_metadata()
    meta["ingest"]["working_space"] = "CIE_XYZ"
    meta["ingest"]["luminance_channel"] = "Y"
    dn = DigitalNegative(image=xyz, metadata=meta)
    y = filtered_luminance(dn, "none")
    red = filtered_luminance(dn, "red")
    # Red filter weights X more / Z less → brighter than plain Y on this patch.
    assert float(red.mean()) > float(y.mean())


def test_develop_ei_moves_density():
    from digital_negative.ingest import ingest_path

    scene = ingest_path(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png")
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    box = develop(
        scene, profile, developer_id="id11_stock", development_minutes=6.5,
        exposure_index=400, commit=False,
    )
    push = develop(
        scene, profile, developer_id="id11_stock", development_minutes=6.5,
        exposure_index=1600, commit=False,
    )
    # Underexposure → less density overall at same development.
    assert float(push.density.mean()) < float(box.density.mean())
