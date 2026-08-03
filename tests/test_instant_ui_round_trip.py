"""Instant chemistry UI: stock round-trip, spot meter, live card draft."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"


def _load_ui():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui_instant", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _state(mod):
    outs = mod.commit_ingest(None, str(FIXTURE), None)
    return next(x for x in outs if isinstance(x, dict) and "roll" in x)


def _args(film, chem, mins, ei, state, *, grain=0.35):
    # Matches live_preview / on_film_change_and_preview signature (color-style).
    return [
        "instant",
        film,
        chem,
        mins,
        0.0,
        grain,
        ei,
        "none",
        0.01,
        0.0,
        "ra4-glossy-v1",  # unused for Instant; required by shared signature
        8.0,
        2.5,
        0.0,
        False,
        0.0,
        5.0,
        4.5,
        3.5,
        False,
        5,
        0.5,
        0.0,
        0.0,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        21.0,
        1.0,
        0.0,
        True,
        state,
    ]


def _arr(packed):
    live = packed[0]
    val = getattr(live, "value", None)
    if val is None and isinstance(live, dict):
        val = live.get("value")
    if val is None:
        val = live
    a = np.asarray(val, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return a


def test_instant_mode_stock_round_trip_and_spot():
    mod = _load_ui()
    state = _state(mod)
    state = {**state, "spot_pos": "0.5000,0.5000", "chemistry_mode": "instant"}

    out600 = mod.on_film_change_and_preview(
        *_args("polaroid-600-instant-v1", "pod", 3.0, 640, state, grain=0.35)
    )
    p600 = out600[3:]
    st600 = p600[-1]
    summary = st600.get("summary_cache") or ""
    assert "Polaroid 600" in summary or "600" in summary
    assert "Instant" in summary or "card" in summary.lower()
    assert getattr(st600.get("development"), "color_process", None) == "instant_integral"
    draft = st600.get("print_draft")
    assert draft is not None
    assert getattr(draft, "reflectance", None) is not None
    spot = p600[-2] or ""
    assert "Zone" in spot
    assert "wait" not in spot.lower()

    out_sx = mod.on_film_change_and_preview(
        *_args("polaroid-sx70-instant-v1", "pod", 4.0, 160, st600, grain=0.35)
    )
    p_sx = out_sx[3:]
    assert "SX-70" in (p_sx[-1].get("summary_cache") or "") or "sx" in (
        p_sx[-1].get("summary_cache") or ""
    ).lower()
    # Stocks must not be identical after a swap (identity retained).
    assert float(np.mean(np.abs(_arr(p600) - _arr(p_sx)))) > 0.005
    assert "Zone" in (p_sx[-2] or "")

    # Stale EI/minutes in inputs must not stick — atomic resolve like B&W/Color.
    out_back = mod.on_film_change_and_preview(
        *_args("polaroid-600-instant-v1", "pod", 4.0, 160, p_sx[-1], grain=0.35)
    )
    p_back = out_back[3:]
    assert float(np.mean(np.abs(_arr(p600) - _arr(p_back)))) < 1e-5
    assert p600[-2] == p_back[-2]


def test_instant_print_draft_matches_card_preview_shape():
    """Meter maps must align with the bordered card the user hovers."""
    sys.path.insert(0, str(ROOT / "src"))
    from digital_negative.curves import load_film_profile
    from digital_negative.development import develop
    from digital_negative.digital_negative import DigitalNegative, default_metadata

    img = np.clip(
        np.linspace(0.04, 1.2, 40 * 48, dtype=np.float32).reshape(40, 48), 0, None
    )
    rgb = np.stack([img, img * 0.9, img * 0.8], axis=-1)
    dn = DigitalNegative(image=rgb, metadata=default_metadata())
    profile = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    dn.metadata.setdefault("development", {})["process_temp_c"] = 21.0
    dn.metadata["development"]["card_border"] = True
    out = develop(dn, profile, development_minutes=3.0, developer_id="pod", commit=False)
    assert out.card_reflectance is not None
    assert out.card_density is not None
    assert out.card_reflectance.shape[:2] == out.positive_preview.shape[:2]
    assert out.card_density.shape[:2] == out.positive_preview.shape[:2]
