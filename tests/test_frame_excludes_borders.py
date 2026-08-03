"""Frame crop/straighten must use the picture well — not Instant/easel borders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.print_engine import apply_border


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_frame_border", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_instant_framing_picture_matches_well_not_card():
    mod = _load_ui()
    img = np.linspace(0.05, 1.1, 48 * 64, dtype=np.float32).reshape(48, 64)
    rgb = np.stack([img, img * 0.92, img * 0.8], axis=-1)
    dn = DigitalNegative(image=rgb, metadata=default_metadata())
    profile = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    dn.metadata.setdefault("development", {})["card_border"] = True
    dn.metadata["development"]["process_temp_c"] = 21.0
    out = develop(dn, profile, development_minutes=3.0, developer_id="pod", commit=False)

    card = mod._to_rgb_u8(out.positive_preview)
    well = mod._to_rgb_u8(out.well_preview)
    assert card.shape[:2] != well.shape[:2]
    assert well.shape[:2] == dn.image.shape[:2]

    state = {
        "dn": dn,
        "development": out,
        "live_rgb": card,
        "controls": {"instant_border": True, "border_frac": 0.0},
        "viewer_mode": "live",
        "geometry_base": np.asarray(dn.image).copy(),
    }
    picture = mod._framing_picture_rgb(state)
    assert picture is not None
    assert picture.shape[:2] == well.shape[:2]
    # Must not be the taller/wider bordered card.
    assert picture.shape[0] < card.shape[0]
    assert picture.shape[1] <= card.shape[1]

    stage = mod._framing_stage_preview(state, 0.0)
    assert stage is not None
    assert stage.shape[:2] == picture.shape[:2]

    sh, sw = mod._stage_shape_for_straighten_crop(state)
    assert (sh, sw) == picture.shape[:2]


def test_easel_border_stripped_for_framing():
    mod = _load_ui()
    h, w = 80, 100
    picture = np.full((h, w, 3), 120, dtype=np.uint8)
    picture[20:60, 30:70] = 40
    bordered = (
        np.clip(apply_border(picture.astype(np.float32) / 255.0, 0.08) * 255.0, 0, 255)
    ).astype(np.uint8)
    assert bordered.shape[0] > h and bordered.shape[1] > w

    class _DN:
        metadata = {"ui_state": {"locked_stages": []}}

    state = {
        "dn": _DN(),
        "live_rgb": bordered,
        "controls": {"border_frac": 0.08},
        "viewer_mode": "live",
    }
    framing = mod._framing_picture_rgb(state)
    assert framing is not None
    assert framing.shape[:2] == (h, w)


def test_entering_frame_tool_shows_picture_well():
    mod = _load_ui()
    img = np.linspace(0.05, 1.0, 40 * 56, dtype=np.float32).reshape(40, 56)
    rgb = np.stack([img, img, img * 0.85], axis=-1)
    dn = DigitalNegative(image=rgb, metadata=default_metadata())
    profile = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    dn.metadata.setdefault("development", {})["card_border"] = True
    out = develop(dn, profile, development_minutes=3.0, developer_id="pod", commit=False)
    card = mod._to_rgb_u8(out.positive_preview)
    state = {
        "dn": dn,
        "development": out,
        "live_rgb": card,
        "controls": {"instant_border": True},
        "viewer_mode": "live",
        "geometry_base": np.asarray(dn.image).copy(),
    }
    live_u, _acc = mod.on_preview_tool_change("frame", state)
    payload = getattr(live_u, "__dict__", None) or {}
    value = payload.get("value")
    if value is None and isinstance(live_u, dict):
        value = live_u.get("value")
    assert value is not None
    assert np.asarray(value).shape[:2] == out.well_preview.shape[:2]
