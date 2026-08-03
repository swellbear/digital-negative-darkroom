"""Auto crop must analyze Frame picture well and keep aspect under straighten."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.auto_crop import remap_crop_box_to_parent, suggest_crop_box
from digital_negative.display import straighten_safe_crop_box


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_autocrop", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_remap_crop_box_preserves_local_aspect_into_parent():
    local = {
        "x": 0.1,
        "y": 0.2,
        "w": 0.6,
        "h": 0.4,
        "aspect": 1.5,
        "rule": "rule_of_thirds",
        "subject": {"x": 0.33, "y": 0.33},
    }
    parent = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
    out = remap_crop_box_to_parent(local, parent)
    assert abs(out["w"] / out["h"] - (0.6 * 0.8) / (0.4 * 0.8)) < 1e-6
    assert abs(out["x"] - (0.1 + 0.1 * 0.8)) < 1e-6
    assert abs(out["subject"]["x"] - (0.1 + 0.33 * 0.8)) < 1e-6


def test_auto_crop_uses_framing_picture_not_original():
    """Film-look Frame picture can disagree with the raw original — Auto crop
    must follow the picture the orange box is drawn on."""
    mod = _load_ui()
    h, w = 120, 180
    # Original: bright blob on the RIGHT.
    original = np.zeros((h, w, 3), dtype=np.uint8)
    original[:, 120:170] = 220
    # Geometry / DN: same.
    geom = (original.astype(np.float32) / 255.0).astype(np.float32)
    # Framing picture (film look): bright blob on the LEFT — different saliency.
    picture = np.zeros((h, w, 3), dtype=np.uint8)
    picture[:, 10:60] = 220

    class _DN:
        image = geom
        metadata = {"ui_state": {"locked_stages": []}, "ingest": {}}

        def to_luminance(self):
            return geom.mean(axis=-1)

        def touch(self):
            return None

    state = {
        "dn": _DN(),
        "original_base": original.copy(),
        "geometry_base": geom.copy(),
        "live_rgb": picture.copy(),
        "controls": {"border_frac": 0.0},
        "viewer_mode": "live",
        "development": None,
    }
    # Monkeypatch framing picture so the test does not need a full develop bake.
    state["_test_picture"] = picture

    real = mod._framing_picture_rgb

    def _fake_picture(s):
        return s.get("_test_picture")

    mod._framing_picture_rgb = _fake_picture
    try:
        rect, hint = mod.suggest_auto_crop("rule_of_thirds", "free", 0.0, state)
    finally:
        mod._framing_picture_rgb = real

    parts = [float(p) for p in rect.split(",")]
    # Crop should bias toward the LEFT blob in the framing picture.
    cx = parts[0] + 0.5 * parts[2]
    assert cx < 0.55, (rect, hint)
    # And must not match a crop driven only by the right-side original.
    box_orig = suggest_crop_box(original, rule="rule_of_thirds", aspect_ratio=None)
    cx_orig = box_orig["x"] + 0.5 * box_orig["w"]
    assert cx_orig > 0.45
    assert abs(cx - cx_orig) > 0.08


def test_auto_crop_keeps_aspect_with_straighten():
    mod = _load_ui()
    # Strong horizontal bands → stable subject; tilt via straighten slider.
    h, w = 160, 240
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bands = np.where((yy * 12).astype(int) % 2 == 0, 0.85, 0.2).astype(np.float32)
    level = np.broadcast_to(bands, (h, w)).copy()
    # Off-center bright block so crop is not full-frame.
    level[40:90, 30:90] = 1.0
    rgb = np.stack([level * 200, level * 200, level * 210], axis=-1).astype(np.uint8)

    class _DN:
        image = level.copy()
        metadata = {"ui_state": {"locked_stages": []}, "ingest": {}}

        def to_luminance(self):
            return level

        def touch(self):
            return None

    state = {
        "dn": _DN(),
        "original_base": rgb.copy(),
        "geometry_base": level.copy(),
        "live_rgb": rgb.copy(),
        "controls": {"border_frac": 0.0},
        "viewer_mode": "live",
        "development": None,
    }
    rect, hint = mod.suggest_auto_crop("rule_of_thirds", "3:2", 3.0, state)
    x, y, bw, bh = [float(p) for p in rect.split(",")]
    assert bh > 1e-6
    assert abs((bw / bh) - 1.5) < 0.08, (rect, hint)
    # Box must sit inside the straighten-safe window.
    safe = straighten_safe_crop_box(w, h, 3.0, aspect_ratio=None)
    assert x + 1e-4 >= safe["x"]
    assert y + 1e-4 >= safe["y"]
    assert x + bw <= safe["x"] + safe["w"] + 1e-4
    assert y + bh <= safe["y"] + safe["h"] + 1e-4


def test_bw_print_auto_crop_follows_live_look_not_raw_upload():
    """End-to-end: after a B&W live bake, Auto crop source == framing picture."""
    mod = _load_ui()
    fixture = ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"
    outs = mod.commit_ingest(None, str(fixture), None)
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)
    args = [
        "bw",
        "tri-x-400-v1",
        "d76",
        6.75,
        0.0,
        0.5,
        400,
        "none",
        0.01,
        0.0,
        "ilford-mgiv-rc-v1",
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
        0.1,
        0.0,
        0.0,
        0.0,
        21.0,
        1.0,
        0.0,
        True,
        {**state, "chemistry_mode": "bw"},
    ]
    st = mod.live_preview(*args, quality="high", mark_dirty=True)[-1]
    picture = mod._framing_picture_rgb(st)
    assert picture is not None
    # Patch suggest_crop_box call path by checking source via subject disagreement
    # already covered above; here assert API returns a valid overlay rect.
    rect, hint = mod.suggest_auto_crop("auto", "free", 0.0, st)
    parts = [float(p) for p in rect.split(",")]
    assert len(parts) == 4
    assert 0.05 <= parts[2] * parts[3] <= 1.0
    assert "Auto crop ready" in hint
    # Live is bordered; framing picture is not — Auto crop must not use live card size.
    assert np.asarray(st["live_rgb"]).shape[:2] != picture.shape[:2]
