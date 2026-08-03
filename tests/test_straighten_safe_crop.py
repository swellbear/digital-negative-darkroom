"""Straighten must shrink the crop box so fill wedges are outside it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.display import (
    apply_framing,
    straighten_image,
    straighten_safe_crop_box,
)


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _update_dict(update):
    if isinstance(update, dict):
        return update
    raw = getattr(update, "__dict__", None) or {}
    if "value" in raw:
        return raw
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return {}


def test_safe_crop_excludes_fill_on_tall_instant_frame():
    # Matches the Instant card shape from the user report (~867×1542, ~11.5°).
    h, w = 1542, 867
    deg = 11.5
    img = np.ones((h, w), dtype=np.float32)
    box = straighten_safe_crop_box(w, h, deg, pad=0.02)
    assert box["w"] < 0.95 and box["h"] < 0.95
    assert abs(box["x"] + box["w"] / 2 - 0.5) < 1e-3
    assert abs(box["y"] + box["h"] / 2 - 0.5) < 1e-3

    left, top = box["x"], box["y"]
    right = 1.0 - (box["x"] + box["w"])
    bottom = 1.0 - (box["y"] + box["h"])
    framed = apply_framing(
        img,
        straighten_degrees_cw=deg,
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        fill=0.0,
    )
    assert float(framed.min()) > 0.99
    assert float((framed < 0.5).mean()) == 0.0


def test_safe_crop_zero_angle_is_full_frame():
    box = straighten_safe_crop_box(800, 600, 0.0)
    assert box == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def test_auto_straighten_returns_trimmed_crop_rect():
    mod = _load_ui()
    from digital_negative.digital_negative import DigitalNegative, default_metadata

    h, w = 200, 280
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bands = np.where((yy * 14).astype(int) % 2 == 0, 0.88, 0.18).astype(np.float32)
    level = np.broadcast_to(bands, (h, w)).copy()
    tilted = straighten_image(level, -2.0)
    live = np.stack(
        [np.clip(tilted * 170, 0, 255), np.clip(tilted * 190, 0, 255), np.clip(tilted * 210, 0, 255)],
        axis=-1,
    ).astype(np.uint8)
    dn = DigitalNegative(image=tilted.copy(), metadata=default_metadata())
    state = {
        "dn": dn,
        "geometry_base": tilted.copy(),
        "live_rgb": live.copy(),
        "viewer_mode": "live",
        "strip_slots": list(mod.STRIP_DEFAULT_SLOTS),
        "stage": "development",
        "dirty": False,
    }
    deg, hint, live_u, rect = mod.suggest_auto_straighten(state, "free")
    assert abs(float(deg) - 2.0) <= 0.75
    assert "black" in str(hint).lower() or "trimmed" in str(hint).lower()
    parts = [float(p) for p in str(rect).split(",")]
    assert len(parts) == 4
    x, y, bw, bh = parts
    assert bw < 1.0 - 1e-6 or bh < 1.0 - 1e-6
    assert x > 0.0 and y > 0.0
    preview = _update_dict(live_u).get("value")
    assert preview is not None
