"""Auto straighten must rotate the live print — not swap in the original photo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
    return dict(update) if hasattr(update, "keys") else {}


def _state_with_distinct_live_and_original(mod):
    from digital_negative.digital_negative import DigitalNegative, default_metadata
    from digital_negative.display import straighten_image

    h, w = 180, 240
    # Strong horizontal bands → clear tilt signal when rotated.
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bands = np.where((yy * 16).astype(int) % 2 == 0, 0.9, 0.15).astype(np.float32)
    level = np.broadcast_to(bands, (h, w)).copy()
    tilted = straighten_image(level, -2.5)

    # Live print look (cool gray) vs a warm "original photo" that must not appear.
    live = np.stack(
        [
            np.clip(tilted * 180, 0, 255),
            np.clip(tilted * 200, 0, 255),
            np.clip(tilted * 220, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    original = np.zeros((h, w, 3), dtype=np.uint8)
    original[..., 0] = 240
    original[..., 1] = 40
    original[..., 2] = 40

    dn = DigitalNegative(image=tilted.copy(), metadata=default_metadata())
    return {
        "dn": dn,
        "geometry_base": tilted.copy(),
        "original_base": original.copy(),
        "live_rgb": live.copy(),
        "viewer_mode": "live",
        "strip_slots": list(mod.STRIP_DEFAULT_SLOTS),
        "stage": "development",
        "dirty": False,
    }


def test_framing_preview_uses_live_not_original():
    mod = _load_ui()
    state = _state_with_distinct_live_and_original(mod)
    preview = mod._framing_stage_preview(state, 0.0)
    assert preview is not None
    # Original is pure red-ish; live is cool gray bands. Mean blue channel
    # on live is high; on original it is low.
    assert float(preview[..., 2].mean()) > 100
    assert float(preview[..., 0].mean()) < 200


def test_auto_straighten_rotates_live_and_sets_nonzero_angle():
    mod = _load_ui()
    state = _state_with_distinct_live_and_original(mod)
    deg, hint, live_u, rect = mod.suggest_auto_straighten(state, "free")
    assert abs(float(deg) - 2.5) <= 0.75
    assert "Auto straighten" in str(hint)
    parts = [float(p) for p in str(rect).split(",")]
    assert len(parts) == 4 and parts[2] * parts[3] < 0.999
    payload = _update_dict(live_u)
    preview = payload.get("value")
    assert preview is not None
    # Still the live look (not the red original).
    assert float(np.asarray(preview)[..., 2].mean()) > 100
    # Rotation moved content — differs from the unstraightened live frame.
    before = np.asarray(state["live_rgb"], dtype=np.float32)
    after = np.asarray(preview, dtype=np.float32)
    assert float(np.mean(np.abs(after - before))) > 1.0
