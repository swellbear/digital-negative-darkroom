"""End-to-end framing smoke: Auto straighten → Apply → Reset on Live print."""

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
    if "value" in raw or "open" in raw:
        return raw
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return dict(update) if hasattr(update, "keys") else {}


def _tilted_live_state(mod):
    from digital_negative.digital_negative import DigitalNegative, default_metadata
    from digital_negative.display import straighten_image

    h, w = 200, 280
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bands = np.where((yy * 14).astype(int) % 2 == 0, 0.88, 0.18).astype(np.float32)
    level = np.broadcast_to(bands, (h, w)).copy()
    tilted = straighten_image(level, -2.0)

    live = np.stack(
        [
            np.clip(tilted * 170, 0, 255),
            np.clip(tilted * 190, 0, 255),
            np.clip(tilted * 210, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    # Distinct original — Auto must not swap this in.
    original = np.zeros((h, w, 3), dtype=np.uint8)
    original[..., 0] = 230
    original[..., 1] = 30
    original[..., 2] = 30

    dn = DigitalNegative(image=tilted.copy(), metadata=default_metadata())
    return {
        "dn": dn,
        "geometry_base": tilted.copy(),
        "original_base": original.copy(),
        "original_view": original.copy(),
        "original_inspect": original.copy(),
        "original_ref": original.copy(),
        "live_rgb": live.copy(),
        "viewer_mode": "live",
        "strip_slots": list(mod.STRIP_DEFAULT_SLOTS),
        "stage": "development",
        "dirty": False,
    }


def test_framing_smoke_auto_apply_reset():
    mod = _load_ui()
    state = _tilted_live_state(mod)

    # 1) Crop preview stays on Live (cool gray), not red original.
    preview0 = mod._framing_stage_preview(state, 0.0)
    assert preview0 is not None
    assert float(np.asarray(preview0)[..., 2].mean()) > 100

    # 2) Auto straighten finds tilt and rotates Live in place.
    deg, hint, live_u = mod.suggest_auto_straighten(state)
    assert abs(float(deg) - 2.0) <= 0.75
    assert "Auto straighten" in str(hint)
    preview = _update_dict(live_u).get("value")
    assert preview is not None
    assert float(np.asarray(preview)[..., 2].mean()) > 100
    assert float(np.mean(np.abs(np.asarray(preview, dtype=np.float32) - state["live_rgb"]))) > 1.0

    # 3) Apply framing bakes crop, exits Frame, closes crop accordion.
    outs = mod.apply_crop_straighten(
        float(deg), "0.20000,0.20000,0.60000,0.60000", "free", state
    )
    assert len(outs) == 20
    assert _update_dict(outs[16]).get("value") == "print"
    assert outs[17] == "develop"
    assert _update_dict(outs[19]).get("open") is False
    state = outs[12]
    assert state["dn"].metadata["ingest"]["crop"]["applied"] is True

    # 4) Reset restores framing UI / re-opens crop.
    reset = mod.reset_crop_straighten(state)
    assert len(reset) == 20
    assert _update_dict(reset[16]).get("value") == "frame"
    assert reset[17] == "frame"
    assert _update_dict(reset[19]).get("open") is True


def test_framing_js_guards_still_present():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "window.__cropEngageUntil" in source
    assert "prev === 'frame' && n !== 'frame'" in source
    assert "mod_crop_acc" in source
    assert "_display_live_rgb(state)" in source
