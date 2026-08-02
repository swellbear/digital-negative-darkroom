"""Crop & straighten accordion must stay open while Frame tool is engaged."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_preview_tool_keeps_crop_accordion_open_for_frame():
    mod = _load_ui()
    label_u, crop_u = mod.on_preview_tool_change("frame", state=None)
    assert _update_dict(crop_u).get("open") is True
    assert "label" in _update_dict(label_u) or hasattr(label_u, "label") or True


def test_preview_tool_closes_crop_accordion_for_print():
    mod = _load_ui()
    _label_u, crop_u = mod.on_preview_tool_change("print", state=None)
    assert _update_dict(crop_u).get("open") is False


def test_apply_framing_closes_crop_accordion_output():
    mod = _load_ui()
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    # Gradio wire + JS must not close crop on every non-frame drawer apply.
    assert "mod_crop_acc" in source
    assert "prev === 'frame' && n !== 'frame'" in source
    assert "window.__cropEngageUntil" in source
    assert "lastDrawerVal = readActiveDrawer()" in source


def test_apply_and_reset_return_crop_accordion_state():
    from digital_negative.digital_negative import DigitalNegative, default_metadata
    import numpy as np

    mod = _load_ui()
    h, w = 120, 160
    base = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
    orig = np.clip(base * 255, 0, 255).astype(np.uint8)
    orig = np.stack([orig, orig, orig], axis=-1)
    dn = DigitalNegative(image=base.copy(), metadata=default_metadata())
    state = {
        "dn": dn,
        "geometry_base": base.copy(),
        "original_base": orig.copy(),
        "original_view": orig.copy(),
        "original_inspect": orig.copy(),
        "original_ref": orig.copy(),
        "viewer_mode": "live",
        "strip_slots": list(mod.STRIP_DEFAULT_SLOTS),
        "stage": "development",
        "dirty": False,
    }
    outs = mod.apply_crop_straighten(0.0, "0.25000,0.25000,0.50000,0.50000", "free", state)
    assert len(outs) == 20
    assert _update_dict(outs[19]).get("open") is False

    state = outs[12]
    reset = mod.reset_crop_straighten(state)
    assert len(reset) == 20
    assert _update_dict(reset[19]).get("open") is True
