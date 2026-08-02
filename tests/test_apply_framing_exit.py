"""Apply framing must bake the crop, exit Frame tool, and match Gradio outputs."""

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


def _minimal_state(mod, h=120, w=160):
    from digital_negative.digital_negative import DigitalNegative, default_metadata

    base = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
    orig = np.clip(base * 255, 0, 255).astype(np.uint8)
    orig = np.stack([orig, orig, orig], axis=-1)
    dn = DigitalNegative(image=base.copy(), metadata=default_metadata())
    return {
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


def test_apply_framing_bakes_crop_and_exits_frame_mode():
    mod = _load_ui()
    state = _minimal_state(mod)
    h0, w0 = state["geometry_base"].shape[:2]

    # Center 50% box → left/top/right/bottom trims of 0.25 each.
    outs = mod.apply_crop_straighten(0.0, "0.25000,0.25000,0.50000,0.50000", "free", state)
    assert len(outs) == 19

    new_state = outs[12]
    assert new_state["geometry_base"].shape[0] < h0
    assert new_state["geometry_base"].shape[1] < w0
    assert new_state["dn"].image.shape[:2] == new_state["geometry_base"].shape[:2]
    assert new_state["original_base"].shape[0] < h0
    assert new_state["dn"].metadata["ingest"]["crop"]["applied"] is True

    # Crop UI resets; preview leaves Frame mode.
    assert outs[13] == 0.0
    assert outs[14] == mod.DEFAULT_CROP_RECT
    assert outs[15] == "free"
    preview_tool = _update_dict(outs[16])
    assert preview_tool.get("value") == "print"
    assert outs[17] == "develop"
    assert "Framing applied" in str(outs[18])


def test_reset_framing_return_arity_matches_apply():
    mod = _load_ui()
    state = _minimal_state(mod)
    # Bake once so reset has a framed baseline to restore.
    framed = mod.apply_crop_straighten(0.0, "0.20000,0.20000,0.60000,0.60000", "free", state)
    state = framed[12]
    outs = mod.reset_crop_straighten(state)
    assert len(outs) == 19
    preview_tool = _update_dict(outs[16])
    assert preview_tool.get("value") == "frame"
    assert outs[17] == "frame"
    assert outs[14] == mod.DEFAULT_CROP_RECT


def test_frame_outputs_wire_preview_tool_and_drawer():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "preview_tool, active_drawer, crop_hint" in source
    assert 'elem_id="apply_framing_btn"' in source
    assert "const closeModule = (id) =>" in source
    assert "writeCropRectBox()" in source
