"""Inspect · zoom module: arming, high-res stage, histogram / clip smoke."""

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


def _upd(update) -> dict:
    if isinstance(update, dict):
        return update
    raw = getattr(update, "__dict__", None) or {}
    if "visible" in raw or "value" in raw or "choices" in raw or "label" in raw or "open" in raw:
        return raw
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return {}


class _Draft:
    def __init__(self, refl: np.ndarray):
        self.reflectance = refl
        self.print_density = -np.log10(np.maximum(refl, 1e-6))


class _DN:
    metadata = {"ui_state": {"locked_stages": ["ingest", "development"]}}


def _print_state(**extra):
    rgb = np.full((40, 56, 3), 110, dtype=np.uint8)
    inspect = np.full((100, 140, 3), 130, dtype=np.uint8)
    refl = np.full((40, 56), 0.18, dtype=np.float32)
    refl[:, 40:] = 0.95  # blown side for clipping
    state = {
        "dn": _DN(),
        "live_rgb": rgb,
        "live_inspect": inspect,
        "print_draft": _Draft(refl),
        "viewer_mode": "live",
        "chemistry_mode": "bw",
        "clip_hi": False,
        "clip_lo": False,
        "stage": "print",
    }
    state.update(extra)
    return state


def test_mod_inspect_arms_preview_tool_like_crop():
    """Accordion open must set preview_tool=inspect (not only the context menu)."""
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    idx = source.index("id === 'mod_inspect'")
    chunk = source[idx : idx + 900]
    assert "setPreviewToolValue('inspect')" in chunk
    assert "setPreviewToolValue('print')" in chunk
    assert "readPreviewTool() === 'inspect'" in chunk
    # Crop still mirrors Frame on its own accordion.
    crop_idx = source.index("id === 'mod_crop'")
    crop_chunk = source[crop_idx : crop_idx + 700]
    assert "setPreviewToolValue('frame')" in crop_chunk


def test_inspect_zoom_keeps_scale_across_image_refresh():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "keepZoom" in source
    assert "readPreviewTool() === 'inspect'" in source.split("keepZoom", 1)[1][:240]


def test_on_preview_tool_inspect_loads_high_res_buffer():
    mod = _load_ui()
    state = _print_state()
    live_u, crop_u = mod.on_preview_tool_change("inspect", state)
    payload = _upd(live_u)
    assert _upd(crop_u).get("open") is False
    assert "Inspect" in str(payload.get("label", ""))
    img = np.asarray(payload.get("value"))
    assert img.shape == state["live_inspect"].shape
    assert np.array_equal(img, state["live_inspect"])


def test_on_preview_tool_print_restores_live_proxy():
    mod = _load_ui()
    state = _print_state()
    live_u, crop_u = mod.on_preview_tool_change("print", state)
    payload = _upd(live_u)
    assert _upd(crop_u).get("open") is False
    img = np.asarray(payload.get("value"))
    assert img.shape == state["live_rgb"].shape
    assert "inspect" not in str(payload.get("label", "")).lower()


def test_refresh_inspect_tools_histogram_and_clip_on_inspect_surface():
    mod = _load_ui()
    state = _print_state()
    hist, tip, live_u, state2 = mod.refresh_inspect_tools(False, False, state)
    assert isinstance(hist, np.ndarray) and hist.ndim == 3
    assert "Histogram" in tip
    assert state2.get("clip_hi") is False
    img = np.asarray(_upd(live_u).get("value"))
    assert img.shape == state["live_inspect"].shape

    hist2, tip2, live_u2, state3 = mod.refresh_inspect_tools(True, False, state2)
    assert state3.get("clip_hi") is True
    clipped = np.asarray(_upd(live_u2).get("value"))
    assert clipped.shape == state["live_inspect"].shape
    # Blown side should shift redder than the mid grey inspect fill.
    assert float(clipped[..., 0].mean()) > float(clipped[..., 1].mean())


def test_pin_and_toggle_ab_smoke():
    mod = _load_ui()
    state = _print_state()
    state, tip, btn = mod.pin_ab_print(state)
    assert "pinned" in tip.lower()
    assert _upd(btn).get("interactive") is True
    live_u, tip2, btn2, state2 = mod.toggle_ab_print(state)
    assert "Viewing A" in tip2
    assert state2.get("ab_showing") == "A"
    img = np.asarray(_upd(live_u).get("value"))
    assert img.shape == state["live_rgb"].shape
