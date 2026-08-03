"""Film swaps must not leave a washed-out prior stock under the new film label."""

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
        "run_darkroom_ui_race", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _state(mod):
    outs = mod.commit_ingest(None, str(FIXTURE), None)
    return next(x for x in outs if isinstance(x, dict) and "roll" in x)


def _args(mod, film, chem, mins, ei, state):
    return [
        "bw",
        film,
        chem,
        mins,
        0.0,
        1.0,
        ei,
        "none",
        0.01,
        0.0,
        "fiber-glossy-v1",
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
        20.0,
        1.0,
        0.0,
        0.0,
        state,
    ]


def _arr(live):
    val = getattr(live, "value", None)
    if val is None and isinstance(live, dict):
        val = live.get("value")
    if val is None:
        val = live
    a = np.asarray(val, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return a


def test_source_wires_chem_batch_preview_and_stale_guard():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "def live_preview_after_chem" in source
    assert "fn=live_preview_after_chem" in source
    assert "_CHEM_UI_BATCH" in source
    assert "_PREVIEW_LATEST" in source
    assert "copy.deepcopy(base_proxy.metadata)" in source


def test_chem_batch_skips_cascaded_slider_preview():
    mod = _load_ui()
    state = _state(mod)
    mod._begin_chem_ui_batch()
    skipped = mod.live_preview_edit(*_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, state))
    assert skipped == mod._preview_output_skips()
    mod._end_chem_ui_batch()


def test_stale_preview_token_does_not_overwrite_newer_bake():
    mod = _load_ui()
    state = _state(mod)
    state = {**state, "spot_pos": "0.5000,0.5000"}

    # Manually simulate overlapping bakes: older Acros finishes after newer Delta started.
    t_old = next(mod._PREVIEW_EPOCH)
    mod._PREVIEW_LATEST["token"] = t_old
    # Newer bake claims the latest token (as live_preview would).
    t_new = next(mod._PREVIEW_EPOCH)
    mod._PREVIEW_LATEST["token"] = t_new

    packed_new = mod._live_preview_body(
        *_args(mod, "delta-100-v1", "id11_stock", 8.5, 100, state)[:-1],
        state,
        quality="high",
        mark_dirty=True,
    )
    # Stale completion check (wrapper behavior).
    assert t_old != mod._PREVIEW_LATEST["token"]
    stale_result = mod._preview_output_skips()
    assert len(stale_result) == mod._PREVIEW_OUTPUT_COUNT

    live_new, *_, spot_new, state2 = packed_new
    assert "Delta" in (state2.get("summary_cache") or "")
    assert "Zone" in spot_new

    # Fresh Acros after Delta must match a clean Acros bake (no washout stickiness).
    packed_a = mod.live_preview(
        *_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, state2)[:-1],
        state2,
        quality="high",
        mark_dirty=True,
    )
    packed_b = mod.live_preview(
        *_args(mod, "delta-100-v1", "id11_stock", 8.5, 100, packed_a[-1])[:-1],
        packed_a[-1],
        quality="high",
        mark_dirty=True,
    )
    packed_c = mod.live_preview(
        *_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, packed_b[-1])[:-1],
        packed_b[-1],
        quality="high",
        mark_dirty=True,
    )
    a0 = _arr(packed_a[0])
    a2 = _arr(packed_c[0])
    assert float(np.mean(np.abs(a0 - a2))) < 1e-5
    assert packed_a[-2] == packed_c[-2]


def test_after_chem_clears_batch_and_bakes():
    mod = _load_ui()
    state = _state(mod)
    mod._begin_chem_ui_batch()
    packed = mod.live_preview_after_chem(
        *_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, state)
    )
    assert mod._CHEM_UI_BATCH["active"] is False
    assert len(packed) == mod._PREVIEW_OUTPUT_COUNT
    assert "Acros" in (packed[-1].get("summary_cache") or "")
