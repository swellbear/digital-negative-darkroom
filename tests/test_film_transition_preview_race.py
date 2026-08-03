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


def test_source_wires_atomic_film_preview():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "def on_film_change_and_preview" in source
    assert "fn=on_film_change_and_preview" in source
    assert "fn=on_developer_change_and_preview" in source
    assert "suppress_cascades" in source
    assert "copy.deepcopy(base_proxy.metadata)" in source
    # Old racing .then chain must be gone.
    assert "cancels=[film_preview_evt]" not in source
    assert "fn=live_preview_after_chem" not in source


def test_chem_cascade_skip_and_atomic_acros_tri_x_round_trip():
    """Acros and Tri-X share default d76 — the sharp race the user hit."""
    mod = _load_ui()
    state = _state(mod)
    state = {**state, "spot_pos": "0.5000,0.5000"}

    out_a = mod.on_film_change_and_preview(*_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, state))
    assert len(out_a) == 3 + mod._PREVIEW_OUTPUT_COUNT
    packed_a = out_a[3:]
    assert "Acros" in (packed_a[-1].get("summary_cache") or "")
    assert mod._CHEM_UI_BATCH["suppress_cascades"] >= 1

    # Cascaded developer.change from film rewrite must not bake again.
    out_dev = mod.on_developer_change_and_preview(
        *_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, packed_a[-1])
    )
    assert out_dev[1:] == mod._preview_output_skips()

    out_tx = mod.on_film_change_and_preview(
        *_args(mod, "tri-x-400-v1", "d76", 7.75, 400, packed_a[-1])
    )
    packed_tx = out_tx[3:]
    assert "Tri-X" in (packed_tx[-1].get("summary_cache") or "")

    out_a2 = mod.on_film_change_and_preview(
        *_args(mod, "acros-100-ii-v1", "d76", 7.75, 400, packed_tx[-1])
    )
    packed_a2 = out_a2[3:]
    # Resolved EI/minutes must be Acros defaults even if slider inputs were Tri-X.
    assert "Acros" in (packed_a2[-1].get("summary_cache") or "")
    assert "10 min" in (packed_a2[-1].get("summary_cache") or "")
    a0 = _arr(packed_a[0])
    a2 = _arr(packed_a2[0])
    assert float(np.mean(np.abs(a0 - a2))) < 1e-5
    assert packed_a[-2] == packed_a2[-2]


def test_stale_preview_token_does_not_overwrite_newer_bake():
    mod = _load_ui()
    state = _state(mod)
    state = {**state, "spot_pos": "0.5000,0.5000"}

    t_old = next(mod._PREVIEW_EPOCH)
    mod._PREVIEW_LATEST["token"] = t_old
    t_new = next(mod._PREVIEW_EPOCH)
    mod._PREVIEW_LATEST["token"] = t_new
    assert t_old != mod._PREVIEW_LATEST["token"]
    assert len(mod._preview_output_skips()) == mod._PREVIEW_OUTPUT_COUNT

    packed_a = mod.live_preview(
        *_args(mod, "acros-100-ii-v1", "d76", 10.0, 100, state)[:-1],
        state,
        quality="high",
        mark_dirty=True,
    )
    packed_b = mod.live_preview(
        *_args(mod, "tri-x-400-v1", "d76", 7.75, 400, packed_a[-1])[:-1],
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
    assert float(np.mean(np.abs(_arr(packed_a[0]) - _arr(packed_c[0])))) < 1e-5
