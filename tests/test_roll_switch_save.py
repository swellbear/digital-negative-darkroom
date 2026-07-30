"""Camera roll: switch frames with save / discard when dirty."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"


def _load_ui():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_from(outputs):
    return next(x for x in outputs if isinstance(x, dict) and "roll" in x)


def _default_controls(mod):
    return (
        mod.FILM_CHOICES[0][1],
        mod._INIT_DEV_ID,
        mod._INIT_TNORM,
        0.0,
        1.0,
        400,
        "none",
        0.01,
        0.0,
        mod.PAPER_CHOICES[0][1],
        8.0,
        2.5,
        0.0,
        False,
        0.0,
        5.0,
        4.0,
        4.0,
        False,
        5,
        0.5,
        0.0,
        0.0,
        "none",
        0.0,
    )


def test_clean_switch_activates_target_frame():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert state["roll_index"] == 1
    assert not mod._is_dirty(state)

    outs = mod.begin_roll_switch(0, state)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert outs[-1] == -1  # pending cleared
    # modal visibility update
    assert getattr(outs[-2], "get", lambda *_: None)("visible", False) is False or outs[-2]["visible"] is False


def test_dirty_switch_prompts_then_save():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    state = {**state, "dirty": True, "summary_cache": "UNSAVED-EDIT"}

    outs = mod.begin_roll_switch(0, state)
    state = _state_from(outs)
    assert state["roll_index"] == 1  # still on current
    assert outs[-1] == 0  # pending target
    assert outs[-2]["visible"] is True

    ctrls = _default_controls(mod)
    outs = mod.save_and_switch_roll(0, state, *ctrls)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert not mod._is_dirty(state)
    assert outs[-2]["visible"] is False
    # Active frame's unsaved edit was written into its roll slot before switch.
    assert state["roll"][1]["summary_cache"] == "UNSAVED-EDIT"
    assert state["roll"][1].get("controls", {}).get("contrast") == 0.0


def test_dirty_switch_discard_keeps_last_saved():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    state = mod._sync_active_into_roll({**state, "summary_cache": "SAVED", "dirty": False})
    state = {**state, "dirty": True, "summary_cache": "UNSAVED-EDIT"}

    outs = mod.begin_roll_switch(0, state)
    assert outs[-2]["visible"] is True
    state = _state_from(outs)

    outs = mod.discard_and_switch_roll(0, state)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert state["roll"][1]["summary_cache"] == "SAVED"
    assert not mod._is_dirty(state)
