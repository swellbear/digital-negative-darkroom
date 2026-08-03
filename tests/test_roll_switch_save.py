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
        "bw",
        mod.FILM_CHOICES_BW[0][1],
        mod._INIT_DEV_ID,
        mod._INIT_TNORM,
        0.0,
        1.0,
        400,
        "none",
        0.01,
        0.0,
        mod.PAPER_CHOICES_BW[0][1],
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
        0.0,
        0.0,
        0.0,
        # Instant knobs (process_temp, chroma, warmth, border)
        38.0,
        0.0,
        0.0,
        True,
    )


def test_clean_switch_activates_target_frame():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert state["roll_index"] == 1
    assert not mod._is_dirty(state)

    outs = mod.begin_roll_switch("0:click", state)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert outs[-1] == -1  # pending cleared
    # modal visibility update
    assert getattr(outs[-2], "get", lambda *_: None)("visible", False) is False or outs[-2]["visible"] is False

    # Blank change events must be ignored (textbox mount).
    outs = mod.begin_roll_switch("", state)
    assert _state_from(outs)["roll_index"] == 0


def test_dirty_switch_prompts_then_save():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    state = {**state, "dirty": True, "summary_cache": "UNSAVED-EDIT"}

    outs = mod.begin_roll_switch("0:click", state)
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

    outs = mod.begin_roll_switch("0:click", state)
    assert outs[-2]["visible"] is True
    state = _state_from(outs)

    outs = mod.discard_and_switch_roll(0, state)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert state["roll"][1]["summary_cache"] == "SAVED"
    assert not mod._is_dirty(state)


def _control_block(outs, mod):
    """Develop/Print control updates sit before chemistry_help + modal + pending."""
    return outs[-(3 + mod._CONTROL_COUNT) : -3]


def test_switch_after_develop_reenables_film_controls():
    """Commit Develop disables film controls; the next undeveloped frame must get them back."""
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert state["roll_index"] == 1

    develop_args = (
        mod.FILM_CHOICES_BW[0][1],
        mod._INIT_DEV_ID,
        mod._INIT_TNORM,
        0.0,
        1.0,
        400,
        "none",
        0.01,
        0.0,
    )
    state = _state_from(mod.commit_develop(*develop_args, False, "4", state))
    assert mod._locked(state, "development")

    # Simulate locked Develop UI, then save-and-switch to the other undeveloped frame.
    state = {**state, "dirty": True}
    outs = mod.begin_roll_switch("0:click", state, *_default_controls(mod))
    state = _state_from(outs)
    outs = mod.save_and_switch_roll(0, state, *_default_controls(mod))
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert not mod._locked(state, "development")

    # control block: chemistry_mode, film, developer, minutes, contrast, ...
    _mode_u, film_u, developer_u, minutes_u, contrast_u = _control_block(outs, mod)[:5]
    assert film_u.get("interactive") is True
    assert developer_u.get("interactive") is True
    assert minutes_u.get("interactive") is True
    assert contrast_u.get("interactive") is True

    # And Commit Develop must succeed on the newly active frame.
    state = _state_from(mod.commit_develop(*develop_args, False, "4", state))
    assert mod._locked(state, "development")


def test_frame_controls_do_not_leak_across_roll():
    """Edits on one frame must not appear on another unless loaded as a recipe."""
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert state["roll_index"] == 1

    edited = list(_default_controls(mod))
    edited[4] = 0.75  # contrast (after chemistry_mode + film + developer + minutes)
    edited[5] = 2.0  # grain
    edited[11] = 12.0  # print_exposure

    # Clean switch away from the edited frame — snapshot should stick on frame 1.
    outs = mod.begin_roll_switch("0:click", state, *edited)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert state["roll"][1]["controls"]["contrast"] == 0.75
    assert state["roll"][1]["controls"]["grain"] == 2.0
    assert state["roll"][1]["controls"]["print_exposure"] == 12.0

    # Active frame 0 keeps its own defaults (not the edited values).
    assert state["controls"]["contrast"] == 0.0
    assert state["controls"]["grain"] == 1.0
    assert state["controls"]["print_exposure"] == 8.0

    contrast_u = _control_block(outs, mod)[4]
    grain_u = _control_block(outs, mod)[5]
    print_exp_u = _control_block(outs, mod)[11]
    assert contrast_u.get("value") == 0.0
    assert grain_u.get("value") == 1.0
    assert print_exp_u.get("value") == 8.0

    # Switch back — frame 1's edits restore, not frame 0's defaults-as-leak.
    outs = mod.begin_roll_switch("1:click", state, *_default_controls(mod))
    state = _state_from(outs)
    assert state["roll_index"] == 1
    assert state["controls"]["contrast"] == 0.75
    assert state["controls"]["grain"] == 2.0
    contrast_u = _control_block(outs, mod)[4]
    grain_u = _control_block(outs, mod)[5]
    assert contrast_u.get("value") == 0.75
    assert grain_u.get("value") == 2.0
