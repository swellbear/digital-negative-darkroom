"""P0 smoke matrix: B&W↔Color + roll switch after Commit Develop."""

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
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _state_from(outputs):
    return next(x for x in outputs if isinstance(x, dict) and "roll" in x)


def _upd(update) -> dict:
    if isinstance(update, dict):
        return update
    raw = getattr(update, "__dict__", None) or {}
    if any(k in raw for k in ("visible", "value", "choices", "interactive")):
        return raw
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return {}


def _default_controls(mod, *, mode="bw"):
    if mode == "color":
        film_id = mod.FILM_CHOICES_COLOR[0][1]
        for _label, fid in mod.FILM_CHOICES_COLOR:
            try:
                if str(mod._film_profile(fid).type).lower() == "color_negative":
                    film_id = fid
                    break
            except Exception:
                continue
        profile = mod._film_profile(film_id)
        chem_id = mod.default_chemistry_id(profile)
        chem = mod.get_chemistry(profile, chem_id) or {"normal_minutes": 3.25}
        _tmin, _tmax, normal = mod.time_slider_bounds(chem)
        paper_id = mod.PAPER_CHOICES_COLOR[0][1]
        return (
            "color",
            film_id,
            chem_id,
            float(normal),
            0.0,
            1.0,
            float(profile.iso),
            "none",
            0.01,
            0.0,
            paper_id,
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
            20.0,
            40.0,
            0.0,
        )
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
    )


def _control_block(outs, mod):
    return outs[-(3 + mod._CONTROL_COUNT) : -3]


def _help_update(outs, mod):
    return outs[-(3)]


def test_bw_color_roll_matrix_no_control_leakage():
    """B&W↔Color with a loaded roll; switch frames after Commit Develop."""
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert state["roll_index"] == 1

    # Frame 1: Commit Develop in B&W — film controls lock.
    bw = _default_controls(mod, mode="bw")
    develop_args = bw[1:10]  # film…halation (no chemistry_mode)
    state = _state_from(mod.commit_develop(*develop_args, state))
    assert mod._locked(state, "development")
    assert not mod._locked(state, "print")

    # Persist B&W controls on the developed frame, switch to undeveloped frame 0.
    state = {**state, "controls": mod._capture_controls(*bw), "dirty": True}
    outs = mod.begin_roll_switch("0:click", state, *bw)
    state = _state_from(outs)
    outs = mod.save_and_switch_roll(0, state, *bw)
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert not mod._locked(state, "development")

    block = _control_block(outs, mod)
    assert _upd(block[0]).get("value") == "bw"
    assert _upd(block[1]).get("interactive") is True  # film re-enabled
    assert _upd(block[12]).get("visible") is True  # print_grade (MG)
    # cc_cyan index: paper=10, exposure=11, grade=12, contrast=13, split=14, ...
    # _PRINT_CONTROL_KEYS: paper, exposure, grade, contrast, split, soft, hard,
    # soft_s, hard_s, test_strips, bands, stops, flash, dry, tone, border, cc_c, cc_m, cc_y
    cc_cyan_i = 10 + mod._PRINT_CONTROL_KEYS.index("cc_cyan")
    assert _upd(block[cc_cyan_i]).get("visible") is False
    help_u = _upd(_help_update(outs, mod))
    assert "Black & White" in str(help_u.get("value", ""))

    # On the undeveloped frame, switch chemistry to Color — catalogs + CC/MG flip.
    color_outs = mod.on_chemistry_mode_change("color")
    assert _upd(color_outs[6]).get("visible") is False  # print_grade
    assert _upd(color_outs[8]).get("visible") is True  # cc_cyan
    color = list(_default_controls(mod, mode="color"))
    # Apply chemistry-mode catalog picks from on_chemistry_mode_change.
    color[1] = _upd(color_outs[0]).get("value") or color[1]
    color[2] = _upd(color_outs[1]).get("value") or color[2]
    color[3] = float(_upd(color_outs[2]).get("value") or color[3])
    color[6] = float(_upd(color_outs[3]).get("value") or color[6])
    color[10] = _upd(color_outs[4]).get("value") or color[10]

    state = {**state, "controls": mod._capture_controls(*color)}
    ctrls = mod._control_updates(state)
    assert _upd(ctrls[0]).get("value") == "color"
    assert _upd(ctrls[cc_cyan_i]).get("visible") is True
    assert _upd(ctrls[12]).get("visible") is False  # MG grade hidden
    # No stale B&W minutes leakage past C-41-friendly values when chem is color.
    assert float(_upd(ctrls[3]).get("value")) <= 5.5

    # Switch back to developed B&W frame — locks + MG restore, CC hidden.
    outs = mod.begin_roll_switch("1:click", state, *color)
    state = _state_from(outs)
    assert state["roll_index"] == 1
    assert mod._locked(state, "development")
    block = _control_block(outs, mod)
    assert _upd(block[0]).get("value") == "bw"
    assert _upd(block[1]).get("interactive") is False  # still locked
    assert _upd(block[12]).get("visible") is True
    assert _upd(block[cc_cyan_i]).get("visible") is False
    assert "Black & White" in str(_upd(_help_update(outs, mod)).get("value", ""))
    # Print controls stay interactive until Commit Print.
    assert _upd(block[10]).get("interactive") is True
    assert _upd(block[11]).get("interactive") is True

    banner = mod._stage_banner("print", mod._locks(state), state)
    assert "`B&W`" in banner
    assert "Live exploring" in banner


def test_default_print_flow_without_dodge_burn():
    """Commit Print works with empty Advanced dodge/burn (module closed)."""
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, path, None))
    bw = _default_controls(mod, mode="bw")
    state = _state_from(mod.commit_develop(*bw[1:10], state))
    assert mod._locked(state, "development")
    assert not (state.get("db_strokes") or [])

    # Live print path must produce a draft without opening Advanced.
    packed = mod.live_preview(*bw, state, quality="high", mark_dirty=False)
    state = packed[-1]
    assert state.get("print_draft") is not None
    assert "not committed" in mod._viewer_label_for("live", state).lower()

    outs = mod.commit_print(bw[10], bw[11], bw[12], bw[13], state)
    state = outs[-1]
    assert mod._locked(state, "print")
    assert "Committed" in (state.get("summary_cache") or "")
    assert mod._viewer_label_for("live", state).startswith("Committed print")
    # Still no dodge/burn requirement.
    assert not (state.get("dn").metadata.get("print", {}).get("dodge_burn") or [])
