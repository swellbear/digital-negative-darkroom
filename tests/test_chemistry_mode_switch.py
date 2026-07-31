"""B&W ↔ Color chemistry switch must not break the Dev-time slider."""

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
    """Normalize Gradio update / dict to a plain mapping."""
    if isinstance(update, dict):
        return update
    # gr.update(...) returns an object with __dict__ or .get in some versions
    raw = getattr(update, "__dict__", None) or {}
    if "value" in raw or "maximum" in raw or "minimum" in raw:
        return raw
    # Gradio 4+ often stores fields under .constructor_args / similar
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return dict(update) if hasattr(update, "keys") else {}


def test_chemistry_mode_switch_keeps_minutes_in_slider_bounds():
    mod = _load_ui()
    # Simulate leaving Tri-X D-76 (7.75) and entering Color — previously Gradio
    # raised when max shrank to C-41's 5.5 while 7.75 was still in the payload.
    outs = mod.on_chemistry_mode_change("color")
    film_u, dev_u, minutes_u, ei_u, paper_u = outs[:5]
    minutes = _update_dict(minutes_u)
    assert minutes.get("minimum") == mod._DEV_TIME_SLIDER_MIN
    assert minutes.get("maximum") == mod._DEV_TIME_SLIDER_MAX
    value = float(minutes["value"])
    assert mod._DEV_TIME_SLIDER_MIN <= value <= mod._DEV_TIME_SLIDER_MAX
    assert value <= 5.5  # C-41 normal is well under the old trap max

    film = _update_dict(film_u)
    assert film.get("value") in {c[1] for c in mod.FILM_CHOICES_COLOR}

    # Round-trip back to B&W still keeps the wide stable span.
    outs_bw = mod.on_chemistry_mode_change("bw")
    minutes_bw = _update_dict(outs_bw[2])
    assert minutes_bw.get("maximum") == mod._DEV_TIME_SLIDER_MAX
    assert mod._DEV_TIME_SLIDER_MIN <= float(minutes_bw["value"]) <= mod._DEV_TIME_SLIDER_MAX


def test_chem_time_update_never_narrows_below_tri_x_normal():
    mod = _load_ui()
    # Portra C-41 chem max is 5.5; slider max must stay wide enough for 7.75.
    update = _update_dict(mod._chem_time_update("portra-400-spectral-v1", "c41_standard"))
    assert float(update["maximum"]) >= 7.75
    assert float(update["value"]) == 3.25


def test_stale_bw_film_ignored_after_color_switch():
    """B&W→Color: film.change may re-fire with Acros against the Color catalog."""
    mod = _load_ui()
    # Stale Acros while Chemistry is already Color — must not overwrite C-41.
    # gr.skip() is an empty update (no value/choices) in Gradio 6.
    skipped = mod.on_film_change("acros-100-ii-v1", "color")
    for upd in skipped:
        payload = _update_dict(upd)
        assert "value" not in payload
        assert "choices" not in payload

    coerced = mod._coerce_film_id("acros-100-ii-v1", "color")
    assert coerced in {c[1] for c in mod.FILM_CHOICES_COLOR}
    assert coerced != "acros-100-ii-v1"

    # Valid Color film still refreshes developer / EI.
    outs = mod.on_film_change(coerced, "color")
    assert "value" in _update_dict(outs[0])

    # Chemistry handler always lands on a Color catalog id.
    film_u = _update_dict(mod.on_chemistry_mode_change("color")[0])
    assert film_u.get("value") in {c[1] for c in mod.FILM_CHOICES_COLOR}
    assert film_u.get("value") != "acros-100-ii-v1"


def test_chemistry_switch_unlocks_instant_commit_and_swaps_catalog():
    """Instant Commit pull locked Print; switching to Color must unlock + swap film."""
    mod = _load_ui()
    if not mod.FILM_CHOICES_INSTANT:
        return

    class _DN:
        def __init__(self):
            self.metadata = {
                "ui_state": {
                    "locked_stages": ["ingest", "development", "print"],
                    "committed_stages": ["ingest", "development", "print"],
                },
                "history": [],
            }

        def touch(self):
            return None

    state = {
        "dn": _DN(),
        "chemistry_mode": "instant",
        "development_full": object(),
        "print_draft": object(),
        "controls": {"chemistry_mode": "instant", "film_id": "polaroid-600-instant-v1"},
    }
    outs = mod.on_chemistry_mode_change("color", False, state)
    film_u = _update_dict(outs[0])
    assert film_u.get("value") in {c[1] for c in mod.FILM_CHOICES_COLOR}
    assert _update_dict(outs[-7]).get("interactive") is True  # Commit Develop
    new_state = outs[-1]
    assert new_state is not None and new_state is not state
    assert "development" not in new_state["dn"].metadata["ui_state"]["locked_stages"]
    assert "print" not in new_state["dn"].metadata["ui_state"]["locked_stages"]
    assert new_state.get("development_full") is None
    assert new_state.get("chemistry_mode") == "color"

    # Round-trip into Instant lands on an Instant catalog id and unlocks.
    outs_i = mod.on_chemistry_mode_change("instant", False, new_state)
    assert _update_dict(outs_i[0]).get("value") in {c[1] for c in mod.FILM_CHOICES_INSTANT}
    assert "Instant" in str(_update_dict(outs_i[5]).get("value", ""))


def test_chemistry_help_update_supports_instant():
    mod = _load_ui()
    state = {"controls": {"chemistry_mode": "instant"}}
    help_u = _update_dict(mod._chemistry_help_update(state))
    assert "Instant" in str(help_u.get("value", ""))
