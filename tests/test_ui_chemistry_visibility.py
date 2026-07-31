"""UI chemistry-mode visibility + path labels (no Gradio server)."""

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


def _upd(update) -> dict:
    if isinstance(update, dict):
        return update
    raw = getattr(update, "__dict__", None) or {}
    if "visible" in raw or "value" in raw or "choices" in raw:
        return raw
    for attr in ("constructor_args", "fields", "_data"):
        payload = getattr(update, attr, None)
        if isinstance(payload, dict):
            return payload
    return {}


def test_print_key_visibility_bw_vs_color():
    mod = _load_ui()
    assert mod._print_key_visible("bw", "print_grade") is True
    assert mod._print_key_visible("bw", "cc_cyan") is False
    assert mod._print_key_visible("color", "print_grade") is False
    assert mod._print_key_visible("color", "cc_magenta") is True
    assert mod._print_key_visible("color", "tone") is False
    assert mod._print_key_visible("bw", "paper_id") is None


def test_chemistry_mode_change_hides_mg_shows_cc():
    mod = _load_ui()
    outs = mod.on_chemistry_mode_change("color")
    # film, developer, minutes, ei, paper, help, then 11 visibility updates
    assert len(outs) == 6 + 11
    help_u = _upd(outs[5])
    assert "Color Chemistry" in str(help_u.get("value", ""))
    # print_grade is first visibility slot after help
    assert _upd(outs[6]).get("visible") is False  # print_grade
    assert _upd(outs[8]).get("visible") is True  # cc_cyan
    assert _upd(outs[-1]).get("visible") is False  # tone

    outs_bw = mod.on_chemistry_mode_change("bw")
    assert _upd(outs_bw[6]).get("visible") is True
    assert _upd(outs_bw[8]).get("visible") is False
    assert "Black & White" in str(_upd(outs_bw[5]).get("value", ""))


def test_path_and_strip_labels_for_e6_and_c41():
    mod = _load_ui()

    class _Dev:
        def __init__(self, process):
            self.color_process = process

    e6_state = {"development": _Dev("e6"), "chemistry_mode": "color"}
    c41_state = {"development": _Dev("c41"), "chemistry_mode": "color"}
    bw_state = {"chemistry_mode": "bw"}

    assert mod._path_label(e6_state) == "E-6"
    assert mod._path_label(c41_state) == "C-41"
    assert mod._path_label(bw_state) == "B&W"
    assert mod._film_strip_short_label("negative", e6_state) == "Slide"
    assert mod._film_strip_short_label("negative", c41_state) == "Neg"
    assert mod._film_strip_short_label("negative", bw_state) == "Negative"
    assert "slide" in mod._viewer_label_for("negative", e6_state).lower()
    assert "light table" in mod._viewer_label_for("negative", c41_state).lower()

    banner = mod._stage_banner("development", ["ingest"], e6_state)
    assert "`E-6`" in banner
