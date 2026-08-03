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
    assert mod._print_key_visible("instant", "print_grade") is False
    assert mod._print_key_visible("instant", "cc_cyan") is False


def test_chemistry_mode_change_hides_mg_shows_cc():
    mod = _load_ui()
    outs = mod.on_chemistry_mode_change("color")
    # film…help (6) + 11 print vis + Instant/Dev knobs + commit/unlock/download/state
    assert len(outs) == 6 + 11 + 12 + 6
    help_u = _upd(outs[5])
    assert "Color Chemistry" in str(help_u.get("value", ""))
    # print_grade is first visibility slot after help
    assert _upd(outs[6]).get("visible") is False  # print_grade
    assert _upd(outs[8]).get("visible") is True  # cc_cyan
    assert _upd(outs[16]).get("visible") is False  # tone

    outs_bw = mod.on_chemistry_mode_change("bw")
    assert _upd(outs_bw[6]).get("visible") is True
    assert _upd(outs_bw[8]).get("visible") is False
    assert "Black & White" in str(_upd(outs_bw[5]).get("value", ""))


def test_chemistry_mode_change_instant_shows_process_hides_print():
    mod = _load_ui()
    if not mod.FILM_CHOICES_INSTANT:
        return
    outs = mod.on_chemistry_mode_change("instant")
    assert len(outs) == 6 + 11 + 12 + 6
    assert "Instant" in str(_upd(outs[5]).get("value", ""))
    assert _upd(outs[6]).get("visible") is False  # print_grade
    assert _upd(outs[17]).get("visible") is True  # process_temp
    assert _upd(outs[20]).get("visible") is True  # Polaroid border
    assert _upd(outs[20]).get("value") is True
    assert _upd(outs[23]).get("visible") is False  # contrast_filter
    assert _upd(outs[24]).get("visible") is False  # scene_exposure
    assert _upd(outs[25]).get("visible") is False  # halation
    assert _upd(outs[27]).get("visible") is False  # print_drawer
    assert "Commit pull" in str(_upd(outs[28]).get("value", ""))
    assert _upd(outs[4]).get("visible") is False  # paper


def test_develop_commit_row_outside_accordion():
    """Commit pull must not sit inside the scrollable accordion (clips More)."""
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="develop_commit_row"' in source
    assert "Outside the accordion so Instant" in source
    assert "#drawer_host #drawer_develop.is-open" in source
    assert "overflow-y: auto !important" in source.split("#drawer_host #drawer_develop #acc_develop")[1][:400]


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
    assert "**E-6**" in banner  # bold chip — markdown code was white-on-white
    assert "Live exploring" in banner


def test_live_vs_committed_viewer_and_banner():
    mod = _load_ui()

    class _DN:
        def __init__(self, locks):
            self.metadata = {"ui_state": {"locked_stages": list(locks)}}

    exploring = {"dn": _DN(["ingest"]), "chemistry_mode": "bw"}
    committed = {"dn": _DN(["ingest", "development", "print"]), "chemistry_mode": "bw"}

    assert "not committed" in mod._viewer_label_for("live", exploring).lower()
    assert mod._viewer_label_for("live", committed).startswith("Committed print")
    assert "Live exploring" in mod._stage_banner("development", ["ingest"], exploring)
    assert "Committed" in mod._stage_banner("print", ["ingest", "development", "print"], committed)
    assert mod._lock_status_label(exploring) == "Live exploring"
    assert mod._lock_status_label(committed) == "Committed"
    # Default Live print stays quiet — "easel" implied dodge/burn was required.
    assert "easel" not in mod._live_print_label(exploring, tool="print").lower()
    assert "frame" in mod._live_print_label(exploring, tool="frame").lower()
    # on_preview_tool_change returns (stage update, crop accordion open).
    label_u, crop_u = mod.on_preview_tool_change("frame", committed)
    assert _upd(label_u).get("label", "").startswith("Committed print")
    assert _upd(crop_u).get("open") is True
    inspect_u, inspect_crop = mod.on_preview_tool_change("inspect", exploring)
    assert "inspect" in str(_upd(inspect_u).get("label", "")).lower()
    assert _upd(inspect_crop).get("open") is False


def test_advanced_dodge_burn_is_quarantined():
    mod = _load_ui()
    assert mod.ADVANCED_DODGE_BURN_LABEL.startswith("Advanced")
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "ADVANCED_DODGE_BURN_LABEL," in source
    assert 'elem_id="mod_dodge_burn"' in source
    # Default Print tip stays on Commit Print — Advanced is not in the primary hint.
    hint = source.split('elem_id="db_hint"', 1)[0][-320:]
    assert "Commit Print" in hint
    assert "Default: paper → exposure → filtration" in hint
    assert "Advanced" not in hint
    assert "Advanced · Dodge" in source  # context menu, labeled Advanced
    assert 'class="db-card-size" hidden' in source
    assert 'db-size-value' in source
    # Crop (common) before Advanced in the Modules panel.
    mod_chunk = source.split('elem_id="module_panel"', 1)[1][:25000]
    assert mod_chunk.index('elem_id="mod_crop"') < mod_chunk.index('elem_id="mod_dodge_burn"')
    # Base timer math should not advertise dodge/burn on the default path.
    assert "Dodge/burn passes are timed against this" not in mod._base_math_md(8.0)


def test_drawer_width_and_progressive_disclosure():
    mod = _load_ui()
    assert mod.DRAWER_WIDTH_PX >= 240
    assert f"--dr-drawer-width: {mod.DRAWER_WIDTH_PX}px" in mod.UI_CSS
    assert "var(--dr-drawer-width)" in mod.UI_CSS
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="acc_develop_more"' in source
    assert 'elem_id="acc_print_more"' in source
    assert "drawer-more" in source
    assert 'elem_id="how_darkroom_works"' in source
    assert "How this darkroom works" in mod.HOW_DARKROOM_WORKS_MD
    assert "Live preview" in mod.HOW_DARKROOM_WORKS_MD
    assert "Upload → Develop → Print" in mod.HOW_DARKROOM_WORKS_MD


def test_split_grade_children_hidden_until_enabled():
    mod = _load_ui()
    assert mod._split_grade_child_visible(False, "bw") is False
    assert mod._split_grade_child_visible(True, "bw") is True
    assert mod._split_grade_child_visible(True, "color") is False

    outs = mod.on_split_grade_toggle(False, "bw")
    assert all(_upd(u).get("visible") is False for u in outs)
    outs_on = mod.on_split_grade_toggle(True, "bw")
    assert all(_upd(u).get("visible") is True for u in outs_on)

    bands, stops = mod.on_test_strips_toggle(False)
    assert _upd(bands).get("visible") is False
    assert _upd(stops).get("visible") is False
    bands_on, stops_on = mod.on_test_strips_toggle(True)
    assert _upd(bands_on).get("visible") is True
    assert _upd(stops_on).get("visible") is True

    # Chemistry mode change with split off keeps soft/hard hidden.
    outs_bw = mod.on_chemistry_mode_change("bw", split_on=False)
    # soft_grade is the 7th visibility slot after help (index 5 + 1+1+3+1 = 11 → soft)
    # help=5, grade=6, contrast=7, cc_c=8, cc_m=9, cc_y=10, split=11, soft=12
    assert _upd(outs_bw[12]).get("visible") is False
    outs_split = mod.on_chemistry_mode_change("bw", split_on=True)
    assert _upd(outs_split[12]).get("visible") is True
