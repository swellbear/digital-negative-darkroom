"""Crop & straighten Modules column must not crush Rule / action controls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_crop_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_auto_crop_rule_ui_labels_are_short():
    mod = _load_ui()
    labels = [label for label, _value in mod.AUTO_CROP_RULE_UI_CHOICES]
    assert "Auto" in labels
    # Long Gradio-truncated form must not be the Modules dropdown label.
    assert not any("best score" in label.lower() for label in labels)
    assert max(len(label) for label in labels) <= 12
    values = {value for _label, value in mod.AUTO_CROP_RULE_UI_CHOICES}
    assert values == {
        "auto",
        "rule_of_thirds",
        "golden_ratio",
        "center",
        "horizon_thirds",
        "leading_room",
    }


def test_crop_module_keeps_crop_rect_and_short_rule_choices():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="auto_crop_rule"' in source
    assert 'elem_id="auto_crop_btn"' in source
    assert 'elem_id="crop_rect"' in source
    assert "AUTO_CROP_RULE_UI_CHOICES" in source
    # Parked off-screen for JS — never Gradio-removed from the DOM.
    assert "do not use visible=False" in source
    assert 'elem_id="straighten_deg"' in source
    assert 'elem_id="auto_straighten_btn"' in source
