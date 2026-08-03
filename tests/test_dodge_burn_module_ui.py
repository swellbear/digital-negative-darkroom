"""Dodge & burn Modules column uses compact labels and pass math."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_db_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_db_ui_choices_are_compact():
    mod = _load_ui()
    shape_labels = [label for label, _ in mod.DB_SHAPE_UI_CHOICES]
    mode_labels = [label for label, _ in mod.DB_MODE_UI_CHOICES]
    assert "Soft oval" in shape_labels
    assert not any("default card" in label.lower() for label in shape_labels)
    assert max(len(label) for label in shape_labels) <= 12
    assert mode_labels == ["Dodge · lighter", "Burn · darker"]
    assert {v for _, v in mod.DB_SHAPE_UI_CHOICES} == {
        "soft_oval",
        "circle",
        "finger",
        "card",
        "custom",
    }
    assert {v for _, v in mod.DB_MODE_UI_CHOICES} == {"dodge", "burn"}


def test_pass_math_md_is_compact():
    mod = _load_ui()
    burn = mod._pass_math_md(8.0, 4.0, "burn")
    dodge = mod._pass_math_md(8.0, 4.0, "dodge")
    assert "Pass math —" not in burn
    assert "**Pass**" in burn and "burn" in burn
    assert "**Pass**" in dodge and "dodge" in dodge
    assert len(burn) < 120
    assert len(dodge) < 120
    assert "held still" not in burn


def test_dodge_module_source_uses_compact_controls():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="db_start_btn"' in source
    assert 'elem_id="db_reset_btn"' in source
    assert 'elem_id="db_shape"' in source
    assert "DB_SHAPE_UI_CHOICES" in source
    assert "DB_MODE_UI_CHOICES" in source
    assert "Start — wave over print" not in source
    assert "Reset local work" not in source
