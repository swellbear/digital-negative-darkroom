"""Inspect · zoom Modules column stays compact when empty."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_inspect_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_refresh_inspect_empty_state_is_compact():
    mod = _load_ui()
    hist, tip, _frame, _state = mod.refresh_inspect_tools(False, False, {})
    assert hist is None
    assert "No print meter" in tip or "Live print" in tip
    assert "Histogram of print reflectance with Zone ticks" not in tip
    assert len(tip) < 180


def test_inspect_module_source_uses_compact_controls():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="inspect_clip_row"' in source
    assert 'elem_id="inspect_actions"' in source
    assert 'label="Blown"' in source
    assert 'label="Crushed"' in source
    assert "Blown (Z VII+)" not in source
    assert "#hist_plot:not(:has(img))" in source
