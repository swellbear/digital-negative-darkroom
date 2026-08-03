"""UI smoke: AI enlarge export controls (no Gradio server)."""

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


def test_ai_enlarge_controls_in_download_strip():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="ai_enlarge"' in source
    assert 'elem_id="ai_enlarge_scale"' in source
    assert 'elem_id="ai_enlarge_tip"' in source
    assert "Download matches the committed on-screen print" in source
    assert "invented-detail" in source
    # Shared strip outside Print drawer so Instant can use it.
    print_chunk = source.split('elem_id="drawer_print"')[1].split(
        'elem_id="drawer_frame"'
    )[0]
    assert "ai_enlarge" not in print_chunk
    assert 'elem_id="download_row"' in source
    # Wired into both commit paths; not into live preview control count.
    assert "ai_enlarge, ai_enlarge_scale, state" in source
    assert "maybe_ai_upscale_rgb" in source


def test_ai_enlarge_scale_helper():
    mod = _load_ui()
    assert mod._ai_enlarge_scale("2") == 2
    assert mod._ai_enlarge_scale("2×") == 2
    assert mod._ai_enlarge_scale("4") == 4
    assert mod._ai_enlarge_scale(None) == 4
    assert mod._ai_enlarge_scale("bogus") == 4
