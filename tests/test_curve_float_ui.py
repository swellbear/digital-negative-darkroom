"""Floating curve editor is wired into the Gradio darkroom UI."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_curve_float_ui_wiring():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert 'elem_id="curve_overlay_json"' in source
    assert 'elem_id="curve_edit_cmd"' in source
    assert 'elem_id="curve_show_btn"' in source
    assert "apply_curve_edit_cmd" in source
    assert "const showCurveFloat" in source
    assert "writeCurveEditCmd" in source
    assert "#curve_float" in source
    assert "curve_outputs = [curve_summary, curve_overlay_json]" in source
    # Chemistry-aware inputs: mode + RA-4 print_contrast.
    assert "chemistry_mode, film, developer, development_minutes, contrast" in source.replace(
        "\n", " "
    ) or "chemistry_mode" in source.split("curve_inputs = [")[1].split("]")[0]
    assert "print_contrast" in source.split("curve_inputs = [")[1].split("]")[0]
