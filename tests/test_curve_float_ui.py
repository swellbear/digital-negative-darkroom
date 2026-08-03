"""Floating curve editor is wired into the Gradio darkroom UI."""

from __future__ import annotations

import importlib.util
import json
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


def test_instant_curves_omit_print_block():
    """Instant cards have no enlarger — curve float must not offer Exp/Grade."""
    mod = _load_ui()
    if not mod.FILM_CHOICES_INSTANT:
        return
    state = _state_from(mod.commit_ingest(None, str(FIXTURE), None))
    fid = mod.FILM_CHOICES_INSTANT[0][1]
    cid = mod.default_chemistry_id(mod._film_profile(fid))
    _summ, overlay = mod.refresh_curves(
        fid, cid, 3.0, 0.0, "mg-standard", 2.5, 8.0, state
    )
    data = json.loads(overlay)
    assert data.get("ok") is True
    assert data.get("print") is None
    assert {h["id"] for h in data["film"]["handles"]} == {"film_dev", "film_n"}
