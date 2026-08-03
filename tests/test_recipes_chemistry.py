"""Recipes must round-trip Color CC and Instant process knobs."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.recipes import build_recipe, load_recipe, save_recipe


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_recipes", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _update_value(u):
    if isinstance(u, dict):
        return u.get("value")
    return getattr(u, "value", u)


def test_build_recipe_stores_chemistry_extras():
    recipe = build_recipe(
        film_id="portra-400-spectral-v1",
        developer_id="c41_standard",
        development_minutes=3.25,
        contrast=0.0,
        grain=0.5,
        paper_id="ra4-glossy-v1",
        print_grade=2.5,
        print_exposure=10.0,
        print_contrast=0.35,
        chemistry_mode="color",
        name="portra-cc",
        extras={
            "cc_cyan": 5.0,
            "cc_magenta": 20.0,
            "cc_yellow": 10.0,
        },
    )
    path = Path(tempfile.mkdtemp()) / "color.json"
    save_recipe(path, recipe)
    loaded = load_recipe(path)
    assert loaded["chemistry_mode"] == "color"
    assert loaded["print_contrast"] == 0.35
    assert loaded["extensions"]["cc_magenta"] == 20.0


def test_export_includes_color_cc_and_instant_knobs():
    mod = _load_ui()
    out = mod.export_recipe_file(
        "color",
        "portra-400-spectral-v1",
        "c41_standard",
        3.25,
        0.0,
        0.5,
        400,
        "none",
        0.01,
        0.0,
        "ra4-glossy-v1",
        2.5,
        10.0,
        0.4,
        "color-cal",
        cc_cyan=5.0,
        cc_magenta=25.0,
        cc_yellow=15.0,
    )
    path = _update_value(out)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["extensions"]["cc_cyan"] == 5.0
    assert data["extensions"]["cc_magenta"] == 25.0
    assert data["extensions"]["cc_yellow"] == 15.0
    assert data["print_contrast"] == 0.4

    out_i = mod.export_recipe_file(
        "instant",
        "polaroid-600-instant-v1",
        "pod",
        4.0,
        0.1,
        0.7,
        640,
        "none",
        0.01,
        0.0,
        "ra4-glossy-v1",
        2.5,
        8.0,
        0.0,
        "instant-cal",
        process_temp_c=24.0,
        instant_chroma=0.65,
        instant_warmth=0.15,
        instant_border=False,
    )
    path_i = _update_value(out_i)
    data_i = json.loads(Path(path_i).read_text(encoding="utf-8"))
    ext = data_i["extensions"]
    assert ext["process_temp_c"] == 24.0
    assert ext["instant_chroma"] == 0.65
    assert ext["instant_warmth"] == 0.15
    assert ext["instant_border"] is False


def test_apply_restores_color_cc_and_skips_mode_reset():
    mod = _load_ui()
    outs = mod.commit_ingest(
        None, str(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"), None
    )
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)
    args = [
        "color",
        "portra-160-spectral-v1",
        "c41_standard",
        3.25,
        0.0,
        0.5,
        160,
        "none",
        0.01,
        0.0,
        "ra4-glossy-v1",
        6.0,
        2.5,
        0.0,
        False,
        0.0,
        5.0,
        4.5,
        3.5,
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
        21.0,
        1.0,
        0.0,
        True,
        {**state, "chemistry_mode": "color"},
    ]
    state = mod.live_preview(*args, quality="high", mark_dirty=True)[-1]
    state = {
        **state,
        "chemistry_mode": "color",
        "controls": {**(state.get("controls") or {}), "chemistry_mode": "color"},
    }

    recipe_path = Path(tempfile.mkdtemp()) / "portra.json"
    save_recipe(
        recipe_path,
        build_recipe(
            film_id="portra-400-spectral-v1",
            developer_id="c41_standard",
            development_minutes=3.25,
            contrast=0.0,
            grain=0.5,
            paper_id="ra4-glossy-v1",
            print_grade=2.5,
            print_exposure=11.0,
            print_contrast=0.3,
            chemistry_mode="color",
            name="portra-cc",
            extras={
                "exposure_index": 400,
                "cc_cyan": 5.0,
                "cc_magenta": 30.0,
                "cc_yellow": 12.0,
            },
        ),
    )
    out = mod.apply_recipe_file(str(recipe_path), "color", state)
    assert len(out) == 24
    # chemistry_mode left untouched (gr.skip → bare update) so mode-change
    # cannot reset film/paper catalogs out from under the recipe.
    mode_u = out[0]
    assert isinstance(mode_u, dict)
    assert "value" not in mode_u
    assert _update_value(out[1]) == "portra-400-spectral-v1"
    assert _update_value(out[12]) == 11.0  # print_exposure
    assert _update_value(out[13]) == 0.3  # print_contrast
    assert _update_value(out[14]) == 5.0  # cc_cyan
    assert _update_value(out[15]) == 30.0
    assert _update_value(out[16]) == 12.0
    ctrl = out[-1].get("controls") or {}
    assert ctrl.get("cc_magenta") == 30.0
    assert ctrl.get("film_id") == "portra-400-spectral-v1"


def test_apply_restores_instant_process_knobs():
    mod = _load_ui()
    outs = mod.commit_ingest(
        None, str(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"), None
    )
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)
    args = [
        "instant",
        "polaroid-600-instant-v1",
        "pod",
        3.0,
        0.0,
        0.5,
        640,
        "none",
        0.01,
        0.0,
        "ra4-glossy-v1",
        8.0,
        2.5,
        0.0,
        False,
        0.0,
        5.0,
        4.5,
        3.5,
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
        21.0,
        1.0,
        0.0,
        True,
        {**state, "chemistry_mode": "instant"},
    ]
    state = mod.live_preview(*args, quality="high", mark_dirty=True)[-1]
    state = {
        **state,
        "chemistry_mode": "instant",
        "controls": {**(state.get("controls") or {}), "chemistry_mode": "instant"},
    }

    recipe_path = Path(tempfile.mkdtemp()) / "instant.json"
    save_recipe(
        recipe_path,
        build_recipe(
            film_id="polaroid-600-instant-v1",
            developer_id="pod",
            development_minutes=4.5,
            contrast=0.15,
            grain=0.8,
            paper_id="ra4-glossy-v1",
            print_grade=2.5,
            print_exposure=8.0,
            chemistry_mode="instant",
            name="600-warm",
            extras={
                "exposure_index": 640,
                "process_temp_c": 24.0,
                "instant_chroma": 0.7,
                "instant_warmth": 0.25,
                "instant_border": False,
            },
        ),
    )
    out = mod.apply_recipe_file(str(recipe_path), "instant", state)
    assert _update_value(out[3]) == 4.5  # process minutes
    assert _update_value(out[17]) == 24.0  # temp
    assert _update_value(out[18]) == 0.7  # chroma
    assert _update_value(out[19]) == 0.25  # warmth
    assert _update_value(out[20]) is False  # border
    ctrl = out[-1].get("controls") or {}
    assert ctrl.get("process_temp_c") == 24.0
    assert ctrl.get("instant_chroma") == 0.7
    assert ctrl.get("instant_border") is False


def test_recipe_upload_path_accepts_uploadbutton_payloads():
    mod = _load_ui()
    assert mod._recipe_upload_path(None) is None
    assert mod._recipe_upload_path("/tmp/a.json") == "/tmp/a.json"
    assert mod._recipe_upload_path(["/tmp/b.json"]) == "/tmp/b.json"
    assert mod._recipe_upload_path({"path": "/tmp/c.json"}) == "/tmp/c.json"

    class _F:
        name = "/tmp/d.json"

    assert mod._recipe_upload_path(_F()) == "/tmp/d.json"


def test_apply_mode_mismatch_uses_ui_chemistry_radio():
    mod = _load_ui()
    outs = mod.commit_ingest(
        None, str(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"), None
    )
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)
    # State claims color, but UI radio is still B&W — must refuse Color recipe.
    state = {
        **state,
        "chemistry_mode": "color",
        "controls": {"chemistry_mode": "color"},
        "dn": state.get("dn") or object(),
    }
    recipe_path = Path(tempfile.mkdtemp()) / "color.json"
    save_recipe(
        recipe_path,
        build_recipe(
            film_id="portra-400-spectral-v1",
            developer_id="c41_standard",
            development_minutes=3.25,
            contrast=0.0,
            grain=0.5,
            paper_id="ra4-glossy-v1",
            print_grade=2.5,
            print_exposure=8.0,
            chemistry_mode="color",
            name="c",
        ),
    )
    with pytest.raises(Exception) as exc:
        mod.apply_recipe_file(str(recipe_path), "bw", state)
    assert "COLOR" in str(exc.value)
