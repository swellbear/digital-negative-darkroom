"""Curves editor must match the live process per chemistry mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import build_curve_report
from digital_negative.curve_edit import apply_curve_handle_edit, curve_overlay_payload
from digital_negative.curves import load_film_profile
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.papers import load_paper_profile


def _scene():
    img = np.clip(np.linspace(0.05, 1.2, 60 * 80, dtype=np.float32).reshape(60, 80), 0, None)
    return DigitalNegative(
        image=np.stack([img, img * 0.95, img * 0.85], axis=-1),
        metadata=default_metadata(),
    )


def test_color_overlay_has_exp_and_contrast_not_grade():
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    report = build_curve_report(
        _scene(),
        film,
        developer_id="c41_standard",
        development_minutes=3.25,
        paper=paper,
        grade=2.5,
        base_exposure_seconds=8.0,
        print_contrast=0.2,
    )
    assert report.stats.get("print_process") == "ra4"
    assert "grade" not in report.stats
    payload = curve_overlay_payload(
        report,
        development_minutes=3.25,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        print_contrast=0.2,
        chemistry_mode="color",
    )
    ids = {h["id"] for h in payload["print"]["handles"]}
    assert ids == {"print_exp", "print_contrast"}
    assert "print_grade" not in ids
    assert "grade" not in payload["print"]["title"].lower()
    assert "RA-4" in payload["foot"] or "contrast" in payload["foot"].lower()


def test_color_grade_handle_edit_rejected_contrast_accepted():
    bad = apply_curve_handle_edit(
        "print_grade",
        dy=0.4,
        development_minutes=3.25,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        print_contrast=0.0,
        chemistry_mode="color",
    )
    assert bad["ok"] is False
    good = apply_curve_handle_edit(
        "print_contrast",
        dy=0.3,
        development_minutes=3.25,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        print_contrast=0.0,
        chemistry_mode="color",
    )
    assert good["ok"] is True
    assert good["print_contrast"] > 0.0


def test_instant_overlay_has_no_print_panel_and_uses_pod_layers():
    film = load_film_profile(ROOT / "profiles" / "films" / "polaroid-600-instant-v1.json")
    report = build_curve_report(
        _scene(),
        film,
        developer_id="pod",
        development_minutes=3.0,
        contrast_modifier=0.0,
        paper=None,
    )
    assert report.stats.get("curve_source") == "instant_layers"
    assert report.print_reflectance is None
    payload = curve_overlay_payload(
        report,
        development_minutes=3.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        chemistry_mode="instant",
    )
    assert payload["print"] is None
    assert {h["id"] for h in payload["film"]["handles"]} == {"film_dev", "film_n"}
    assert "enlarger" in payload["foot"].lower() or "pod" in payload["foot"].lower()


def test_instant_print_handle_edit_rejected():
    out = apply_curve_handle_edit(
        "print_exp",
        dy=0.3,
        development_minutes=3.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        chemistry_mode="instant",
    )
    assert out["ok"] is False


def test_bw_still_has_grade_handle():
    film = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    report = build_curve_report(
        _scene(),
        film,
        developer_id="d76",
        development_minutes=6.75,
        paper=paper,
        grade=2.5,
        base_exposure_seconds=8.0,
    )
    payload = curve_overlay_payload(
        report,
        development_minutes=6.75,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
        chemistry_mode="bw",
    )
    assert {h["id"] for h in payload["print"]["handles"]} == {"print_exp", "print_grade"}


def test_ui_refresh_curves_chemistry_aware():
    import importlib.util

    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_curves", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    outs = mod.commit_ingest(None, str(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"), None)
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)

    _s, overlay = mod.refresh_curves(
        "color",
        "portra-400-spectral-v1",
        "c41_standard",
        3.25,
        0.0,
        "ra4-glossy-v1",
        2.5,
        8.0,
        0.0,
        {**state, "chemistry_mode": "color"},
    )
    payload = json.loads(overlay)
    assert payload["chemistry_mode"] == "color"
    assert {h["id"] for h in payload["print"]["handles"]} == {"print_exp", "print_contrast"}

    _s2, overlay_i = mod.refresh_curves(
        "instant",
        "polaroid-600-instant-v1",
        "pod",
        3.0,
        0.0,
        "ra4-glossy-v1",
        2.5,
        8.0,
        0.0,
        {**state, "chemistry_mode": "instant"},
    )
    payload_i = json.loads(overlay_i)
    assert payload_i["print"] is None
    assert payload_i["stats"].get("curve_source") == "instant_layers" or (
        "pod" in (payload_i.get("film") or {}).get("title", "").lower()
        or payload_i.get("chemistry_mode") == "instant"
    )
