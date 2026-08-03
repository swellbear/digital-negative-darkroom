"""Parametric curve-handle edits map onto real darkroom controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import CurveReport, build_curve_report
from digital_negative.curve_edit import apply_curve_handle_edit, curve_overlay_payload
from digital_negative.curves import load_film_profile
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.papers import load_paper_profile


def _tri_x_report():
    film = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    # Mild scene so percentiles exist.
    img = np.clip(np.linspace(0.05, 1.2, 80 * 100, dtype=np.float32).reshape(80, 100), 0, None)
    dn = DigitalNegative(image=img, metadata=default_metadata())
    return build_curve_report(
        dn,
        film,
        relative_time=1.0,
        contrast_modifier=0.0,
        developer_id="d76",
        development_minutes=7.75,
        paper=paper,
        grade=2.5,
        base_exposure_seconds=8.0,
    ), film


def test_overlay_payload_has_handles_and_polylines():
    report, _ = _tri_x_report()
    payload = curve_overlay_payload(
        report,
        development_minutes=7.75,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert payload["ok"]
    assert len(payload["film"]["polyline"]) >= 8
    assert {h["id"] for h in payload["film"]["handles"]} == {"film_dev", "film_n"}
    assert payload["print"] is not None
    assert {h["id"] for h in payload["print"]["handles"]} == {"print_exp", "print_grade"}


def test_film_dev_handle_lengthens_time_when_dragged_up():
    out = apply_curve_handle_edit(
        "film_dev",
        dy=0.25,
        development_minutes=8.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert out["ok"]
    assert out["development_minutes"] > 8.0


def test_film_n_handle_raises_contrast():
    out = apply_curve_handle_edit(
        "film_n",
        dy=0.3,
        development_minutes=8.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert out["contrast"] > 0.0


def test_print_exp_brighter_shortens_timer():
    out = apply_curve_handle_edit(
        "print_exp",
        dy=0.3,
        development_minutes=8.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert out["print_exposure"] < 8.0


def test_print_grade_handle_hardens_filtration():
    out = apply_curve_handle_edit(
        "print_grade",
        dy=0.25,
        development_minutes=8.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert out["print_grade"] > 2.5


def test_unknown_handle_rejected():
    out = apply_curve_handle_edit(
        "nope",
        dy=0.2,
        development_minutes=8.0,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert out["ok"] is False


def test_ra4_overlay_omits_mg_grade_handle():
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    img = np.clip(np.linspace(0.05, 1.2, 80 * 100, dtype=np.float32).reshape(80, 100), 0, None)
    dn = DigitalNegative(image=img, metadata=default_metadata())
    report = build_curve_report(
        dn,
        film,
        relative_time=1.0,
        contrast_modifier=0.0,
        developer_id="c41_standard",
        development_minutes=3.25,
        paper=paper,
        grade=2.5,
        base_exposure_seconds=8.0,
    )
    payload = curve_overlay_payload(
        report,
        development_minutes=3.25,
        contrast=0.0,
        print_grade=2.5,
        print_exposure=8.0,
    )
    assert payload["print"] is not None
    assert {h["id"] for h in payload["print"]["handles"]} == {"print_exp"}
    assert report.stats.get("paper_type") == "color_ra4"
