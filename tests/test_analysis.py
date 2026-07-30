"""Curve inspector / zone helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import (
    build_curve_report,
    curve_summary_markdown,
    reflectance_to_zone,
    render_curve_plot,
    zone_reflectance,
)
from digital_negative.curves import load_film_profile
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile


def test_zone_reflectance_anchors():
    assert abs(zone_reflectance(5) - 0.18) < 1e-9
    assert abs(float(reflectance_to_zone(0.18)) - 5.0) < 1e-6


def test_curve_report_and_plot():
    dn = ingest_path(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png")
    profile = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")

    report = build_curve_report(
        dn,
        profile,
        relative_time=1.0,
        developer_id="d76",
        development_minutes=9.75,
        paper=paper,
        grade=2.0,
        base_exposure_seconds=8.0,
    )
    assert report.stats["contrast_index"] > 0
    assert report.stats["scene_stops"] > 0
    assert report.print_reflectance is not None
    assert "shadow_zone" in report.stats

    plot = render_curve_plot(report)
    assert isinstance(plot, np.ndarray)
    assert plot.ndim == 3 and plot.shape[2] == 3
    assert plot.shape[0] > 200 and plot.shape[1] > 200

    md = curve_summary_markdown(report)
    assert "CI" in md
    assert "Zone" in md
