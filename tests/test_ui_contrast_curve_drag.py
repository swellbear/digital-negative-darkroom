"""Spot / ritual values readable; curve float commits on release only."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_preview_value_chips_have_dark_theme_contrast():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "#ritual_status code" in source
    assert "#spot_readout code" in source
    assert "#spot_readout .prose strong" in source
    assert "color: #e0954f" in source or "color: #f0b06a" in source
    # Path chip must not use markdown code spans (white-on-white on the float).
    assert 'extras.append(f"**{path}**")' in source


def test_spot_markdown_avoids_code_chips():
    from digital_negative.analysis import spot_markdown

    md = spot_markdown(
        {
            "ok": True,
            "zone_label": "III",
            "zone": 3.0,
            "density": 2.06,
            "reflectance": 0.033,
        }
    )
    assert "`" not in md
    assert "**III**" in md
    assert "**2.06**" in md
    assert "**0.033**" in md


def test_curve_handle_commits_on_pointerup_only():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "no Gradio traffic until release" in source
    assert "writeCurveEditCmd({ id: g.getAttribute('data-handle'), dy: totalDy })" in source
    # Mid-drag onMove must not call writeCurveEditCmd (commit lives in onUp).
    move_section = source.split("const onMove = (e) => {")[1].split("const onUp = (e) => {")[0]
    assert "writeCurveEditCmd" not in move_section
    assert "writeCurveEditCmd" in source.split("const onUp = (e) => {")[1].split(
        "circ.addEventListener('pointerdown'"
    )[0]
