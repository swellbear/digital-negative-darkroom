"""Recipes, spot readout, histogram, clipping overlay."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import (
    apply_clipping_overlay,
    render_print_histogram,
    spot_at,
    spot_markdown,
)
from digital_negative.recipes import build_recipe, load_recipe, save_recipe


def test_spot_and_histogram():
    r = np.full((32, 32), 0.18, dtype=np.float32)
    d = -np.log10(r)
    sample = spot_at(r, d, 0.5, 0.5)
    assert sample["ok"] == 1
    assert sample["zone_label"] == "V"
    assert "Zone" in spot_markdown(sample)
    hist = render_print_histogram(r)
    assert hist is not None and hist.ndim == 3


def test_clipping_overlay_tints():
    rgb = np.full((16, 16, 3), 120, dtype=np.uint8)
    refl = np.ones((16, 16), dtype=np.float32) * 0.95  # ~Zone VII+ / paper white
    out = apply_clipping_overlay(rgb, refl, show_highlights=True, show_shadows=False)
    assert out[..., 0].mean() > out[..., 1].mean()
    dark = np.full((16, 16), 0.01, dtype=np.float32)
    out2 = apply_clipping_overlay(rgb, dark, show_highlights=False, show_shadows=True)
    assert out2[..., 2].mean() > out2[..., 0].mean()


def test_recipe_roundtrip():
    recipe = build_recipe(
        film_id="hp5-plus-v1",
        developer_id="id11",
        development_minutes=7.5,
        contrast=0.1,
        grain=1.2,
        paper_id="mg-standard-v1",
        print_grade=3.0,
        print_exposure=10.0,
        name="cal-1",
    )
    path = Path(tempfile.mkdtemp()) / "r.json"
    save_recipe(path, recipe)
    loaded = load_recipe(path)
    assert loaded["film_id"] == "hp5-plus-v1"
    assert loaded["print_grade"] == 3.0
