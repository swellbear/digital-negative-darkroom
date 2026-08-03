"""Fit-to-paper exposure / grade suggestions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import suggest_tone_fit, zone_reflectance


def _zones_to_refl(z_lo: float, z_hi: float, n: int = 64) -> np.ndarray:
    zones = np.linspace(z_lo, z_hi, n * n, dtype=np.float64)
    refl = 0.18 * (2.0 ** (zones - 5.0))
    return refl.astype(np.float32).reshape(n, n)


def test_fit_darkens_blown_highlights():
    # Midtones healthy, hot side past Zone VII.
    refl = np.full((48, 48), zone_reflectance(5.0), dtype=np.float32)
    refl[:, 36:] = float(zone_reflectance(9.0))
    fit = suggest_tone_fit(refl, base_seconds=8.0, grade=2.5)
    assert fit["ok"] == 1
    assert fit["base_seconds"] > 8.0
    assert fit["blown"]


def test_fit_lightens_crushed_shadows():
    refl = np.full((48, 48), zone_reflectance(5.0), dtype=np.float32)
    refl[:, :12] = float(zone_reflectance(0.4))
    fit = suggest_tone_fit(refl, base_seconds=8.0, grade=2.5)
    assert fit["ok"] == 1
    assert fit["base_seconds"] < 8.0
    assert fit["crushed"]


def test_wide_span_softens_grade():
    refl = _zones_to_refl(0.3, 9.6)
    fit = suggest_tone_fit(refl, base_seconds=8.0, grade=4.0)
    assert fit["ok"] == 1
    assert fit["grade"] < 4.0
    assert "grade" in fit["message"]


def test_fit_parks_midtones_not_paper_white_speculars():
    """Real paper tops out ~Z7.5. A stuck p95 must not keep lengthening a dark print."""
    # Mostly dark midtones with a thin paper-white strip (specular/sky).
    refl = np.full((64, 64), zone_reflectance(2.4), dtype=np.float32)
    refl[:, 60:] = float(zone_reflectance(7.4))  # ~6% at paper white
    fit = suggest_tone_fit(refl, base_seconds=48.0, grade=2.5)
    assert fit["ok"] == 1
    assert fit["base_seconds"] < 48.0
    assert "mid" in fit["message"].lower() or "balance" in fit["message"].lower()


def test_fit_darkens_when_midtones_are_blown_to_paper_white():
    # Whole frame piled on paper white — midtone park must add exposure.
    # (Shoulder-aware Fit takes a modest step; a second click continues.)
    refl = np.full((48, 48), float(0.93), dtype=np.float32)  # ~Z7.37
    fit = suggest_tone_fit(refl, base_seconds=2.0, grade=2.5)
    assert fit["ok"] == 1
    assert fit["base_seconds"] > 2.0
    assert fit["base_seconds"] >= 4.0
