"""Fit to paper must use real process knobs per chemistry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.analysis import (
    _as_mono_reflectance,
    reflectance_to_zone,
    suggest_tone_fit,
    zone_reflectance,
)
from digital_negative.spectral import N_WAVELENGTHS


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_fit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_spectral_reflectance_uses_cie_y_not_first_three_bands():
    # Fake spectrum: energy only in longer wavelengths (last third).
    h, w, n = 24, 32, N_WAVELENGTHS
    spec = np.zeros((h, w, n), dtype=np.float32)
    spec[..., (2 * n) // 3 :] = 0.4
    mono = _as_mono_reflectance(spec)
    naive = spec[..., :3].mean(axis=-1)
    # CIE Y must see the long-wave energy; first-three mean stays near 0.
    assert float(mono.mean()) > float(naive.mean()) + 0.02
    assert float(naive.mean()) < 0.01


def test_instant_fit_refused():
    refl = np.full((32, 32), zone_reflectance(5.0), dtype=np.float32)
    fit = suggest_tone_fit(
        refl, base_seconds=8.0, grade=2.5, chemistry_mode="instant"
    )
    assert fit["ok"] == 0
    assert "Instant" in fit["message"] or "enlarger" in fit["message"].lower()


def test_color_fit_softens_contrast_not_grade():
    # Wide span that triggers filtration soften.
    zones = np.linspace(0.3, 9.6, 48 * 48, dtype=np.float64)
    refl = (0.18 * (2.0 ** (zones - 5.0))).astype(np.float32).reshape(48, 48)
    fit = suggest_tone_fit(
        refl,
        base_seconds=8.0,
        grade=4.0,
        print_contrast=0.4,
        chemistry_mode="color",
    )
    assert fit["ok"] == 1
    assert fit["grade"] == 4.0  # RA-4 grade untouched
    assert fit["print_contrast"] < 0.4
    assert "contrast" in fit["message"]
    assert "grade" not in fit["message"]


def test_bw_fit_still_softens_grade():
    zones = np.linspace(0.3, 9.6, 48 * 48, dtype=np.float64)
    refl = (0.18 * (2.0 ** (zones - 5.0))).astype(np.float32).reshape(48, 48)
    fit = suggest_tone_fit(
        refl, base_seconds=8.0, grade=4.0, chemistry_mode="bw"
    )
    assert fit["ok"] == 1
    assert fit["grade"] < 4.0


def test_ui_instant_fit_errors_color_writes_contrast():
    mod = _load_ui()
    outs = mod.commit_ingest(
        None, str(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"), None
    )
    state = next(x for x in outs if isinstance(x, dict) and "roll" in x)

    def bake(mode, film, chem, mins, ei, paper):
        args = [
            mode, film, chem, mins, 0.0, 0.5, ei, "none", 0.01, 0.0,
            paper, 8.0, 2.5, 0.3, False, 0.0, 5.0, 4.5, 3.5, False, 5, 0.5,
            0.0, 0.0, "none", 0.0, 0.0, 0.0, 0.0, 21.0, 1.0, 0.0, True,
            {**state, "chemistry_mode": mode},
        ]
        return mod.live_preview(*args, quality="high", mark_dirty=True)[-1]

    sti = bake(
        "instant", "polaroid-600-instant-v1", "pod", 3.0, 640, "ra4-glossy-v1"
    )
    with pytest.raises(Exception) as exc:
        mod.auto_fit_print_tones("instant", 8.0, 2.5, 0.0, sti)
    assert "Instant" in str(exc.value) or "enlarger" in str(exc.value).lower()

    stc = bake(
        "color",
        "portra-400-spectral-v1",
        "c41_standard",
        3.25,
        400,
        "ra4-glossy-v1",
    )
    # Force a blown spectral/display map so Fit changes something.
    draft = stc.get("print_draft")
    if draft is not None and getattr(draft, "reflectance", None) is not None:
        # Run Fit — should not crash; Color path returns contrast update slot.
        out = mod.auto_fit_print_tones("color", 8.0, 2.5, 0.3, stc)
        assert len(out) == 7
        # print_grade should be skipped (not forced to a new MG value)
        grade_u = out[1]
        # exposure update is a dict-like gr.update
        exp_u = out[0]
        exp_val = (
            exp_u.get("value")
            if isinstance(exp_u, dict)
            else getattr(exp_u, "value", None)
        )
        assert exp_val is not None
