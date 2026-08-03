"""Color chemistry UI: mode switch, stock round-trip, E-6 finish path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"


def _load_ui():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui_color", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _state(mod):
    outs = mod.commit_ingest(None, str(FIXTURE), None)
    return next(x for x in outs if isinstance(x, dict) and "roll" in x)


def _args(film, chem, mins, ei, state, *, paper="ra4-glossy-v1"):
    return [
        "color",
        film,
        chem,
        mins,
        0.0,
        1.0,
        ei,
        "none",
        0.01,
        0.0,
        paper,
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
        20.0,
        1.0,
        0.0,
        0.0,
        state,
    ]


def _arr(packed):
    live = packed[0]
    val = getattr(live, "value", None)
    if val is None and isinstance(live, dict):
        val = live.get("value")
    if val is None:
        val = live
    a = np.asarray(val, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return a


def test_color_mode_portra_round_trip_and_e6():
    mod = _load_ui()
    state = _state(mod)
    state = {**state, "spot_pos": "0.5000,0.5000", "chemistry_mode": "color"}

    out400 = mod.on_film_change_and_preview(
        *_args("portra-400-spectral-v1", "c41_standard", 3.25, 400, state)
    )
    p400 = out400[3:]
    summary = p400[-1].get("summary_cache") or ""
    assert "Portra 400" in summary
    assert "C41" in summary or "C-41" in summary or "c41" in summary.lower()
    assert getattr(p400[-1].get("development"), "color_process", None) == "c41"
    assert p400[-1].get("spectral_transmittance") is not None
    assert "Zone" in (p400[-2] or "")

    out160 = mod.on_film_change_and_preview(
        *_args("portra-160-spectral-v1", "c41_standard", 3.25, 160, p400[-1])
    )
    p160 = out160[3:]
    assert "Portra 160" in (p160[-1].get("summary_cache") or "")
    assert float(np.mean(np.abs(_arr(p400) - _arr(p160)))) > 0.01

    # Stale EI/minutes in inputs must not stick — atomic resolve like B&W.
    out_back = mod.on_film_change_and_preview(
        *_args("portra-400-spectral-v1", "c41_standard", 3.25, 160, p160[-1])
    )
    p_back = out_back[3:]
    assert float(np.mean(np.abs(_arr(p400) - _arr(p_back)))) < 1e-5
    assert p400[-2] == p_back[-2]

    out_e6 = mod.on_film_change_and_preview(
        *_args("velvia-50-spectral-v1", "e6_standard", 6.0, 50, p_back[-1])
    )
    p_e6 = out_e6[3:]
    summary_e6 = p_e6[-1].get("summary_cache") or ""
    assert "Velvia" in summary_e6
    assert "E-6" in summary_e6 or "slide" in summary_e6.lower()
    assert getattr(p_e6[-1].get("development"), "color_process", None) == "e6"
