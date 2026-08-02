"""Commit Print must use the same transmittance source as Live (spectral for C-41)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.papers import load_paper_profile
from digital_negative.print_engine import print_negative


def _load_ui():
    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _scene():
    h, w = 40, 32
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.15 + 0.85 * (xx / w)
    rgb[..., 1] = 0.15 + 0.85 * (yy / h)
    rgb[..., 2] = 0.55
    rgb[8:28, 8:22] = (0.92, 0.58, 0.38)
    return DigitalNegative(image=rgb, metadata=default_metadata())


def test_print_transmittance_prefers_spectral():
    mod = _load_ui()
    dn = _scene()
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    developed = develop(
        dn, film, developer_id="c41_standard", development_minutes=3.25, commit=False
    )
    assert developed.spectral_transmittance is not None
    t = mod._print_transmittance(developed)
    assert t.shape == developed.spectral_transmittance.shape
    assert t is developed.spectral_transmittance or np.allclose(t, developed.spectral_transmittance)


def test_mono_transmittance_commit_diverges_from_spectral_live():
    """Regression context: using mono T for RA-4 was the wash-out bug."""
    dn = _scene()
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    developed = develop(
        dn, film, developer_id="c41_standard", development_minutes=3.25, commit=False
    )
    dn.metadata["print"] = {
        "cc_cyan": 47.0,
        "cc_magenta": 0.0,
        "cc_yellow": 0.0,
        "filtration": {
            "type": "color",
            "values": {"cc_cyan": 47.0, "cc_magenta": 0.0, "cc_yellow": 0.0},
        },
    }
    live = print_negative(
        developed.spectral_transmittance,
        dn,
        paper,
        base_exposure_seconds=8.0,
        commit=False,
    )
    mono = print_negative(
        developed.transmittance,
        dn,
        paper,
        base_exposure_seconds=8.0,
        commit=False,
    )
    mae = float(np.mean(np.abs(live.preview - mono.preview)))
    assert mae > 0.04


def test_commit_print_source_uses_print_transmittance_helper():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    chunk = source.split("def commit_print(")[1].split("def _unlock_stage(")[0]
    assert "t_print = _print_transmittance(development)" in chunk
    assert "development.transmittance," not in chunk
