"""Color stock identity must survive C-41 → RA-4 (process fidelity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.papers import load_paper_profile
from digital_negative.print_engine import print_negative

ROOT = Path(__file__).resolve().parents[1]


def _scene():
    h, w = 80, 60
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.15 + 0.7 * (xx / w)
    rgb[..., 1] = 0.12 + 0.65 * (yy / h)
    rgb[..., 2] = 0.25 + 0.4 * ((xx + yy) / (w + h))
    blob = np.exp(-(((yy - h / 2) / 28) ** 2 + ((xx - w / 2) / 22) ** 2))
    return np.clip(rgb + 0.35 * blob[..., None], 0, 1)


def _print_portra(film_id: str, *, seconds: float = 8.0, grain: float = 0.0):
    film = load_film_profile(ROOT / "profiles" / "films" / f"{film_id}.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    dn = DigitalNegative(image=_scene(), metadata=default_metadata())
    dn.metadata["process_seed"] = 7
    dn.metadata.setdefault("print", {}).update(
        {"cc_cyan": 0.0, "cc_magenta": 0.0, "cc_yellow": 0.0}
    )
    developed = develop(
        dn,
        film,
        developer_id="c41_standard",
        development_minutes=3.25,
        grain_strength=grain,
        commit=True,
    )
    printed = print_negative(
        developed.spectral_transmittance,
        dn,
        paper,
        base_exposure_seconds=seconds,
        commit=False,
    )
    return developed, np.asarray(printed.preview, dtype=np.float32)


def test_denser_portra_800_prints_lighter_than_160_at_same_timer():
    d160, p160 = _print_portra("portra-160-spectral-v1")
    d800, p800 = _print_portra("portra-800-spectral-v1")
    assert float(d800.dye_concentrations.mean()) > float(d160.dye_concentrations.mean()) + 0.05
    # Denser C-41 neg → less enlarger light → lighter RA-4 print at 8s.
    assert float(p800.mean()) > float(p160.mean()) + 0.03
    assert float(np.mean(np.abs(p800 - p160))) > 0.02


def test_opening_ra4_timer_darkens_print():
    _d, soft = _print_portra("portra-400-spectral-v1", seconds=4.0)
    _d, hard = _print_portra("portra-400-spectral-v1", seconds=16.0)
    assert float(hard.mean()) < float(soft.mean()) - 0.08


def test_portra_stocks_remain_distinct_after_print():
    _, p160 = _print_portra("portra-160-spectral-v1")
    _, p400 = _print_portra("portra-400-spectral-v1")
    _, p800 = _print_portra("portra-800-spectral-v1")
    assert float(np.mean(np.abs(p160 - p400))) > 0.015
    assert float(np.mean(np.abs(p400 - p800))) > 0.012
    # Usable print range — not crushed cyan sludge / not clipped white.
    for p in (p160, p400, p800):
        assert 0.20 < float(p.mean()) < 0.85
        ch = p.reshape(-1, 3).mean(0)
        assert float(ch.min()) > 0.05


def test_ra4_balance_does_not_force_channel_medians_to_same_level():
    """Full median→0.42 wipe is gone; chroma-only keeps stock density spread."""
    _, p160 = _print_portra("portra-160-spectral-v1")
    _, p800 = _print_portra("portra-800-spectral-v1")
    med160 = np.median(p160.reshape(-1, 3), axis=0)
    med800 = np.median(p800.reshape(-1, 3), axis=0)
    # Overall level must still differ (density identity).
    assert abs(float(med800.mean()) - float(med160.mean())) > 0.02
