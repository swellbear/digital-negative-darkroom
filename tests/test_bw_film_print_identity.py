"""B&W film stock identity must survive Develop → Print (process fidelity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.curves import load_film_profile
from digital_negative.papers import load_paper_profile
from digital_negative.print_engine import print_negative

ROOT = Path(__file__).resolve().parents[1]


def _scene():
    h, w = 120, 90
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.1 + 0.8 * (xx / w)
    rgb[..., 1] = 0.1 + 0.8 * (yy / h)
    rgb[..., 2] = 0.3
    blob = np.exp(-(((yy - h / 2) / 35) ** 2 + ((xx - w / 2) / 28) ** 2))
    return np.clip(rgb + 0.4 * blob[..., None], 0, 1)


def _print_film(film_id: str, *, grain: float = 0.0):
    film = load_film_profile(ROOT / "profiles" / "films" / f"{film_id}.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "fiber-glossy-v1.json")
    dn = DigitalNegative(image=_scene(), metadata=default_metadata())
    dn.metadata["process_seed"] = 7
    developed = develop(
        dn,
        film,
        developer_id="d76",
        development_minutes=7.75,
        grain_strength=grain,
        commit=True,
    )
    printed = print_negative(
        developed.transmittance,
        dn,
        paper,
        base_exposure_seconds=8.0,
        grade=2.5,
        contrast=0.0,
        commit=False,
    )
    return developed, np.asarray(printed.preview, dtype=np.float32)


def test_tri_x_denser_than_acros_and_prints_differently_at_same_timer():
    """Absolute stock density must not be percentile-normalized away."""
    d_tx, p_tx = _print_film("tri-x-400-v1")
    d_ac, p_ac = _print_film("acros-100-ii-v1")
    assert float(d_tx.density.mean()) > float(d_ac.density.mean()) + 0.05
    # Denser Tri-X passes less enlarger light → lighter print at the same 8s
    # (open the timer for Tri-X; stop down for Acros) — real darkroom physics.
    assert float(p_tx.mean()) > float(p_ac.mean()) + 0.05
    assert float(np.mean(np.abs(p_tx - p_ac))) > 0.08
    # Both remain in a usable tone range (not clipped white / crushed black).
    assert 0.25 < float(p_tx.mean()) < 0.92
    assert 0.15 < float(p_ac.mean()) < 0.85


def test_tri_x_grainier_than_acros_in_density():
    d_tx, _ = _print_film("tri-x-400-v1", grain=1.0)
    d_ac, _ = _print_film("acros-100-ii-v1", grain=1.0)
    neigh = lambda d: float(np.mean(np.abs(d[1:] - d[:-1])))
    assert neigh(d_tx.density) > neigh(d_ac.density) * 1.5


def test_opening_timer_darkens_print_with_fixed_anchor():
    film = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    dn = DigitalNegative(image=_scene(), metadata=default_metadata())
    developed = develop(dn, film, grain_strength=0.0, commit=True)
    soft = print_negative(
        developed.transmittance, dn, paper, base_exposure_seconds=4.0, grade=2.0, commit=False
    )
    hard = print_negative(
        developed.transmittance, dn, paper, base_exposure_seconds=16.0, grade=2.0, commit=False
    )
    assert float(hard.preview.mean()) < float(soft.preview.mean())
