"""RA-4 Border % must be a white easel, not a black crop of the picture."""

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
    img = np.linspace(0.08, 1.1, 40 * 56, dtype=np.float32).reshape(40, 56)
    return np.stack([img, img * 0.95, img * 0.85], axis=-1)


def test_color_border_is_white_easel_not_black_inset():
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    dn = DigitalNegative(image=_scene(), metadata=default_metadata())
    dn.metadata.setdefault("print", {}).update(
        {"cc_cyan": 0.0, "cc_magenta": 0.0, "cc_yellow": 0.0}
    )
    developed = develop(
        dn,
        film,
        developer_id="c41_standard",
        development_minutes=3.25,
        grain_strength=0.0,
        commit=False,
    )
    bare = print_negative(
        developed.spectral_transmittance,
        dn,
        paper,
        base_exposure_seconds=8.0,
        border_frac=0.0,
        commit=False,
    )
    framed = print_negative(
        developed.spectral_transmittance,
        dn,
        paper,
        base_exposure_seconds=8.0,
        border_frac=0.10,
        commit=False,
    )
    # Expands the sheet (easel) — does not keep HxW and paint black margins.
    assert framed.preview.shape[0] > bare.preview.shape[0]
    assert framed.preview.shape[1] > bare.preview.shape[1]
    edge = float(np.mean(framed.preview[0:2, 0:2]))
    assert edge > 0.9  # white unexposed paper, not black
    # Picture content still present in the well (not crushed to a tiny inset).
    h, w = bare.preview.shape[:2]
    bh = (framed.preview.shape[0] - h) // 2
    bw = (framed.preview.shape[1] - w) // 2
    well = framed.preview[bh : bh + h, bw : bw + w]
    assert float(np.mean(np.abs(well - bare.preview))) < 1e-5
