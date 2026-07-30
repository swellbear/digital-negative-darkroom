"""Split-grade, test strips, flash, dry-down, tone, borders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.print_engine import (
    apply_border,
    apply_dry_down,
    apply_test_strip_bands,
    apply_tone,
    print_negative,
)


def _ready():
    dn = ingest_path(ROOT / "tests" / "fixtures" / "scene_linear_srgb.png")
    film = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    dev = develop(
        dn, film, developer_id="id11_stock", development_minutes=6.5, commit=False
    )
    return dn, paper, dev.transmittance


def test_split_grade_differs_from_single():
    dn, paper, t = _ready()
    single = print_negative(t, dn, paper, base_exposure_seconds=8, grade=2.5, commit=False)
    split = print_negative(
        t, dn, paper, base_exposure_seconds=8, grade=2.5, commit=False,
        split_grade=True, soft_grade=0.0, hard_grade=5.0,
        soft_exposure_seconds=4.5, hard_exposure_seconds=3.5,
    )
    assert float(np.mean(np.abs(single.reflectance - split.reflectance))) > 1e-4


def test_test_strips_vary_across_bands():
    light = np.ones((40, 100), dtype=np.float32)
    banded = apply_test_strip_bands(light, bands=5, stop_step=0.5)
    means = [float(banded[:, i * 20 : (i + 1) * 20].mean()) for i in range(5)]
    assert max(means) / min(means) > 1.5


def test_flash_dry_tone_border():
    dn, paper, t = _ready()
    base = print_negative(t, dn, paper, base_exposure_seconds=8, grade=2.5, commit=False)
    flashed = print_negative(
        t, dn, paper, base_exposure_seconds=8, grade=2.5, flash_stops=0.5, commit=False
    )
    # Flash adds exposure → denser / darker print (lower reflectance)
    assert float(flashed.reflectance.mean()) < float(base.reflectance.mean())

    dried = apply_dry_down(base.reflectance, 10.0)
    assert float(dried.mean()) < float(base.reflectance.mean())

    sepia = apply_tone(base.preview, "sepia")
    assert sepia.ndim == 3 and sepia.shape[-1] == 3
    selenium = apply_tone(base.preview, "selenium")
    assert selenium[..., 2].mean() >= sepia[..., 2].mean() * 0.9

    bordered = apply_border(base.preview, 0.05)
    assert bordered.shape[0] > base.preview.shape[0]
    assert float(bordered[0, 0]) > 0.9
