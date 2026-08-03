"""Runtime: film swap must change center spot when meter re-samples print_draft."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from digital_negative.analysis import spot_at
from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative, default_metadata
from digital_negative.papers import load_paper_profile
from digital_negative.print_engine import print_negative

ROOT = Path(__file__).resolve().parents[1]


def _print_spot(film_id: str, chem: str, minutes: float):
    img = np.asarray(
        Image.open(ROOT / "tests/fixtures/scene_linear_srgb.png").convert("RGB"),
        dtype=np.float32,
    ) / 255.0
    img = img[::3, ::3]
    film = load_film_profile(ROOT / "profiles" / "films" / f"{film_id}.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "fiber-glossy-v1.json")
    dn = DigitalNegative(image=img, metadata=default_metadata())
    dn.metadata["process_seed"] = 7
    developed = develop(
        dn,
        film,
        developer_id=chem,
        development_minutes=minutes,
        grain_strength=1.0,
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
    return spot_at(np.asarray(printed.reflectance), np.asarray(printed.print_density), 0.5, 0.5)


def test_center_spot_differs_across_user_compared_stocks():
    a = _print_spot("delta-400-v1", "id11_stock", 9.0)
    b = _print_spot("tri-x-400-v1", "d76", 7.75)
    c = _print_spot("hp5-plus-v1", "ilfotec_hc_1_31", 6.5)
    assert abs(a["zone"] - b["zone"]) > 0.4
    assert abs(a["density"] - b["density"]) > 0.15
    assert abs(b["zone"] - c["zone"]) > 0.3
    # Sticky-meter failure mode: identical Zone V (5.3) / D 0.64 across swaps.
    readings = {(round(s["zone"], 1), round(s["density"], 2)) for s in (a, b, c)}
    assert len(readings) == 3
