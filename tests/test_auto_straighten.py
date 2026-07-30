"""Auto-straighten horizon estimate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.auto_crop import estimate_straighten_degrees
from digital_negative.display import straighten_image


def _striped_frame(h: int = 240, w: int = 320) -> np.ndarray:
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bands = np.where((yy * 18).astype(int) % 2 == 0, 0.92, 0.12).astype(np.float32)
    img = np.broadcast_to(bands, (h, w)).copy()
    rng = np.random.default_rng(1)
    return np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1).astype(np.float32)


def test_estimate_straighten_recovers_tilt():
    level = _striped_frame()
    # Tilt the content so a +2.5° CW straighten levels it again.
    tilted = straighten_image(level, -2.5)
    est = estimate_straighten_degrees(tilted, max_degrees=8.0, step=0.25)
    assert abs(est - 2.5) <= 0.75


def test_level_image_stays_near_zero():
    level = _striped_frame()
    est = estimate_straighten_degrees(level, max_degrees=8.0, step=0.25)
    assert abs(est) <= 0.5
