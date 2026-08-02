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


def _vertical_mullion_frame(h: int = 280, w: int = 360) -> np.ndarray:
    """Building-like vertical window mullions — the case Auto used to miss."""
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    bands = np.where((xx * 22).astype(int) % 2 == 0, 0.9, 0.18).astype(np.float32)
    img = np.broadcast_to(bands, (h, w)).copy()
    # Soft horizontal sill near the bottom (weaker than mullions).
    img[int(h * 0.72) : int(h * 0.76), :] = 0.55
    rng = np.random.default_rng(2)
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)


def _soft_horizon_scene(h: int = 320, w: int = 480) -> np.ndarray:
    """Realistic soft sky/ground split — the case that used to snap to 0°."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    # Soft horizon around mid-frame rather than a hard stripe.
    sky = 0.78 - 0.08 * yy
    ground = 0.28 + 0.12 * yy
    blend = 1.0 / (1.0 + np.exp(-(yy - 0.58) * 28.0))
    img = (sky * (1.0 - blend) + ground * blend).astype(np.float32)
    img = np.broadcast_to(img, (h, w)).copy()
    # Faint distant verticals (trees / poles) — weaker than the horizon.
    for x in (0.22, 0.48, 0.71):
        c = int(round(x * (w - 1)))
        img[int(h * 0.45) : int(h * 0.78), max(0, c - 1) : min(w, c + 2)] *= 0.55
    rng = np.random.default_rng(7)
    return np.clip(img + rng.normal(0, 0.018, img.shape), 0, 1).astype(np.float32)


def test_estimate_straighten_recovers_tilt():
    level = _striped_frame()
    # Tilt the content so a +2.5° CW straighten levels it again.
    tilted = straighten_image(level, -2.5)
    est = estimate_straighten_degrees(tilted, max_degrees=8.0, step=0.25)
    assert abs(est - 2.5) <= 0.75


def test_estimate_straighten_recovers_vertical_mullion_tilt():
    level = _vertical_mullion_frame()
    tilted = straighten_image(level, -3.0)
    est = estimate_straighten_degrees(tilted, max_degrees=8.0, step=0.25)
    assert abs(est - 3.0) <= 0.75


def test_estimate_straighten_recovers_soft_horizon_tilt():
    level = _soft_horizon_scene()
    tilted = straighten_image(level, -2.25)
    est = estimate_straighten_degrees(tilted, max_degrees=8.0, step=0.25)
    assert abs(est - 2.25) <= 1.0


def test_level_image_stays_near_zero():
    level = _striped_frame()
    est = estimate_straighten_degrees(level, max_degrees=8.0, step=0.25)
    assert abs(est) <= 0.5


def test_level_soft_horizon_stays_near_zero():
    level = _soft_horizon_scene()
    est = estimate_straighten_degrees(level, max_degrees=8.0, step=0.25)
    assert abs(est) <= 0.75
