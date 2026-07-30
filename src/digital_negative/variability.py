"""Mild seed-controlled process variability — living materials, not chaos."""

from __future__ import annotations

import numpy as np


def process_micro_variation(
    density: np.ndarray,
    *,
    process_seed: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply very low-amplitude, spatially smooth density drift.

    Mimics tiny tank / agitation unevenness. Noticeable on close inspection,
    never a texture overlay. Strength 1.0 ≈ ±0.01–0.02 density in midtones.
    """
    amp = 0.014 * float(np.clip(strength, 0.0, 2.0))
    if amp <= 1e-6:
        return density

    rng = np.random.default_rng((int(process_seed) ^ 0xA5A5) & 0x7FFFFFFF)
    h, w = density.shape[:2]
    # Coarse grid, then upsample — keeps variation large-scale / soft
    gh, gw = max(4, h // 64), max(4, w // 64)
    field = rng.standard_normal((gh, gw), dtype=np.float32)
    # Separable blur via repeated box (cheap, dependency-free)
    for _ in range(2):
        field = 0.25 * (
            np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        )
    yy = (np.linspace(0, gh - 1, h)).astype(np.float32)
    xx = (np.linspace(0, gw - 1, w)).astype(np.float32)
    # Bilinear sample
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = (yy - y0)[:, None]
    wx = (xx - x0)[None, :]
    y0 = np.clip(y0, 0, gh - 1)
    x0 = np.clip(x0, 0, gw - 1)
    f00 = field[y0][:, x0]
    f01 = field[y0][:, x1]
    f10 = field[y1][:, x0]
    f11 = field[y1][:, x1]
    smooth = (f00 * (1 - wx) + f01 * wx) * (1 - wy) + (f10 * (1 - wx) + f11 * wx) * wy

    # Slightly more visible in mid densities
    mid = np.clip((density - 0.2) / 1.2, 0.0, 1.0)
    envelope = 0.35 + 0.65 * np.clip(np.sin(np.pi * mid), 0.0, 1.0)
    out = density + smooth.astype(np.float32) * amp * envelope.astype(np.float32)
    return np.maximum(out, 0.02).astype(np.float32)
