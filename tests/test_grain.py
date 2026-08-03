"""Grain must survive Commit / preview downscale (Tri-X-class stocks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digital_negative.curves import load_film_profile
from digital_negative.grain import apply_grain

ROOT = Path(__file__).resolve().parents[1]


def _local_std(gray: np.ndarray, k: int = 5) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    g = np.asarray(gray, dtype=np.float32)
    win = sliding_window_view(g, (k, k))
    return float(win.reshape(win.shape[0], win.shape[1], -1).std(axis=-1).mean())


def test_tri_x_grain_survives_lanczos_half_scale():
    from PIL import Image

    film = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    h, w = 400, 300
    # Flat midtone density — envelope peaks here.
    density = np.full((h, w), film.base_plus_fog + 0.7, dtype=np.float32)
    grained = apply_grain(density, profile=film, grain_strength=1.0, process_seed=42)
    assert float(np.std(grained - density)) > 0.01

    # Density → crude display proxy, then Lanczos half (like commit preview).
    disp = np.clip(1.0 - (grained - film.base_plus_fog) / 1.8, 0, 1)
    u8 = (disp * 255.0 + 0.5).astype(np.uint8)
    half = np.asarray(
        Image.fromarray(u8).resize((w // 2, h // 2), resample=Image.Resampling.LANCZOS)
    )
    # Spatially correlated Tri-X grain should still read after half-scale.
    assert _local_std(half) > 3.5


def test_fine_grain_stock_quieter_than_tri_x():
    trix = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    delta = load_film_profile(ROOT / "profiles" / "films" / "delta-100-v1.json")
    density = np.full((256, 256), 0.9, dtype=np.float32)
    g_tri = apply_grain(density, profile=trix, grain_strength=1.0, process_seed=7)
    g_del = apply_grain(density, profile=delta, grain_strength=1.0, process_seed=7)
    assert float(np.std(g_tri - density)) > float(np.std(g_del - density))
