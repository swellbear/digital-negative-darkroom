"""Controlled grain / micro-variation using the process seed."""

from __future__ import annotations

import numpy as np

from .curves import FilmProfile


def apply_grain(
    density: np.ndarray,
    *,
    profile: FilmProfile,
    grain_strength: float,
    process_seed: int,
) -> np.ndarray:
    """Add density-domain grain scaled by film and local tone.

    Midtones receive the most visible grain; deep shadows and dense
    highlights are quieter — closer to real negative reading.
    """
    strength = float(np.clip(grain_strength, 0.0, 3.0))
    if strength <= 1e-6:
        return density

    film_scale = float(profile.raw.get("grain_scale", 1.0))
    iso_scale = np.sqrt(max(profile.iso, 25) / 400.0)
    amplitude = 0.035 * strength * film_scale * iso_scale

    rng = np.random.default_rng(int(process_seed) & 0x7FFFFFFF)
    noise = rng.standard_normal(density.shape, dtype=np.float32)

    # Soft envelope: peak around mid densities
    fog = profile.base_plus_fog
    t = np.clip((density - fog) / 1.4, 0.0, 1.0)
    # Clip before the fractional power — sin(pi) can be a tiny negative float
    envelope = np.power(np.clip(np.sin(np.pi * t), 0.0, 1.0), 1.2)
    envelope = 0.25 + 0.75 * envelope

    grained = density + noise * amplitude * envelope.astype(np.float32)
    return np.maximum(grained, fog * 0.9).astype(np.float32)
