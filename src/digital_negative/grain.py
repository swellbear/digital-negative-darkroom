"""Controlled grain / micro-variation using the process seed."""

from __future__ import annotations

import numpy as np

from .curves import FilmProfile


def _film_noise(
    shape: tuple[int, ...],
    *,
    film_scale: float,
    iso_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mostly fine grain, plus a *small* correlated component for clump structure.

    Important: after blurring, do **not** re-inflate coarse noise to unit
    variance — that made Tri-X look like oatmeal and drowned curve character.
    Fine grain stays dominant (datasheet-faithful); a light clump mix keeps a
    little structure after interactive-res / download downscale.
    """
    fine = rng.standard_normal(shape, dtype=np.float32)
    # Mild clump size from stock; Tri-X ~0.7px, Delta 100 stays tighter.
    sigma = float(np.clip(0.35 + 0.45 * film_scale * iso_scale, 0.35, 1.15))
    clump_w = float(np.clip(0.12 + 0.18 * film_scale * iso_scale, 0.08, 0.32))
    if sigma >= 0.4 and clump_w > 0.05:
        from scipy.ndimage import gaussian_filter

        coarse = rng.standard_normal(shape, dtype=np.float32)
        if len(shape) >= 3:
            axes_sigma = tuple(float(sigma) for _ in shape[:-1]) + (0.0,)
        else:
            axes_sigma = float(sigma)
        coarse = gaussian_filter(coarse, sigma=axes_sigma, mode="reflect").astype(np.float32)
        cstd = float(np.std(coarse))
        if cstd > 1e-8:
            coarse /= cstd
        noise = fine + clump_w * coarse
    else:
        noise = fine
    std = float(np.std(noise))
    if std > 1e-8:
        noise /= std
    return noise.astype(np.float32)


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

    Amplitude follows ``grain_scale`` / ISO from the film profile (Tri-X >
    HP5 > Delta 100). Structure is mostly fine with a light clump mix so
    Commit Print preview does not erase all grain, without overpowering
    the characteristic curve.
    """
    strength = float(np.clip(grain_strength, 0.0, 3.0))
    if strength <= 1e-6:
        return density

    film_scale = float(profile.raw.get("grain_scale", 1.0))
    iso_scale = float(np.sqrt(max(profile.iso, 25) / 400.0))
    # Same calibration as the pre-AI-enlarge datasheet path.
    amplitude = 0.035 * strength * film_scale * iso_scale

    rng = np.random.default_rng(int(process_seed) & 0x7FFFFFFF)
    noise = _film_noise(
        density.shape, film_scale=film_scale, iso_scale=iso_scale, rng=rng
    )

    # Soft envelope: peak around mid densities
    fog = profile.base_plus_fog
    t = np.clip((density - fog) / 1.4, 0.0, 1.0)
    # Clip before the fractional power — sin(pi) can be a tiny negative float
    envelope = np.power(np.clip(np.sin(np.pi * t), 0.0, 1.0), 1.2)
    envelope = 0.25 + 0.75 * envelope

    grained = density + noise * amplitude * envelope.astype(np.float32)
    return np.maximum(grained, fog * 0.9).astype(np.float32)
