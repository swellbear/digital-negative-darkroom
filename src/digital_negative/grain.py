"""Controlled grain / micro-variation using the process seed."""

from __future__ import annotations

import numpy as np

from .curves import FilmProfile


def _correlated_noise(shape: tuple[int, ...], *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-ish Gaussian noise with spatial correlation (silver-grain clumps).

    Per-pixel white noise vanishes under Lanczos preview/download downscale;
    real Tri-X / HP5 clumps survive viewing size. ``sigma`` is in pixels.
    """
    noise = rng.standard_normal(shape, dtype=np.float32)
    if sigma >= 0.45:
        from scipy.ndimage import gaussian_filter

        # Slightly less blur on trailing spectral/dye axis if present.
        if len(shape) >= 3:
            axes_sigma = tuple(float(sigma) for _ in shape[:-1]) + (0.0,)
        else:
            axes_sigma = float(sigma)
        noise = gaussian_filter(noise, sigma=axes_sigma, mode="reflect").astype(np.float32)
    std = float(np.std(noise))
    if std > 1e-8:
        noise /= std
    return noise


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

    Grain is spatially correlated so Tri-X-class stocks still read as
    grainy after Commit Print / interactive-res preview downscale (white
    single-pixel noise was being erased by Lanczos).
    """
    strength = float(np.clip(grain_strength, 0.0, 3.0))
    if strength <= 1e-6:
        return density

    film_scale = float(profile.raw.get("grain_scale", 1.0))
    iso_scale = np.sqrt(max(profile.iso, 25) / 400.0)
    amplitude = 0.038 * strength * film_scale * iso_scale
    # Clump size: fine-grain stocks stay tight; Tri-X / 800-class open up.
    sigma = float(np.clip(0.45 + 0.9 * film_scale * iso_scale * (0.75 + 0.25 * strength), 0.45, 2.8))

    rng = np.random.default_rng(int(process_seed) & 0x7FFFFFFF)
    noise = _correlated_noise(density.shape, sigma=sigma, rng=rng)

    # Soft envelope: peak around mid densities
    fog = profile.base_plus_fog
    t = np.clip((density - fog) / 1.4, 0.0, 1.0)
    # Clip before the fractional power — sin(pi) can be a tiny negative float
    envelope = np.power(np.clip(np.sin(np.pi * t), 0.0, 1.0), 1.2)
    envelope = 0.25 + 0.75 * envelope

    grained = density + noise * amplitude * envelope.astype(np.float32)
    return np.maximum(grained, fog * 0.9).astype(np.float32)
