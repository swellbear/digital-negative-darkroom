"""Characteristic curve loading and interpolation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class FilmProfile:
    """Film stock profile with a digitized characteristic curve."""

    id: str
    name: str
    type: str
    version: str
    iso: int
    base_plus_fog: float
    log_exposure: np.ndarray
    density: np.ndarray
    source: dict[str, Any]
    defaults: dict[str, Any]
    raw: dict[str, Any]

    def density_from_log_exposure(self, log_e: np.ndarray) -> np.ndarray:
        """Interpolate density for relative log exposure values."""
        interpolator = PchipInterpolator(self.log_exposure, self.density, extrapolate=False)
        log_min = float(self.log_exposure[0])
        log_max = float(self.log_exposure[-1])
        clipped = np.clip(log_e, log_min, log_max)
        dens = interpolator(clipped)
        # Flat extrapolation outside the measured range
        dens = np.where(log_e < log_min, self.density[0], dens)
        dens = np.where(log_e > log_max, self.density[-1], dens)
        return dens.astype(np.float32)


def load_film_profile(path: str | Path) -> FilmProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = np.asarray(data["curve"]["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid curve points in {path}")

    # Ensure strictly increasing log exposure for interpolation
    order = np.argsort(points[:, 0])
    points = points[order]
    if np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError(f"Curve log-exposure values must be strictly increasing: {path}")

    return FilmProfile(
        id=data["id"],
        name=data["name"],
        type=data["type"],
        version=data["version"],
        iso=int(data["iso"]),
        base_plus_fog=float(data["curve"].get("base_plus_fog", points[0, 1])),
        log_exposure=points[:, 0],
        density=points[:, 1],
        source=data.get("source", {}),
        defaults=data.get("defaults", {}),
        raw=data,
    )


def modify_curve(
    profile: FilmProfile,
    *,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
) -> FilmProfile:
    """Apply Level-3 development modifiers to a base characteristic curve.

    relative_time:
        1.0 = normal; >1 push (more density / contrast); <1 pull.
    contrast_modifier:
        Added to the effective slope around midtones (-1..+1 typical).
    """
    log_e = profile.log_exposure.copy()
    dens = profile.density.copy()
    fog = profile.base_plus_fog

    # Relative development: scale density above fog (push/pull)
    # Mild non-linearity keeps toe/shoulder character.
    scale = float(np.clip(relative_time, 0.4, 2.5))
    dens = fog + (dens - fog) * (0.65 + 0.35 * scale) * (scale**0.35)

    # Contrast: pivot around mid-curve density and stretch
    pivot = float(np.interp(0.5, np.linspace(0, 1, len(dens)), dens))
    contrast_scale = 1.0 + 0.45 * float(np.clip(contrast_modifier, -1.5, 1.5))
    dens = pivot + (dens - pivot) * contrast_scale
    dens = np.maximum(dens, fog * 0.98)

    return FilmProfile(
        id=profile.id,
        name=profile.name,
        type=profile.type,
        version=profile.version,
        iso=profile.iso,
        base_plus_fog=fog,
        log_exposure=log_e,
        density=dens.astype(np.float64),
        source=profile.source,
        defaults=profile.defaults,
        raw=profile.raw,
    )
