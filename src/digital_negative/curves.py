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


# Tuned to read like tank chemistry, not generic contrast sliders:
# - High Definition ≈ fine-grain / solvent developer (cleaner, slightly leaner)
# - High Energy ≈ speed-enhancing / vigorous developer (punchier, grainier, more fog)
DEVELOPER_STYLES = {
    "standard": {
        "name": "Standard",
        "contrast_bias": 0.0,
        "density_bias": 1.0,
        "grain_bias": 1.0,
        "fog_lift": 0.0,
        "toe_softness": 0.0,
        "shoulder_roll": 0.0,
    },
    "high_definition": {
        "name": "High Definition",
        "contrast_bias": 0.08,
        "density_bias": 0.92,
        "grain_bias": 0.55,
        "fog_lift": -0.015,
        "toe_softness": 0.08,
        "shoulder_roll": 0.05,
    },
    "high_energy": {
        "name": "High Energy",
        "contrast_bias": 0.42,
        "density_bias": 1.18,
        "grain_bias": 1.45,
        "fog_lift": 0.025,
        "toe_softness": -0.06,
        "shoulder_roll": -0.04,
    },
}


def modify_curve(
    profile: FilmProfile,
    *,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
    developer_id: str = "standard",
) -> FilmProfile:
    """Apply Level-3 development modifiers to a base characteristic curve.

    relative_time:
        1.0 = N / normal. >1 push (CI up, more density in highlights).
        <1 pull (flatter, leaner).
    contrast_modifier:
        Extra straight-line stretch around midtones (−1…+1), like aiming N− / N+.
    """
    style = DEVELOPER_STYLES.get(developer_id, DEVELOPER_STYLES["standard"])
    log_e = profile.log_exposure.copy()
    dens = profile.density.copy()
    fog = max(0.02, profile.base_plus_fog + float(style["fog_lift"]))

    # --- Relative development (push / pull) ---
    # Push raises average gradient and builds highlight density faster than the toe.
    # Pull does the reverse — closer to real tank timing than a uniform scale.
    scale = float(np.clip(relative_time, 0.45, 2.2))
    above = dens - profile.base_plus_fog
    # Position along the curve 0..1
    t = np.linspace(0.0, 1.0, len(dens))
    # Push weights the upper scale more; pull flattens it
    push = max(scale - 1.0, 0.0)
    pull = max(1.0 - scale, 0.0)
    local = 1.0 + 0.55 * push * (0.35 + 0.65 * t) - 0.50 * pull * (0.55 + 0.45 * (1.0 - t))
    dens = fog + above * local * float(style["density_bias"]) * (0.78 + 0.22 * scale)

    # Developer toe / shoulder character
    toe_k = float(style["toe_softness"])
    sh_k = float(style["shoulder_roll"])
    if abs(toe_k) > 1e-6:
        dens = dens + toe_k * (1.0 - t) ** 2 * (dens - fog)
    if abs(sh_k) > 1e-6:
        dens = dens - sh_k * (t**2) * (dens - fog)

    # --- Contrast (N− / N+) around a mid-curve pivot ---
    pivot_idx = int(0.45 * (len(dens) - 1))
    pivot = float(dens[pivot_idx])
    contrast_amt = float(np.clip(contrast_modifier + float(style["contrast_bias"]), -1.5, 1.5))
    # Asymmetric: +contrast steepens highlights a bit more (classic N+ feel)
    stretch = 1.0 + 0.55 * contrast_amt
    highlight_extra = 1.0 + 0.20 * max(contrast_amt, 0.0) * t
    dens = pivot + (dens - pivot) * stretch * highlight_extra
    dens = np.maximum(dens, fog * 0.98)

    return FilmProfile(
        id=profile.id,
        name=profile.name,
        type=profile.type,
        version=profile.version,
        iso=profile.iso,
        base_plus_fog=float(fog),
        log_exposure=log_e,
        density=dens.astype(np.float64),
        source=profile.source,
        defaults=profile.defaults,
        raw=profile.raw,
    )
