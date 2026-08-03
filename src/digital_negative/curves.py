"""Characteristic curve loading, curve-family interpolation, and development morphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .chemistry import DEVELOPER_STYLES, resolve_style


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


def _points_to_arrays(points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("curve points must be Nx2")
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    if np.any(np.diff(pts[:, 0]) <= 0):
        raise ValueError("curve log-exposure must be strictly increasing")
    return pts[:, 0], pts[:, 1]


def _resample_density(log_e: np.ndarray, dens: np.ndarray, grid: np.ndarray) -> np.ndarray:
    interpolator = PchipInterpolator(log_e, dens, extrapolate=False)
    clipped = np.clip(grid, float(log_e[0]), float(log_e[-1]))
    out = interpolator(clipped)
    out = np.where(grid < log_e[0], dens[0], out)
    out = np.where(grid > log_e[-1], dens[-1], out)
    return out.astype(np.float64)


def interpolate_curve_family(
    family: list[dict[str, Any]], minutes: float
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Interpolate a digitized chemistry×time curve family at ``minutes``.

    Each family member: ``{minutes, points: [[logE, D], ...], base_plus_fog?}``.
    Returns ``(log_exposure_grid, density, base_plus_fog, meta)``.
    """
    if not family:
        raise ValueError("empty curve family")
    members = sorted(family, key=lambda m: float(m["minutes"]))
    times = np.asarray([float(m["minutes"]) for m in members], dtype=np.float64)
    # Keep the requested time for clamp metadata — clipping first made
    # clamp_low / clamp_high unreachable (always reported as "exact").
    minutes_req = float(minutes)
    t_lo = float(times[0])
    t_hi = float(times[-1])

    # Common log-E grid = sorted union of all member abscissae
    all_logs: list[float] = []
    parsed: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for m in members:
        log_e, dens = _points_to_arrays(m["points"])
        fog = float(m.get("base_plus_fog", dens[0]))
        parsed.append((float(m["minutes"]), log_e, dens, fog))
        all_logs.extend(log_e.tolist())
    grid = np.asarray(sorted(set(round(x, 6) for x in all_logs)), dtype=np.float64)

    dens_rows = []
    fog_vals = []
    for _t, log_e, dens, fog in parsed:
        dens_rows.append(_resample_density(log_e, dens, grid))
        fog_vals.append(fog)
    dens_mat = np.vstack(dens_rows)  # (n_times, n_log)
    fog_arr = np.asarray(fog_vals, dtype=np.float64)

    if minutes_req < t_lo:
        dens_out = dens_mat[0]
        fog_out = float(fog_arr[0])
        mode = "clamp_low"
        minutes_used = t_lo
    elif minutes_req > t_hi:
        dens_out = dens_mat[-1]
        fog_out = float(fog_arr[-1])
        mode = "clamp_high"
        minutes_used = t_hi
    elif any(np.isclose(times, minutes_req)):
        idx = int(np.where(np.isclose(times, minutes_req))[0][0])
        dens_out = dens_mat[idx]
        fog_out = float(fog_arr[idx])
        mode = "exact"
        minutes_used = float(times[idx])
    else:
        # PCHIP across time at each log-E sample
        dens_out = np.empty(grid.shape[0], dtype=np.float64)
        for j in range(grid.shape[0]):
            dens_out[j] = float(PchipInterpolator(times, dens_mat[:, j])(minutes_req))
        fog_out = float(PchipInterpolator(times, fog_arr)(minutes_req))
        mode = "interpolated"
        minutes_used = minutes_req

    meta = {
        "curve_source": "family",
        "family_mode": mode,
        "family_times": [float(t) for t in times],
        "family_minutes": float(minutes_used),
        "family_minutes_requested": minutes_req,
    }
    return grid, dens_out, fog_out, meta


def has_curve_family(chem: dict[str, Any] | None) -> bool:
    if not chem:
        return False
    family = chem.get("curve_family")
    return isinstance(family, list) and len(family) >= 1


def modify_curve(
    profile: FilmProfile,
    *,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
    developer_id: str = "standard",
    development_minutes: float | None = None,
) -> FilmProfile:
    """Apply development to a base characteristic curve.

    Prefer a digitized ``curve_family`` on the selected chemistry when
    ``development_minutes`` is provided — true chemistry×time D–logE data with
    interpolation between published times. Otherwise fall back to the Level-3
    relative-time morph of the film's single base curve.
    """
    style, chem = resolve_style(profile, developer_id)
    family_meta: dict[str, Any] = {"curve_source": "morph"}

    if has_curve_family(chem) and development_minutes is not None:
        log_e, dens, fog, family_meta = interpolate_curve_family(
            chem["curve_family"], float(development_minutes)
        )
        # Curve family already encodes chemistry + time. Only apply the user's
        # N± contrast aim as a light straight-line tweak (not the morph push/pull).
        t = np.linspace(0.0, 1.0, len(dens))
        pivot_idx = int(0.42 * (len(dens) - 1))
        pivot = float(dens[pivot_idx])
        contrast_amt = float(np.clip(contrast_modifier, -1.5, 1.5))
        if abs(contrast_amt) > 1e-6:
            stretch = 1.0 + 0.45 * contrast_amt
            highlight_extra = 1.0 + 0.15 * max(contrast_amt, 0.0) * (t**1.15)
            toe_protect = 1.0 - 0.10 * max(-contrast_amt, 0.0) * ((1.0 - t) ** 1.4)
            dens = pivot + (dens - pivot) * stretch * highlight_extra * toe_protect
            dens = np.maximum(dens, fog * 0.98)
        raw = dict(profile.raw)
        raw = {**raw, "_last_curve_meta": family_meta}
        return FilmProfile(
            id=profile.id,
            name=profile.name,
            type=profile.type,
            version=profile.version,
            iso=profile.iso,
            base_plus_fog=float(fog),
            log_exposure=log_e.astype(np.float64),
            density=dens.astype(np.float64),
            source=profile.source,
            defaults=profile.defaults,
            raw=raw,
        )

    # --- Morph path (single base curve + relative time + character) ---
    log_e = profile.log_exposure.copy()
    dens = profile.density.copy()
    fog = max(0.02, profile.base_plus_fog + float(style["fog_lift"]))

    scale = float(np.clip(relative_time, 0.45, 2.2))
    above = dens - profile.base_plus_fog
    t = np.linspace(0.0, 1.0, len(dens))
    push = max(scale - 1.0, 0.0)
    pull = max(1.0 - scale, 0.0)
    local = (
        1.0
        + 0.62 * push * (0.28 + 0.72 * t)
        - 0.48 * pull * (0.40 + 0.60 * t)
    )
    dens = fog + above * local * float(style["density_bias"]) * (0.82 + 0.18 * scale)

    toe_k = float(style["toe_softness"])
    sh_k = float(style["shoulder_roll"])
    if abs(toe_k) > 1e-6:
        dens = dens + toe_k * (1.0 - t) ** 2 * (dens - fog)
    if abs(sh_k) > 1e-6:
        dens = dens - sh_k * (t**2) * (dens - fog)

    pivot_idx = int(0.42 * (len(dens) - 1))
    pivot = float(dens[pivot_idx])
    contrast_amt = float(np.clip(contrast_modifier + float(style["contrast_bias"]), -1.5, 1.5))
    stretch = 1.0 + 0.62 * contrast_amt
    highlight_extra = 1.0 + 0.22 * max(contrast_amt, 0.0) * (t**1.15)
    toe_protect = 1.0 - 0.12 * max(-contrast_amt, 0.0) * ((1.0 - t) ** 1.4)
    dens = pivot + (dens - pivot) * stretch * highlight_extra * toe_protect
    dens = np.maximum(dens, fog * 0.98)

    raw = dict(profile.raw)
    raw = {**raw, "_last_curve_meta": family_meta}
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
        raw=raw,
    )
