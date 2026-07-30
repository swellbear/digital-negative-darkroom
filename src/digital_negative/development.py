"""Development engine: linear scene → log exposure → density → viewable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import DEVELOPER_STYLES, FilmProfile, modify_curve
from .digital_negative import DigitalNegative
from .grain import apply_grain
from .variability import process_micro_variation


@dataclass
class DevelopmentResult:
    density: np.ndarray
    transmittance: np.ndarray
    positive_preview: np.ndarray
    log_exposure: np.ndarray
    profile: FilmProfile


def linear_to_relative_log_exposure(
    linear: np.ndarray,
    *,
    mid_log_e: float = 2.2,
    mid_scene: float | None = None,
) -> np.ndarray:
    """Map near-linear scene values onto a film relative log-E axis."""
    eps = 1e-6
    positive = linear[linear > eps]
    if mid_scene is None:
        mid_scene = float(np.median(positive)) if positive.size else 0.18
    mid_scene = max(mid_scene, eps)
    log_e = mid_log_e + np.log10(np.maximum(linear, eps) / mid_scene)
    return log_e.astype(np.float32)


def density_to_transmittance(density: np.ndarray) -> np.ndarray:
    return np.power(10.0, -density).astype(np.float32)


def transmittance_to_positive_preview(
    transmittance: np.ndarray,
    *,
    fog: float,
    d_max: float,
) -> np.ndarray:
    """Contact-style positive from negative transmittance (inspection aid)."""
    density = -np.log10(np.maximum(transmittance, 1e-6))
    span = max(d_max - fog, 1e-3)
    positive = np.clip((density - fog) / span, 0.0, 1.0)
    positive = np.power(positive, 0.90)
    return positive.astype(np.float32)


def develop(
    dn: DigitalNegative,
    profile: FilmProfile,
    *,
    relative_time: float | None = None,
    contrast_modifier: float | None = None,
    grain_strength: float | None = None,
    developer_id: str | None = None,
    mid_log_e: float = 2.2,
    process_variation: float = 1.0,
) -> DevelopmentResult:
    """Apply film characteristic curve development to a Digital Negative."""
    dev = dn.metadata.setdefault("development", {})
    rel = float(relative_time if relative_time is not None else dev.get("relative_time", 1.0))
    contrast = float(
        contrast_modifier if contrast_modifier is not None else dev.get("contrast_modifier", 0.0)
    )
    grain = float(
        grain_strength
        if grain_strength is not None
        else dev.get("grain_strength", profile.defaults.get("grain_strength", 1.0))
    )
    developer = str(
        developer_id if developer_id is not None else dev.get("developer_id", "standard")
    )
    if developer not in DEVELOPER_STYLES:
        developer = "standard"
    style = DEVELOPER_STYLES[developer]
    # Push slightly increases perceived grain; pull reduces it
    grain_eff = grain * float(style["grain_bias"]) * (0.85 + 0.25 * rel)

    working_profile = modify_curve(
        profile,
        relative_time=rel,
        contrast_modifier=contrast,
        developer_id=developer,
    )
    luminance = dn.to_luminance()
    log_e = linear_to_relative_log_exposure(luminance, mid_log_e=mid_log_e)
    density = working_profile.density_from_log_exposure(log_e)
    seed = int(dn.metadata.get("process_seed", 0))
    density = process_micro_variation(density, process_seed=seed, strength=process_variation)
    density = apply_grain(
        density,
        profile=working_profile,
        grain_strength=grain_eff,
        process_seed=seed,
    )
    transmittance = density_to_transmittance(density)
    positive = transmittance_to_positive_preview(
        transmittance,
        fog=working_profile.base_plus_fog,
        d_max=max(1.55, float(np.percentile(density, 99.5))),
    )

    dn.metadata["film_profile"] = {
        "id": profile.id,
        "name": profile.name,
        "type": profile.type,
        "version": profile.version,
        "iso": profile.iso,
    }
    user_grain = float(
        grain_strength
        if grain_strength is not None
        else dev.get("grain_strength", profile.defaults.get("grain_strength", 1.0))
    )
    dn.metadata["development"].update(
        {
            "enabled": True,
            "developer_id": developer,
            "developer_name": style["name"],
            "relative_time": rel,
            "contrast_modifier": contrast,
            "grain_strength": user_grain,
            "process_variation": float(process_variation),
        }
    )
    stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
    if "development" not in stages:
        stages.append("development")
    dn.metadata["ui_state"]["current_stage"] = "development"
    dn.touch()
    dn.metadata.setdefault("history", []).append(
        {
            "op": "develop",
            "film_profile_id": profile.id,
            "developer_id": developer,
            "relative_time": rel,
            "contrast_modifier": contrast,
            "grain_strength": user_grain,
            "process_variation": float(process_variation),
        }
    )

    return DevelopmentResult(
        density=density,
        transmittance=transmittance,
        positive_preview=positive,
        log_exposure=log_e,
        profile=working_profile,
    )
