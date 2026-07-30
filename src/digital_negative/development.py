"""Development engine: linear scene → log exposure → density → viewable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import (
    apply_halation,
    ei_log_shift,
    filtered_luminance,
    reciprocity_factors,
)
from .chemistry import resolve_relative_time
from .curves import FilmProfile, modify_curve
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
    development_minutes: float | None = None,
    contrast_modifier: float | None = None,
    grain_strength: float | None = None,
    developer_id: str | None = None,
    mid_log_e: float = 2.2,
    process_variation: float = 1.0,
    exposure_index: float | None = None,
    contrast_filter: str | None = None,
    scene_exposure_seconds: float | None = None,
    halation: float | None = None,
    commit: bool = True,
) -> DevelopmentResult:
    """Apply film characteristic curve development to a Digital Negative.

    commit=False: live preview only — updates working params, no history/lock.

    Prefer ``development_minutes`` with a named film chemistry; ``relative_time``
    remains for legacy abstract styles and tests.

    Capture realism knobs (EI, contrast filter, reciprocity, halation) shift
    where the scene sits on the curve before chemistry runs.
    """
    dev = dn.metadata.setdefault("development", {})
    ingest = dn.metadata.setdefault("ingest", {})
    developer = str(
        developer_id if developer_id is not None else dev.get("developer_id", "standard")
    )
    minutes_arg = development_minutes
    if minutes_arg is None and "development_minutes" in dev and relative_time is None:
        minutes_arg = float(dev["development_minutes"])
    rel_arg = relative_time if relative_time is not None else (
        None if minutes_arg is not None else float(dev.get("relative_time", 1.0))
    )
    rel, minutes, style = resolve_relative_time(
        profile,
        developer,
        development_minutes=minutes_arg,
        relative_time=rel_arg,
    )
    contrast = float(
        contrast_modifier if contrast_modifier is not None else dev.get("contrast_modifier", 0.0)
    )
    grain = float(
        grain_strength
        if grain_strength is not None
        else dev.get("grain_strength", profile.defaults.get("grain_strength", 1.0))
    )
    ei = float(
        exposure_index
        if exposure_index is not None
        else ingest.get("exposure_index", profile.iso)
    )
    filt = str(
        contrast_filter
        if contrast_filter is not None
        else ingest.get("contrast_filter", "none")
    )
    scene_t = scene_exposure_seconds
    if scene_t is None and "scene_exposure_seconds" in ingest:
        scene_t = float(ingest["scene_exposure_seconds"])
    hal = float(
        halation if halation is not None else dev.get("halation", 0.0)
    )
    # Push slightly increases perceived grain; pull reduces it
    grain_eff = grain * float(style["grain_bias"]) * (0.85 + 0.25 * rel)

    recip_shift, recip_contrast = reciprocity_factors(
        scene_t, profile_raw=profile.raw
    )
    contrast_eff = contrast + recip_contrast

    working_profile = modify_curve(
        profile,
        relative_time=rel,
        contrast_modifier=contrast_eff,
        developer_id=developer,
        development_minutes=minutes,
    )
    curve_meta = working_profile.raw.get("_last_curve_meta") or {}
    luminance = filtered_luminance(dn, filt)
    log_e = linear_to_relative_log_exposure(luminance, mid_log_e=mid_log_e)
    log_e = (log_e + ei_log_shift(profile.iso, ei) + recip_shift).astype(np.float32)
    density = working_profile.density_from_log_exposure(log_e)
    seed = int(dn.metadata.get("process_seed", 0))
    density = process_micro_variation(density, process_seed=seed, strength=process_variation)
    density = apply_halation(density, strength=hal, fog=working_profile.base_plus_fog)
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

    user_grain = float(
        grain_strength
        if grain_strength is not None
        else dev.get("grain_strength", profile.defaults.get("grain_strength", 1.0))
    )
    # Always keep working parameters current for the UI
    dn.metadata["film_profile"] = {
        "id": profile.id,
        "name": profile.name,
        "type": profile.type,
        "version": profile.version,
        "iso": profile.iso,
    }
    ingest.update(
        {
            "exposure_index": ei,
            "box_speed": float(profile.iso),
            "contrast_filter": filt,
        }
    )
    if scene_t is not None:
        ingest["scene_exposure_seconds"] = float(scene_t)
    update = {
        "enabled": True,
        "developer_id": developer,
        "developer_name": style["name"],
        "relative_time": rel,
        "contrast_modifier": contrast,
        "grain_strength": user_grain,
        "process_variation": float(process_variation),
        "curve_source": curve_meta.get("curve_source", "morph"),
        "halation": hal,
        "reciprocity_log_shift": recip_shift,
        "reciprocity_contrast": recip_contrast,
    }
    if minutes is not None:
        update["development_minutes"] = float(minutes)
    if curve_meta.get("family_mode"):
        update["curve_family_mode"] = curve_meta["family_mode"]
    dn.metadata["development"].update(update)
    dn.metadata.setdefault("ui_state", {})["current_stage"] = "development"

    if commit:
        stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
        if "development" not in stages:
            stages.append("development")
        dn.touch()
        hist = {
            "op": "develop",
            "film_profile_id": profile.id,
            "developer_id": developer,
            "developer_name": style["name"],
            "relative_time": rel,
            "contrast_modifier": contrast,
            "grain_strength": user_grain,
            "process_variation": float(process_variation),
            "exposure_index": ei,
            "contrast_filter": filt,
            "halation": hal,
        }
        if minutes is not None:
            hist["development_minutes"] = float(minutes)
        if scene_t is not None:
            hist["scene_exposure_seconds"] = float(scene_t)
        dn.metadata.setdefault("history", []).append(hist)

    return DevelopmentResult(
        density=density,
        transmittance=transmittance,
        positive_preview=positive,
        log_exposure=log_e,
        profile=working_profile,
    )
