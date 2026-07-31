"""Spectral color development: C-41 negatives and E-6 slides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .capture import ei_log_shift, reciprocity_factors
from .chemistry import resolve_relative_time
from .curves import FilmProfile, modify_curve
from .digital_negative import DigitalNegative
from .grain import apply_grain
from .spectral import (
    combine_dye_densities,
    density_to_transmittance_spectral,
    encode_srgb,
    gaussian_spectrum,
    layer_exposures_from_spectral,
    profile_layer_spectra,
    rgb_display_from_xyz,
    spectral_to_xyz,
    xyz_to_spectral,
)
from .variability import process_micro_variation


@dataclass
class ColorDevelopmentResult:
    """Color develop output — dye concentrations + spectral transmittance."""

    density: np.ndarray  # …×K layer densities (cyan, magenta, yellow)
    transmittance: np.ndarray  # …×N spectral transmittance
    positive_preview: np.ndarray  # HxWx3 display-referred preview
    log_exposure: np.ndarray  # …×K per-layer log-E
    profile: FilmProfile
    process: str  # "c41" | "e6"
    dye_concentrations: np.ndarray
    spectral_density: np.ndarray


def _layer_curve_interpolator(profile: FilmProfile, layer_name: str):
    raw = profile.raw.get("spectral", {}).get("layers", {}).get(layer_name) or {}
    points = raw.get("curve", {}).get("points")
    if points:
        pts = np.asarray(points, dtype=np.float64)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        fog = float(raw.get("curve", {}).get("base_plus_fog", pts[0, 1]))
        return PchipInterpolator(pts[:, 0], pts[:, 1], extrapolate=False), fog, pts
    # Fall back to master B&W-compatible curve.
    return (
        PchipInterpolator(profile.log_exposure, profile.density, extrapolate=False),
        float(profile.base_plus_fog),
        np.stack([profile.log_exposure, profile.density], axis=1),
    )


def _density_from_log_e(log_e: np.ndarray, interpolator, fog: float, pts: np.ndarray) -> np.ndarray:
    log_min = float(pts[0, 0])
    log_max = float(pts[-1, 0])
    clipped = np.clip(log_e, log_min, log_max)
    dens = interpolator(clipped)
    dens = np.where(log_e < log_min, pts[0, 1], dens)
    dens = np.where(log_e > log_max, pts[-1, 1], dens)
    return np.maximum(dens, fog * 0.5).astype(np.float32)


def _scene_xyz(dn: DigitalNegative) -> np.ndarray:
    img = np.asarray(dn.image, dtype=np.float32)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[-1] >= 3:
        return img[..., :3]
    raise ValueError("Color develop requires a 3-channel Digital Negative")


def _linear_to_log_e(linear: np.ndarray, *, mid_log_e: float = 2.2) -> np.ndarray:
    eps = 1e-6
    positive = linear[linear > eps]
    mid = float(np.median(positive)) if positive.size else 0.18
    mid = max(mid, eps)
    return (mid_log_e + np.log10(np.maximum(linear, eps) / mid)).astype(np.float32)


def _apply_interimage(densities: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        return densities
    m = np.asarray(matrix, dtype=np.float32)
    flat = densities.reshape(-1, m.shape[0])
    out = flat @ m.T
    return np.clip(out, 0.0, None).reshape(densities.shape).astype(np.float32)


def _preview_from_spectral_T(transmittance: np.ndarray, *, invert: bool) -> np.ndarray:
    """Spectral T → display sRGB.

    ``invert=False`` — light-table view of the negative (C-41 orange mask).
    ``invert=True`` — approximate positive scan / optical invert for inspection.
    """
    xyz = spectral_to_xyz(transmittance)
    if invert:
        # Approximate optical print / scan invert for inspection.
        xyz = np.maximum(xyz, 1e-4)
        # Normalize by 99th percentile so mask orange doesn't crush.
        scale = np.percentile(xyz, 99.0, axis=(0, 1) if xyz.ndim == 3 else 0)
        scale = np.maximum(scale, 1e-4)
        xyz = np.clip(scale / xyz, 0.0, 16.0)
        xyz = xyz / max(float(np.percentile(xyz, 99.0)), 1e-4)
    rgb = encode_srgb(rgb_display_from_xyz(xyz))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def color_negative_lightbox_preview(transmittance: np.ndarray) -> np.ndarray:
    """C-41 spectral transmittance as seen on a light table (orange mask, not inverted)."""
    return _preview_from_spectral_T(transmittance, invert=False)


def develop_color(
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
    scene_exposure_seconds: float | None = None,
    halation: float | None = None,
    commit: bool = True,
) -> ColorDevelopmentResult:
    """Spectral C-41 or E-6 develop from a color film profile."""
    process = "e6" if str(profile.type).lower() == "color_slide" else "c41"
    spectral_block = profile.raw.get("spectral") or {}
    layer_order = list(spectral_block.get("layer_order") or ["cyan", "magenta", "yellow"])
    sensitivities, dye_spectra, mask = profile_layer_spectra(spectral_block)
    inter = spectral_block.get("interimage")
    inter_m = np.asarray(inter, dtype=np.float32) if inter is not None else None

    dev = dn.metadata.setdefault("development", {})
    ingest = dn.metadata.setdefault("ingest", {})
    developer = str(
        developer_id if developer_id is not None else dev.get("developer_id", "c41_standard")
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
        exposure_index if exposure_index is not None else ingest.get("exposure_index", profile.iso)
    )
    scene_t = scene_exposure_seconds
    if scene_t is None and "scene_exposure_seconds" in ingest:
        scene_t = float(ingest["scene_exposure_seconds"])
    hal = float(halation if halation is not None else dev.get("halation", 0.0))

    # Morph master curve for chemistry×time (affects layer fog/contrast via rel).
    working_profile = modify_curve(
        profile,
        relative_time=rel,
        contrast_modifier=contrast,
        developer_id=developer,
        development_minutes=minutes,
    )
    curve_meta = working_profile.raw.get("_last_curve_meta") or {}

    recip_shift, recip_contrast = reciprocity_factors(scene_t, profile_raw=profile.raw)
    xyz = _scene_xyz(dn)
    spectral = xyz_to_spectral(xyz)
    # Mild spectral halation: blur not available cheaply — lift long-wave fog.
    if hal > 1e-6:
        lift = hal * 0.08 * gaussian_spectrum(650, 80, amplitude=1.0)
        spectral = spectral + lift.astype(np.float32)

    exposures = layer_exposures_from_spectral(spectral, sensitivities)  # …×3
    log_e = _linear_to_log_e(exposures, mid_log_e=mid_log_e)
    log_e = (log_e + ei_log_shift(profile.iso, ei) + recip_shift).astype(np.float32)
    # Push/pull via relative time shifts log-E slightly per layer.
    log_e = (log_e + np.log10(max(rel, 0.05)) * 0.35).astype(np.float32)
    log_e = (log_e + recip_contrast * 0.15).astype(np.float32)

    layer_dens = []
    for i, name in enumerate(layer_order):
        interp, fog, pts = _layer_curve_interpolator(working_profile, name)
        # Contrast modifier stretches around mid.
        le = mid_log_e + (log_e[..., i] - mid_log_e) * (1.0 + 0.35 * contrast)
        d = _density_from_log_e(le, interp, fog, pts)
        layer_dens.append(d)
    densities = np.stack(layer_dens, axis=-1).astype(np.float32)
    densities = _apply_interimage(densities, inter_m)

    seed = int(dn.metadata.get("process_seed", 0))
    # Grain per dye layer (reuse mono grain on each plane).
    grain_eff = grain * float(style.get("grain_bias", 1.0)) * (0.85 + 0.25 * rel)
    grained = []
    for i in range(densities.shape[-1]):
        plane = process_micro_variation(
            densities[..., i], process_seed=seed + i * 17, strength=process_variation
        )
        plane = apply_grain(
            plane, profile=working_profile, grain_strength=grain_eff, process_seed=seed + i
        )
        grained.append(plane)
    densities = np.stack(grained, axis=-1).astype(np.float32)

    if process == "e6":
        # Reversal: invert layer densities around a pivot, then form image dyes.
        dmax = np.percentile(densities, 99.5, axis=(0, 1))
        dmax = np.maximum(dmax, 1.2)
        concentrations = np.clip(dmax - densities, 0.0, None)
        # Clear mask for slides.
        mask_vec = np.zeros(len(layer_order), dtype=np.float32)
    else:
        # C-41: dye amount follows developed density; add orange mask.
        concentrations = np.maximum(densities - working_profile.base_plus_fog * 0.5, 0.0)
        mask_vec = np.array(
            [float(mask.get(n, 0.0)) for n in layer_order], dtype=np.float32
        )
        concentrations = concentrations + mask_vec.reshape((1,) * (concentrations.ndim - 1) + (-1,))

    spectral_density = combine_dye_densities(concentrations, dye_spectra)
    transmittance = density_to_transmittance_spectral(spectral_density)
    preview = _preview_from_spectral_T(transmittance, invert=(process == "c41"))

    user_grain = float(
        grain_strength
        if grain_strength is not None
        else dev.get("grain_strength", profile.defaults.get("grain_strength", 1.0))
    )
    dn.metadata["film_profile"] = {
        "id": profile.id,
        "name": profile.name,
        "type": profile.type,
        "version": profile.version,
        "iso": profile.iso,
    }
    ingest.update({"exposure_index": ei, "box_speed": float(profile.iso)})
    if scene_t is not None:
        ingest["scene_exposure_seconds"] = float(scene_t)
    update: dict[str, Any] = {
        "enabled": True,
        "process": process,
        "chemistry_mode": "color",
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
        "layer_order": layer_order,
    }
    if minutes is not None:
        update["development_minutes"] = float(minutes)
    dn.metadata["development"].update(update)
    dn.metadata.setdefault("ui_state", {})["current_stage"] = "development"

    if commit:
        stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
        if "development" not in stages:
            stages.append("development")
        dn.touch()
        hist = {
            "op": "develop_color",
            "process": process,
            "film_profile_id": profile.id,
            "developer_id": developer,
            "developer_name": style["name"],
            "relative_time": rel,
            "contrast_modifier": contrast,
            "grain_strength": user_grain,
            "exposure_index": ei,
        }
        if minutes is not None:
            hist["development_minutes"] = float(minutes)
        dn.metadata.setdefault("history", []).append(hist)

    return ColorDevelopmentResult(
        density=densities,
        transmittance=transmittance,
        positive_preview=preview,
        log_exposure=log_e,
        profile=working_profile,
        process=process,
        dye_concentrations=concentrations.astype(np.float32),
        spectral_density=spectral_density,
    )
