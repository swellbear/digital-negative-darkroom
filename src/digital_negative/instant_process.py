"""Integral Polaroid / Instant film — expose → pod diffusion → finished card.

Level-3 appearance model: per-channel print H&D curves (temp-family morph),
mild diffusion, and a white border card. Not a dye-migration PDE. Profiles are
authored from public Polaroid datasheet aims (T-600 family and siblings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from .capture import ei_log_shift, reciprocity_factors
from .curves import FilmProfile
from .development import DevelopmentResult, linear_to_relative_log_exposure
from .digital_negative import DigitalNegative
from .display import linear_to_srgb


@dataclass
class InstantResult:
    """Finished integral card + mono proxies for shared UI strips."""

    reflectance: np.ndarray  # HxWx3 linear-ish reflectance 0..~1
    preview: np.ndarray  # bordered sRGB-ish 0..1 for display
    density: np.ndarray  # mono print density proxy
    transmittance: np.ndarray  # unused for enlarger; 10^(-D) proxy
    log_exposure: np.ndarray
    profile: FilmProfile
    process: str
    card_rgb: np.ndarray  # HxWx3 uint8 display card
    meta: dict[str, Any]


def _sigmoid_curve(
    log_e: np.ndarray,
    *,
    dmin: float,
    dmax: float,
    toe: float,
    slope: float,
    shoulder: float,
) -> np.ndarray:
    """Smooth medium-contrast print H&D on a relative log-E grid."""
    x = (log_e - toe) * slope
    # Logistic → density; shoulder softens the top.
    dens = dmin + (dmax - dmin) / (1.0 + np.exp(-x))
    roll = 1.0 / (1.0 + np.exp((log_e - shoulder) * 2.2))
    dens = dmin + (dens - dmin) * (0.55 + 0.45 * roll)
    return dens.astype(np.float64)


def _default_layer_points(channel: str) -> list[list[float]]:
    """Author T-600-like medium-contrast RGB print curves when JSON omits them."""
    log_e = np.linspace(0.5, 4.5, 19)
    # Blue dyes sit slightly steeper / toe later on classic integral packs.
    if channel == "blue":
        dens = _sigmoid_curve(log_e, dmin=0.08, dmax=1.72, toe=2.05, slope=2.05, shoulder=3.55)
    elif channel == "green":
        dens = _sigmoid_curve(log_e, dmin=0.07, dmax=1.68, toe=2.15, slope=1.95, shoulder=3.65)
    else:  # red
        dens = _sigmoid_curve(log_e, dmin=0.06, dmax=1.62, toe=2.22, slope=1.85, shoulder=3.75)
    return [[float(x), float(y)] for x, y in zip(log_e, dens, strict=True)]


def _interp_density(points: list[list[float]], log_e: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    order = np.argsort(pts[:, 0])
    xs, ys = pts[order, 0], pts[order, 1]
    fn = PchipInterpolator(xs, ys, extrapolate=False)
    lo, hi = float(xs[0]), float(xs[-1])
    clipped = np.clip(log_e, lo, hi)
    out = fn(clipped)
    out = np.where(log_e < lo, ys[0], out)
    out = np.where(log_e > hi, ys[-1], out)
    return out.astype(np.float32)


def _temp_morph_scales(instant: dict[str, Any], temp_c: float) -> tuple[float, float, float]:
    """Contrast / Dmax / time scales from cold–normal–hot datasheet families."""
    fam = instant.get("temp_family") or {}
    normal = float(fam.get("normal_c", instant.get("normal_temp_c", 21.0)))
    cold = float(fam.get("cold_c", 13.0))
    hot = float(fam.get("hot_c", 35.0))
    t = float(np.clip(temp_c, cold, hot))
    if t <= normal:
        u = (t - cold) / max(normal - cold, 1e-6)
        contrast = float(fam.get("cold_contrast_scale", 0.82)) * (1.0 - u) + 1.0 * u
        dmax = float(fam.get("cold_dmax_scale", 0.90)) * (1.0 - u) + 1.0 * u
        time_s = float(fam.get("cold_time_scale", 1.7)) * (1.0 - u) + 1.0 * u
    else:
        u = (t - normal) / max(hot - normal, 1e-6)
        contrast = 1.0 * (1.0 - u) + float(fam.get("hot_contrast_scale", 1.18)) * u
        dmax = 1.0 * (1.0 - u) + float(fam.get("hot_dmax_scale", 1.06)) * u
        time_s = 1.0 * (1.0 - u) + float(fam.get("hot_time_scale", 0.7)) * u
    return contrast, dmax, time_s


def auto_process_minutes(profile: FilmProfile, temp_c: float) -> float:
    instant = profile.raw.get("instant") or {}
    normal = float(instant.get("normal_minutes", profile.defaults.get("development_minutes", 3.0)))
    _c, _d, time_scale = _temp_morph_scales(instant, temp_c)
    minutes = normal * time_scale
    return float(np.clip(round(minutes * 4.0) / 4.0, 0.75, 12.0))


def _scene_linear_rgb(dn: DigitalNegative) -> np.ndarray:
    img = np.asarray(dn.image, dtype=np.float32)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[-1] >= 3:
        return img[..., :3].astype(np.float32)
    return np.stack([img[..., 0], img[..., 0], img[..., 0]], axis=-1)


def _apply_border(rgb: np.ndarray, border: dict[str, float]) -> np.ndarray:
    """Composite the image into a classic white integral card."""
    h, w = rgb.shape[:2]
    left = float(border.get("left", 0.08))
    right = float(border.get("right", 0.08))
    top = float(border.get("top", 0.08))
    bottom = float(border.get("bottom", 0.22))
    # Outer card size from image + margins.
    card_w = int(round(w / max(1.0 - left - right, 0.2)))
    card_h = int(round(h / max(1.0 - top - bottom, 0.2)))
    card_w = max(card_w, w + 8)
    card_h = max(card_h, h + 8)
    card = np.ones((card_h, card_w, 3), dtype=np.float32) * 0.96
    x0 = int(round(left * card_w))
    y0 = int(round(top * card_h))
    x1 = min(card_w, x0 + w)
    y1 = min(card_h, y0 + h)
    patch = rgb[: y1 - y0, : x1 - x0]
    card[y0:y1, x0:x1] = patch
    # Soft drop-shadow along the image well.
    card[y0 : min(y0 + 2, y1), x0:x1] *= 0.92
    return card


def process_instant(
    dn: DigitalNegative,
    profile: FilmProfile,
    *,
    process_temp_c: float = 21.0,
    process_minutes: float | None = None,
    contrast_modifier: float = 0.0,
    chroma: float = 1.0,
    warmth: float = 0.0,
    diffusion: float | None = None,
    border: bool = True,
    exposure_index: float | None = None,
    scene_exposure_seconds: float | None = None,
    mid_log_e: float = 2.2,
    commit: bool = True,
) -> InstantResult:
    """Render an integral instant card from the working Digital Negative."""
    if not str(profile.type).lower().startswith("instant_"):
        raise ValueError(f"Not an instant film profile: {profile.type}")

    instant = dict(profile.raw.get("instant") or {})
    look = dict(instant.get("look") or {})
    temp_c = float(process_temp_c)
    contrast_scale, dmax_scale, _time_scale = _temp_morph_scales(instant, temp_c)
    minutes = (
        float(process_minutes)
        if process_minutes is not None
        else auto_process_minutes(profile, temp_c)
    )
    # Slight under/over process vs auto-for-temp nudges Dmax / contrast.
    normal_auto = auto_process_minutes(profile, temp_c)
    time_ratio = float(np.clip(minutes / max(normal_auto, 1e-6), 0.45, 2.2))
    contrast_scale *= 0.92 + 0.08 * time_ratio
    dmax_scale *= 0.90 + 0.10 * time_ratio

    ei = float(
        exposure_index
        if exposure_index is not None
        else dn.metadata.get("ingest", {}).get("exposure_index", profile.iso)
    )
    scene_t = scene_exposure_seconds
    if scene_t is None and "scene_exposure_seconds" in dn.metadata.get("ingest", {}):
        scene_t = float(dn.metadata["ingest"]["scene_exposure_seconds"])
    recip_shift, recip_contrast = reciprocity_factors(scene_t, profile_raw=profile.raw)

    rgb_lin = _scene_linear_rgb(dn)
    # Per-channel log-E (same mid anchor as develop()).
    log_e = np.stack(
        [
            linear_to_relative_log_exposure(rgb_lin[..., c], mid_log_e=mid_log_e)
            for c in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    log_e = log_e + ei_log_shift(profile.iso, ei) + recip_shift
    # User N± + reciprocity contrast.
    n_mod = float(np.clip(contrast_modifier, -1.5, 1.5)) + float(recip_contrast) * 0.15
    pivot = float(mid_log_e)
    log_e = pivot + (log_e - pivot) * (1.0 + 0.35 * n_mod) * contrast_scale

    layers_block = instant.get("layers") or {}
    channel_names = ("red", "green", "blue")
    dens_ch = []
    dye_ceilings: list[float] = []
    for name in channel_names:
        pts = layers_block.get(name) or _default_layer_points(name)
        d = _interp_density(pts, log_e[..., channel_names.index(name)])
        # Morph Dmax with temperature / process time against the authored fog floor.
        dmin_curve = float(min(p[1] for p in pts))
        dmax_curve = float(max(p[1] for p in pts))
        dens_ch.append(dmin_curve + (d - dmin_curve) * dmax_scale)
        dye_ceilings.append(dmin_curve + (dmax_curve - dmin_curve) * dmax_scale)
    # Authored layer curves follow negative-forming / donor-sheet dye density
    # (rises with exposure). Integral transfer migrates the *unexposed* dye onto
    # the positive receiver — invert so the theoretical card is a finished
    # positive print, not a colour negative.
    density_neg = np.stack(dens_ch, axis=-1).astype(np.float32)
    ceiling = np.asarray(dye_ceilings, dtype=np.float32).reshape(1, 1, 3)
    density_rgb = np.clip(ceiling - density_neg, 0.0, None).astype(np.float32)

    # Reflection density of the finished integral card → reflectance.
    reflectance = np.power(10.0, -np.clip(density_rgb, 0.0, 4.0)).astype(np.float32)

    # Mild pod diffusion (spatial smear) — classic soft integral look.
    diff = float(
        diffusion if diffusion is not None else look.get("diffusion", 0.12)
    )
    diff = float(np.clip(diff, 0.0, 1.0))
    if diff > 1e-4:
        sigma = 0.4 + 2.8 * diff
        soft = np.stack(
            [gaussian_filter(reflectance[..., c], sigma=sigma, mode="nearest") for c in range(3)],
            axis=-1,
        ).astype(np.float32)
        reflectance = (1.0 - 0.65 * diff) * reflectance + (0.65 * diff) * soft

    # Chroma / warmth look knobs (grounded, not free grading).
    chroma = float(np.clip(chroma * float(look.get("saturation", 1.0)), 0.0, 2.0))
    warmth = float(np.clip(warmth + float(look.get("warmth", 0.0)), -1.0, 1.0))
    luma = (
        0.2126 * reflectance[..., 0]
        + 0.7152 * reflectance[..., 1]
        + 0.0722 * reflectance[..., 2]
    )
    reflectance = luma[..., None] + chroma * (reflectance - luma[..., None])
    reflectance = reflectance.copy()
    reflectance[..., 0] *= 1.0 + 0.08 * warmth
    reflectance[..., 2] *= 1.0 - 0.06 * warmth
    reflectance = np.clip(reflectance, 0.0, 1.25).astype(np.float32)

    # Display encoding
    disp = linear_to_srgb(np.clip(reflectance, 0.0, 1.0))
    border_spec = instant.get("border") or {
        "left": 0.08,
        "right": 0.08,
        "top": 0.08,
        "bottom": 0.22,
    }
    if border:
        card = _apply_border(disp, border_spec)
    else:
        card = disp
    card_u8 = np.clip(np.round(card * 255.0), 0, 255).astype(np.uint8)

    dens_mono = (
        0.2126 * density_rgb[..., 0]
        + 0.7152 * density_rgb[..., 1]
        + 0.0722 * density_rgb[..., 2]
    ).astype(np.float32)
    t_proxy = np.power(10.0, -np.clip(dens_mono, 0.0, 4.0)).astype(np.float32)

    meta = {
        "process": "instant_integral",
        "film_id": profile.id,
        "film_name": profile.name,
        "process_temp_c": temp_c,
        "process_minutes": minutes,
        "contrast_modifier": float(contrast_modifier),
        "chroma": chroma,
        "warmth": warmth,
        "diffusion": diff,
        "exposure_index": ei,
        "chemistry_mode": "instant",
    }

    if commit:
        dev = dn.metadata.setdefault("development", {})
        dev.update(
            {
                "enabled": True,
                "developer_id": "pod",
                "developer_name": "Integral reagent pod",
                "development_minutes": minutes,
                "relative_time": minutes
                / max(float(instant.get("normal_minutes", 3.0)), 1e-6),
                "contrast_modifier": float(contrast_modifier),
                "process": "instant_integral",
                "process_temp_c": temp_c,
                "chroma": chroma,
                "warmth": warmth,
                "diffusion": diff,
            }
        )
        dn.metadata.setdefault("print", {})["enabled"] = False
        dn.metadata.setdefault("instant", {}).update(meta)
        hist = dn.metadata.setdefault("history", [])
        hist.append({"op": "instant_process", **meta})
        dn.touch()

    return InstantResult(
        reflectance=reflectance,
        preview=card.astype(np.float32),
        density=dens_mono,
        transmittance=t_proxy,
        log_exposure=log_e.mean(axis=-1).astype(np.float32),
        profile=profile,
        process="instant_integral",
        card_rgb=card_u8,
        meta=meta,
    )


def develop_instant_as_result(
    dn: DigitalNegative,
    profile: FilmProfile,
    **kwargs: Any,
) -> DevelopmentResult:
    """Adapter so ``develop()`` can return a DevelopmentResult for Instant films."""
    inst = process_instant(dn, profile, **kwargs)
    # positive_preview is the finished card (0..1 float), matching color path usage.
    return DevelopmentResult(
        density=inst.density,
        transmittance=inst.transmittance,
        positive_preview=inst.preview,
        log_exposure=inst.log_exposure,
        profile=inst.profile,
        spectral_transmittance=None,
        color_process=inst.process,
        dye_concentrations=None,
    )
