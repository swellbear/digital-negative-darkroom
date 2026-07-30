"""Capture-side realism: EI, contrast filters, reciprocity, halation.

These sit between the Digital Negative (CIE XYZ) and the film curve. They are
approximations — three-channel XYZ is not a spectral model — but they move the
simulator in the direction a film shooter actually steers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .digital_negative import DigitalNegative

# ——— Contrast filters (XYZ weights) ———————————————————————————————
# Relative tristimulus weights for common B&W contrast filters. Tuned so that
# "Yellow" gently darkens blue sky, "Red" goes further, "Green" lifts foliage
# relative to skin/brick. Not a substitute for measured spectral dye curves.

FILTER_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "none": (0.0, 1.0, 0.0),          # plain Y
    "yellow": (0.18, 0.72, 0.10),     # Wratten 8-ish
    "orange": (0.28, 0.68, 0.04),     # Wratten 21-ish
    "red": (0.55, 0.45, 0.00),        # Wratten 25-ish
    "green": (0.08, 0.78, 0.14),      # Wratten 58-ish
}

FILTER_LABELS: list[tuple[str, str]] = [
    ("None (Y)", "none"),
    ("Yellow", "yellow"),
    ("Orange", "orange"),
    ("Red", "red"),
    ("Green", "green"),
]


def filtered_luminance(
    dn: DigitalNegative,
    filter_id: str | None = "none",
) -> np.ndarray:
    """Scene luminance after an optional contrast filter.

    For CIE XYZ payloads, applies ``FILTER_WEIGHTS``. For RGB or mono, falls
    back to :meth:`DigitalNegative.to_luminance` (filter is a no-op unless XYZ).
    """
    fid = str(filter_id or "none").lower()
    img = dn.image
    if img.ndim == 2:
        return img.astype(np.float32)
    ingest = dn.metadata.get("ingest", {})
    space = str(ingest.get("working_space", ""))
    is_xyz = space == "CIE_XYZ" or ingest.get("luminance_channel") == "Y"
    if not is_xyz or fid not in FILTER_WEIGHTS or fid == "none":
        return dn.to_luminance()
    wx, wy, wz = FILTER_WEIGHTS[fid]
    # Renormalize so mid-grey Y stays roughly stable across filters.
    wsum = wx + wy + wz
    if wsum <= 1e-9:
        return dn.to_luminance()
    wx, wy, wz = wx / wsum, wy / wsum, wz / wsum
    xyz = img.astype(np.float32)
    out = wx * xyz[..., 0] + wy * xyz[..., 1] + wz * xyz[..., 2]
    return out.astype(np.float32)


# ——— Exposure index ——————————————————————————————————————————————


def ei_log_shift(box_iso: float, exposure_index: float) -> float:
    """Log10-E shift for rating film away from box speed.

    Rating HP5 at 1600 (2 stops over box) underexposes the scene by 2 stops,
    sliding it toward the toe — the same place push development then has to
    climb out of.
    """
    box = max(float(box_iso), 1.0)
    ei = max(float(exposure_index), 1.0)
    stops = np.log2(ei / box)
    return float(-stops * np.log10(2.0))


# ——— Reciprocity failure —————————————————————————————————————————


def default_reciprocity(profile_raw: dict[str, Any] | None = None) -> dict[str, float]:
    """Per-film reciprocity knobs, with sane B&W defaults.

    Model (Schwarzschild-ish): for exposure times above ``threshold_s``, the
    effective exposure falls as ``(t / t_th)^p`` with p < 1, and contrast rises
    slightly with the same factor.
    """
    raw = (profile_raw or {}).get("reciprocity") or {}
    return {
        "threshold_s": float(raw.get("threshold_s", 1.0)),
        "p": float(raw.get("p", 0.80)),
        "contrast_gain": float(raw.get("contrast_gain", 0.12)),
    }


def reciprocity_factors(
    exposure_seconds: float | None,
    *,
    profile_raw: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Return (log_e_shift, contrast_boost) for a scene exposure time."""
    if exposure_seconds is None or float(exposure_seconds) <= 0:
        return 0.0, 0.0
    t = float(exposure_seconds)
    cfg = default_reciprocity(profile_raw)
    th = max(cfg["threshold_s"], 1e-6)
    if t <= th:
        return 0.0, 0.0
    # Effective exposure fraction relative to reciprocity law.
    p = float(np.clip(cfg["p"], 0.4, 1.0))
    eff = (t / th) ** p
    # Actual time grew as t/th, but film only saw eff — deficit in stops.
    deficit_stops = np.log2(max(t / th, 1e-6) / max(eff, 1e-6))
    log_shift = float(-deficit_stops * np.log10(2.0))
    contrast = float(cfg["contrast_gain"] * deficit_stops)
    return log_shift, contrast


# ——— Halation ————————————————————————————————————————————————————


def apply_halation(
    density: np.ndarray,
    *,
    strength: float = 0.0,
    fog: float = 0.1,
) -> np.ndarray:
    """Bloom highlight density into neighbouring shadows (halation).

    Operates in density: bright scene areas (low density on the negative after
    inversion thinking — wait: high scene exposure → high negative density.
    Halation is light scattering in the base that exposes neighbouring
    emulsion, adding density around bright image areas — a soft glow of
    *extra density* around dense highlights on the negative, which prints as
    a light bloom around bright subject areas.
    """
    s = float(np.clip(strength, 0.0, 2.0))
    if s <= 1e-6:
        return density
    try:
        from scipy import ndimage
    except ImportError:
        return density

    d = np.asarray(density, dtype=np.float32)
    # Dense (bright-scene) regions seed the bloom.
    threshold = float(np.percentile(d, 92))
    seeds = np.maximum(d - threshold, 0.0)
    if float(seeds.max()) <= 1e-8:
        return d
    sigma = 1.2 + 2.8 * s
    bloom = ndimage.gaussian_filter(seeds, sigma=sigma)
    bloom = bloom / max(float(bloom.max()), 1e-6)
    # Add a little density around highlights; scale stays modest.
    out = d + bloom * (0.08 * s) * max(float(np.percentile(d, 99) - fog), 0.2)
    return out.astype(np.float32)
