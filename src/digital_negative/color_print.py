"""Spectral RA-4 color print from a C-41 negative transmittance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .digital_negative import DigitalNegative
from .papers import PaperProfile
from .spectral import (
    N_WAVELENGTHS,
    combine_dye_densities,
    density_to_transmittance_spectral,
    encode_srgb,
    gaussian_spectrum,
    illuminant_d65,
    layer_exposures_from_spectral,
    profile_layer_spectra,
    rgb_display_from_xyz,
    spectral_to_xyz,
    transmittance_to_density_spectral,
)


@dataclass
class ColorPrintResult:
    print_density: np.ndarray  # …×K paper dye densities
    reflectance: np.ndarray  # …×N spectral reflectance
    preview: np.ndarray  # HxWx3 display-referred
    filtration: dict[str, float]


def _balance_ra4_preview(preview: np.ndarray, *, target_median: float = 0.42) -> np.ndarray:
    """Neutralize median RGB and lift overall level (approx. dichroic starting pack)."""
    img = np.asarray(preview, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] < 3:
        return img
    flat = img[..., :3].reshape(-1, 3)
    # Ignore pure border blacks if present.
    lit = flat[flat.max(axis=1) > 1e-4]
    if lit.size == 0:
        return img
    med = np.median(lit, axis=0)
    target = float(np.clip(target_median, 0.15, 0.65))
    gains = target / np.maximum(med, 1e-5)
    # Cap gains so a failed exposure can't blow out a single channel wildly.
    gains = np.clip(gains, 0.25, 12.0)
    out = img.copy()
    out[..., :3] = np.clip(out[..., :3] * gains.reshape((1, 1, 3)), 0.0, 1.0)
    return out.astype(np.float32)


def cc_filter_spectrum(cyan: float, magenta: float, yellow: float) -> np.ndarray:
    """Dichroic / CC pack as multiplicative spectral transmittance."""
    # CC units ≈ 0.01 density at dye peak; map knobs in 0–100 range.
    t = np.ones(N_WAVELENGTHS, dtype=np.float64)
    for amount, peak, width in (
        (cyan, 650.0, 45.0),
        (magenta, 550.0, 45.0),
        (yellow, 450.0, 45.0),
    ):
        dens = max(float(amount), 0.0) * 0.01 * gaussian_spectrum(peak, width, amplitude=1.0)
        t *= np.power(10.0, -dens)
    return t.astype(np.float32)


def enlarger_illuminant() -> np.ndarray:
    """Tungsten-ish enlarger lamp SPD."""
    spd = (
        0.35 * gaussian_spectrum(450, 60)
        + 0.75 * gaussian_spectrum(580, 80)
        + 1.00 * gaussian_spectrum(650, 70)
    )
    return (spd / (spd.max() + 1e-12)).astype(np.float32)


def print_color_negative(
    transmittance: np.ndarray,
    dn: DigitalNegative,
    paper: PaperProfile,
    *,
    base_exposure_seconds: float = 8.0,
    cc_cyan: float = 0.0,
    cc_magenta: float = 0.0,
    cc_yellow: float = 0.0,
    contrast: float = 0.0,
    local_stops: np.ndarray | None = None,
    dry_down: float = 0.0,
    border_frac: float = 0.0,
    commit: bool = True,
) -> ColorPrintResult:
    """RA-4: negative spectral T × lamp × CC → paper dyes → reflectance preview."""
    if str(paper.type).lower() not in {"color_ra4", "color"}:
        raise ValueError(f"Paper {paper.id} is not a color RA-4 profile")

    spectral_block = paper.raw.get("spectral") or {}
    layer_order = list(spectral_block.get("layer_order") or ["cyan", "magenta", "yellow"])
    sensitivities, dye_spectra, _mask = profile_layer_spectra(spectral_block)

    lamp = enlarger_illuminant()
    filt = cc_filter_spectrum(cc_cyan, cc_magenta, cc_yellow)
    # Light reaching the paper through the negative.
    # T_neg × lamp × filter
    t_neg = np.asarray(transmittance, dtype=np.float32)
    if t_neg.ndim == 2:
        # Mono fallback — broadcast
        t_neg = np.repeat(t_neg[..., None], N_WAVELENGTHS, axis=-1)
    incident = t_neg * lamp.reshape((1,) * (t_neg.ndim - 1) + (-1,))
    incident = incident * filt.reshape((1,) * (t_neg.ndim - 1) + (-1,))

    if local_stops is not None:
        # Dodge/burn: stops modulate overall exposure (luminance-linked).
        stops = np.asarray(local_stops, dtype=np.float32)
        if stops.ndim == t_neg.ndim - 1:
            factor = np.power(2.0, stops)[..., None]
        else:
            factor = np.power(2.0, stops)
        incident = incident * factor

    # Timer exposure scales irradiance.
    seconds = max(float(base_exposure_seconds), 0.05)
    # Reference: 8s → mid scale 1.0
    exposure_scale = seconds / 8.0
    # Contrast knob steepens paper response around mid.
    layer_exp = layer_exposures_from_spectral(incident * exposure_scale, sensitivities)

    # Paper characteristic: density = dmin + (dmax-dmin) * smoothstep(log E)
    dmin = float(paper.dmin)
    dmax = float(paper.dmax)
    eps = 1e-8
    log_e = np.log10(np.maximum(layer_exp, eps))
    # Center log-E so mid greys land mid-scale.
    center = float(np.median(log_e))
    x = (log_e - center) * (1.0 + 0.45 * float(contrast))
    # Logistic paper curve
    toe = float(spectral_block.get("toe", 0.35))
    shoulder = float(spectral_block.get("shoulder", 0.35))
    # Smooth response 0..1
    resp = 1.0 / (1.0 + np.exp(-x / max(toe, 0.05)))
    resp = np.clip(resp, 0.0, 1.0)
    # Soft shoulder compression
    resp = resp / (1.0 + shoulder * resp)
    resp = resp / (resp.max() + 1e-8) if float(resp.max()) > 0 else resp
    concentrations = (dmin + (dmax - dmin) * resp).astype(np.float32)

    if dry_down > 1e-6:
        # Dry-down deepens dyes slightly.
        concentrations = concentrations * (1.0 + 0.01 * float(dry_down))

    spectral_density = combine_dye_densities(concentrations, dye_spectra)
    # Reflectance ≈ 10^(-density) under viewing illuminant (simplified).
    reflectance = density_to_transmittance_spectral(spectral_density)
    xyz = spectral_to_xyz(reflectance, illuminant=illuminant_d65())
    preview = encode_srgb(rgb_display_from_xyz(xyz))
    preview = np.clip(preview, 0.0, 1.0).astype(np.float32)
    # Level-3 stand-in for a starting dichroic pack: neutralize median RGB and
    # lift crushed exposures so Live print reads as a print, not a cyan mask.
    preview = _balance_ra4_preview(preview)

    bf = float(np.clip(border_frac, 0.0, 0.45))
    if bf > 1e-6 and preview.ndim >= 2:
        h, w = preview.shape[:2]
        y0, x0 = int(h * bf), int(w * bf)
        y1, x1 = h - y0, w - x0
        bordered = np.zeros_like(preview)
        bordered[y0:y1, x0:x1] = preview[y0:y1, x0:x1]
        preview = bordered

    filtration = {
        "cc_cyan": float(cc_cyan),
        "cc_magenta": float(cc_magenta),
        "cc_yellow": float(cc_yellow),
        "base_exposure_seconds": seconds,
        "contrast": float(contrast),
    }

    if commit:
        dn.metadata["print"] = {
            "enabled": True,
            "process": "ra4",
            "chemistry_mode": "color",
            "paper_id": paper.id,
            "paper_name": paper.name,
            "filtration": {"type": "color", "values": filtration},
        }
        stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
        if "print" not in stages:
            stages.append("print")
        dn.metadata.setdefault("ui_state", {})["current_stage"] = "print"
        dn.touch()
        dn.metadata.setdefault("history", []).append(
            {
                "op": "print_color",
                "paper_id": paper.id,
                **filtration,
            }
        )

    return ColorPrintResult(
        print_density=concentrations,
        reflectance=reflectance.astype(np.float32),
        preview=preview,
        filtration=filtration,
    )


def finish_slide(
    transmittance: np.ndarray,
    dn: DigitalNegative,
    *,
    commit: bool = True,
) -> ColorPrintResult:
    """E-6 finish: spectral positive → display preview (no RA-4 paper)."""
    t = np.asarray(transmittance, dtype=np.float32)
    xyz = spectral_to_xyz(t, illuminant=illuminant_d65())
    preview = encode_srgb(rgb_display_from_xyz(xyz))
    preview = np.clip(preview, 0.0, 1.0).astype(np.float32)
    dens = transmittance_to_density_spectral(t)
    # Collapse to 3 dye-ish planes for shape compatibility.
    if dens.ndim >= 1 and dens.shape[-1] == N_WAVELENGTHS:
        # Sample at C/M/Y peaks
        from .spectral import WAVELENGTHS_NM

        idxs = [
            int(np.argmin(np.abs(WAVELENGTHS_NM - p)))
            for p in (650.0, 550.0, 450.0)
        ]
        planes = dens[..., idxs]
    else:
        planes = dens

    if commit:
        dn.metadata["print"] = {
            "enabled": True,
            "process": "e6_finish",
            "chemistry_mode": "color",
            "paper_id": None,
            "paper_name": "Slide mount",
            "filtration": {"type": "none", "values": {}},
        }
        stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
        if "print" not in stages:
            stages.append("print")
        dn.metadata.setdefault("ui_state", {})["current_stage"] = "print"
        dn.touch()
        dn.metadata.setdefault("history", []).append({"op": "finish_slide"})

    return ColorPrintResult(
        print_density=planes.astype(np.float32),
        reflectance=t,
        preview=preview,
        filtration={},
    )
