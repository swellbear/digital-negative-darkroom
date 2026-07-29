"""Development engine: linear scene → log exposure → density → viewable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import FilmProfile, modify_curve
from .digital_negative import DigitalNegative


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
    """Map near-linear scene values onto a film relative log-E axis.

    mid_scene (default: image median of positive pixels) is placed at mid_log_e
    so a typical exposure sits on the straight-line portion of HP5-like curves.
    """
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
    """Simple enlarger-style positive from a negative transmission image.

    Thin negative (low density / shadows) passes more light → dark print.
    Dense negative (highlights) holds light back → light print.
    So display lightness rises with negative density.
    """
    density = -np.log10(np.maximum(transmittance, 1e-6))
    span = max(d_max - fog, 1e-3)
    positive = np.clip((density - fog) / span, 0.0, 1.0)
    # Mild paper-like shoulder so the preview doesn't look purely linear
    positive = np.power(positive, 0.90)
    return positive.astype(np.float32)


def develop(
    dn: DigitalNegative,
    profile: FilmProfile,
    *,
    relative_time: float | None = None,
    contrast_modifier: float | None = None,
    mid_log_e: float = 2.2,
) -> DevelopmentResult:
    """Apply film characteristic curve development to a Digital Negative."""
    dev = dn.metadata.setdefault("development", {})
    rel = float(relative_time if relative_time is not None else dev.get("relative_time", 1.0))
    contrast = float(
        contrast_modifier if contrast_modifier is not None else dev.get("contrast_modifier", 0.0)
    )

    working_profile = modify_curve(profile, relative_time=rel, contrast_modifier=contrast)
    luminance = dn.to_luminance()
    log_e = linear_to_relative_log_exposure(luminance, mid_log_e=mid_log_e)
    density = working_profile.density_from_log_exposure(log_e)
    transmittance = density_to_transmittance(density)
    # Practical enlarging range: deep shadows near fog through a solid
    # highlight around densitometric ~1.6 (typical MG paper scale).
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
    dn.metadata["development"].update(
        {
            "enabled": True,
            "relative_time": rel,
            "contrast_modifier": contrast,
        }
    )
    dn.touch()
    history = dn.metadata.setdefault("history", [])
    history.append(
        {
            "op": "develop",
            "film_profile_id": profile.id,
            "relative_time": rel,
            "contrast_modifier": contrast,
        }
    )

    return DevelopmentResult(
        density=density,
        transmittance=transmittance,
        positive_preview=positive,
        log_exposure=log_e,
        profile=working_profile,
    )
