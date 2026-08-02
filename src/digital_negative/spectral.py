"""Spectral helpers for color-negative / slide / RA-4 simulation.

Wavelength-sampled model (380–730 nm @ 10 nm). Scene XYZ is expanded to a
spectral estimate via CIE CMFs + a smooth non-negative reconstruction — an
explicit approximation (same honesty as the XYZ contrast-filter note in
``capture.py``). Not a copy of third-party spectral engines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ——— Wavelength grid ———————————————————————————————————————————————

WAVELENGTHS_NM = np.arange(380, 731, 10, dtype=np.float64)
N_WAVELENGTHS = int(WAVELENGTHS_NM.size)


def wavelength_grid() -> np.ndarray:
    return WAVELENGTHS_NM.copy()


# CIE 1931 2° CMFs, sampled / interpolated onto WAVELENGTHS_NM (approx).
# Values adapted from CIE standard observer tables (public domain).
def _build_cmfs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sparse CIE XYZ bar values (nm, x̄, ȳ, z̄) then interpolate.
    table = np.array(
        [
            [380, 0.0014, 0.0000, 0.0065],
            [390, 0.0042, 0.0001, 0.0201],
            [400, 0.0143, 0.0004, 0.0679],
            [410, 0.0435, 0.0012, 0.2074],
            [420, 0.1344, 0.0040, 0.6456],
            [430, 0.2839, 0.0116, 1.3856],
            [440, 0.3483, 0.0230, 1.7471],
            [450, 0.3362, 0.0380, 1.7721],
            [460, 0.2908, 0.0600, 1.6692],
            [470, 0.1954, 0.0910, 1.2876],
            [480, 0.0956, 0.1390, 0.8130],
            [490, 0.0320, 0.2080, 0.4652],
            [500, 0.0049, 0.3230, 0.2720],
            [510, 0.0093, 0.5030, 0.1582],
            [520, 0.0633, 0.7100, 0.0782],
            [530, 0.1655, 0.8620, 0.0422],
            [540, 0.2904, 0.9540, 0.0203],
            [550, 0.4334, 0.9950, 0.0087],
            [560, 0.5945, 0.9950, 0.0039],
            [570, 0.7621, 0.9520, 0.0021],
            [580, 0.9163, 0.8700, 0.0017],
            [590, 1.0263, 0.7570, 0.0011],
            [600, 1.0622, 0.6310, 0.0008],
            [610, 1.0026, 0.5030, 0.0003],
            [620, 0.8544, 0.3810, 0.0002],
            [630, 0.6424, 0.2650, 0.0000],
            [640, 0.4479, 0.1750, 0.0000],
            [650, 0.2835, 0.1070, 0.0000],
            [660, 0.1649, 0.0610, 0.0000],
            [670, 0.0874, 0.0320, 0.0000],
            [680, 0.0468, 0.0170, 0.0000],
            [690, 0.0227, 0.0082, 0.0000],
            [700, 0.0114, 0.0041, 0.0000],
            [710, 0.0058, 0.0021, 0.0000],
            [720, 0.0029, 0.0010, 0.0000],
            [730, 0.0014, 0.0005, 0.0000],
        ],
        dtype=np.float64,
    )
    wl = table[:, 0]
    xbar = np.interp(WAVELENGTHS_NM, wl, table[:, 1])
    ybar = np.interp(WAVELENGTHS_NM, wl, table[:, 2])
    zbar = np.interp(WAVELENGTHS_NM, wl, table[:, 3])
    return xbar, ybar, zbar


CMF_X, CMF_Y, CMF_Z = _build_cmfs()
_CMF_MATRIX = np.stack([CMF_X, CMF_Y, CMF_Z], axis=0)  # 3×N


def gaussian_spectrum(
    peak_nm: float,
    width_nm: float,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Unit-ish Gaussian lobe on the wavelength grid."""
    w = max(float(width_nm), 1.0)
    g = np.exp(-0.5 * ((WAVELENGTHS_NM - float(peak_nm)) / w) ** 2)
    peak = float(g.max()) or 1.0
    return (amplitude * g / peak).astype(np.float64)


def illuminant_d65() -> np.ndarray:
    """Smooth D65-like SPD on the grid (relative)."""
    # Piecewise approximation — cooler in blue, gentle roll toward red.
    spd = (
        0.55 * gaussian_spectrum(450, 55, amplitude=1.0)
        + 0.85 * gaussian_spectrum(560, 90, amplitude=1.0)
        + 0.70 * gaussian_spectrum(650, 70, amplitude=1.0)
    )
    return (spd / (spd.max() + 1e-12)).astype(np.float64)


def spectral_to_xyz(spectral: np.ndarray, *, illuminant: np.ndarray | None = None) -> np.ndarray:
    """Integrate spectrum → XYZ. ``spectral`` is …×N_WAVELENGTHS."""
    ill = illuminant_d65() if illuminant is None else np.asarray(illuminant, dtype=np.float64)
    weights = _CMF_MATRIX * ill[None, :]  # 3×N
    norm = float(np.dot(CMF_Y, ill)) or 1.0
    flat = spectral.reshape(-1, N_WAVELENGTHS)
    xyz = (flat @ weights.T) / norm
    return xyz.reshape(spectral.shape[:-1] + (3,)).astype(np.float32)


def xyz_to_spectral(xyz: np.ndarray, *, illuminant: np.ndarray | None = None) -> np.ndarray:
    """Non-negative least-squares-ish reconstruction of a spectrum from XYZ.

    Uses three CIE-weighted basis lobes under the illuminant, solves for
    coefficients, and clamps — enough for film-layer exposure estimates.
    """
    ill = illuminant_d65() if illuminant is None else np.asarray(illuminant, dtype=np.float64)
    # Basis: illuminant-modulated CMFs (pseudo-inverse path).
    basis = (_CMF_MATRIX * ill[None, :]).T  # N×3
    # Solve min ||B c - target_weights|| via normal equations on XYZ.
    # Map desired XYZ to coefficients through (M^T M)^{-1} M^T style on CMF·ill.
    m = _CMF_MATRIX * ill[None, :]  # 3×N
    gram = m @ m.T  # 3×3
    gram = gram + np.eye(3) * 1e-6
    inv = np.linalg.inv(gram)

    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
        # actually HxW luminance? expect HxWx3
    if arr.ndim == 2 and arr.shape[-1] != 3:
        raise ValueError("xyz_to_spectral expects …×3 XYZ")
    flat = arr.reshape(-1, 3)
    # Coefficients in CMF space, then project onto wavelength via basis rows.
    coef = flat @ inv.T  # …×3
    spec = np.clip(coef @ basis.T, 0.0, None)
    # Scale so reconstructed Y roughly matches input Y.
    recon = spectral_to_xyz(spec.reshape(arr.shape[:-1] + (N_WAVELENGTHS,)), illuminant=ill)
    y_in = np.maximum(flat[:, 1], 1e-8)
    y_out = np.maximum(recon.reshape(-1, 3)[:, 1], 1e-8)
    spec = spec * (y_in / y_out)[:, None]
    return spec.reshape(arr.shape[:-1] + (N_WAVELENGTHS,)).astype(np.float32)


def density_to_transmittance_spectral(density: np.ndarray) -> np.ndarray:
    return np.power(10.0, -np.maximum(density, 0.0)).astype(np.float32)


def transmittance_to_density_spectral(transmittance: np.ndarray) -> np.ndarray:
    return (-np.log10(np.maximum(transmittance, 1e-8))).astype(np.float32)


def dye_absorption_spectrum(peak_nm: float, width_nm: float, *, peak_density: float = 1.0) -> np.ndarray:
    """Spectral absorption coefficient for a dye (density per unit concentration)."""
    return gaussian_spectrum(peak_nm, width_nm, amplitude=peak_density).astype(np.float32)


def combine_dye_densities(
    concentrations: np.ndarray,
    dye_spectra: np.ndarray,
) -> np.ndarray:
    """``concentrations`` …×K, ``dye_spectra`` K×N → spectral density …×N."""
    c = np.asarray(concentrations, dtype=np.float32)
    dyes = np.asarray(dye_spectra, dtype=np.float32)
    flat = c.reshape(-1, dyes.shape[0])
    dens = flat @ dyes
    return dens.reshape(c.shape[:-1] + (dyes.shape[1],)).astype(np.float32)


def layer_exposures_from_spectral(
    spectral: np.ndarray,
    sensitivities: np.ndarray,
) -> np.ndarray:
    """Integrate spectral irradiance × layer sensitivity → …×K linear exposures."""
    s = np.asarray(spectral, dtype=np.float32)
    sens = np.asarray(sensitivities, dtype=np.float32)  # K×N
    # Normalize each sensitivity to unit integral so mid-grey stays stable.
    norms = np.maximum(sens.sum(axis=-1, keepdims=True), 1e-8)
    sens_n = sens / norms
    flat = s.reshape(-1, N_WAVELENGTHS)
    exp = flat @ sens_n.T
    return exp.reshape(s.shape[:-1] + (sens.shape[0],)).astype(np.float32)


def rgb_display_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """Linear XYZ → approx linear sRGB (clip), for previews."""
    # Bradford-ish sRGB matrix (D65).
    m = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float32,
    )
    flat = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(flat @ m.T, 0.0, None)
    return rgb.reshape(np.asarray(xyz).shape).astype(np.float32)


def encode_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    x = np.clip(linear_rgb, 0.0, None).astype(np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def profile_layer_spectra(raw_spectral: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build K×N sensitivity and dye matrices from a film/paper ``spectral`` block."""
    layers = raw_spectral.get("layers") or {}
    order = list(raw_spectral.get("layer_order") or ["cyan", "magenta", "yellow"])
    sens_rows = []
    dye_rows = []
    mask = dict(raw_spectral.get("mask") or {})
    for name in order:
        layer = layers.get(name) or {}
        sens_rows.append(
            gaussian_spectrum(
                float(layer.get("sensitivity_peak_nm", 550)),
                float(layer.get("sensitivity_width_nm", 40)),
                amplitude=float(layer.get("sensitivity_gain", 1.0)),
            )
        )
        dye_rows.append(
            dye_absorption_spectrum(
                float(layer.get("dye_peak_nm", 650)),
                float(layer.get("dye_width_nm", 50)),
                peak_density=float(layer.get("dye_peak_density", 1.0)),
            )
        )
    return (
        np.stack(sens_rows, axis=0).astype(np.float32),
        np.stack(dye_rows, axis=0).astype(np.float32),
        {k: float(v) for k, v in mask.items()},
    )


INSTANT_FILM_TYPES = frozenset({"instant_integral_color", "instant_integral_bw"})
COLOR_FILM_TYPES = frozenset({"color_negative", "color_slide", "color_ra4"})


def chemistry_mode_for_film_type(film_type: str) -> str:
    t = str(film_type or "bw").lower()
    if t in COLOR_FILM_TYPES:
        return "color"
    if t in INSTANT_FILM_TYPES:
        return "instant"
    return "bw"


def is_color_film_type(film_type: str) -> bool:
    return chemistry_mode_for_film_type(film_type) == "color"


def is_instant_film_type(film_type: str) -> bool:
    return chemistry_mode_for_film_type(film_type) == "instant"


def is_color_paper_type(paper_type: str) -> bool:
    return str(paper_type or "").lower() in {"color_ra4", "color"}
