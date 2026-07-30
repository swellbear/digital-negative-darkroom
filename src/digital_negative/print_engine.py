"""Print / enlarger stage: negative transmittance → paper response.

Multigrade filtration includes approximate filter speed factors so that
changing grade without touching the timer shifts midtone density — the
familiar enlarger behavior.

Also: split-grade (soft+hard), test strips, flashing, dry-down, toning,
and easel borders — the techniques a fibre printer actually reaches for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .digital_negative import DigitalNegative
from .dodge_burn import REFERENCE_BASE_SECONDS, base_seconds_to_stops, stops_to_base_seconds
from .papers import PaperProfile

# Approximate relative speeds vs grade 2 (Ilford MG filter family, simplified).
# Multiplicative light factors before the paper curve — soft filters are "faster"
# (more effective light), hard filters need more timer for the same midtone.
# v4: slightly closer to published MG exposure-factor shape (~½ stop soft→2, ~1 stop 2→5).
MG_FILTER_SPEED = {
    0.0: 1.42,
    0.5: 1.32,
    1.0: 1.20,
    1.5: 1.10,
    2.0: 1.00,
    2.5: 0.90,
    3.0: 0.78,
    3.5: 0.68,
    4.0: 0.58,
    4.5: 0.50,
    5.0: 0.42,
}

TONE_LABELS: list[tuple[str, str]] = [
    ("None", "none"),
    ("Selenium", "selenium"),
    ("Sepia", "sepia"),
]


def _filter_speed(grade: float) -> float:
    keys = sorted(MG_FILTER_SPEED)
    if grade <= keys[0]:
        return MG_FILTER_SPEED[keys[0]]
    if grade >= keys[-1]:
        return MG_FILTER_SPEED[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= grade <= hi:
            t = (grade - lo) / (hi - lo) if hi != lo else 0.0
            return (1.0 - t) * MG_FILTER_SPEED[lo] + t * MG_FILTER_SPEED[hi]
    return 1.0


@dataclass
class PrintResult:
    print_density: np.ndarray
    reflectance: np.ndarray
    preview: np.ndarray  # float HxW or HxWx3 after toning/borders


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def paper_response(
    exposure: np.ndarray,
    *,
    paper: PaperProfile,
    grade: float,
    log_center: float | None = None,
) -> np.ndarray:
    """Map relative print exposure to paper density (enlarger-like)."""
    params = paper.grade_params(grade)
    gamma = float(params["gamma"])
    toe = float(params["toe"])
    shoulder = float(params["shoulder"])

    log_e = np.log10(np.maximum(exposure, 1e-6))
    if log_center is None:
        log_center = float(np.percentile(log_e, 50))
    x = log_e - float(log_center)

    # Grade span: 00 long & soft; 5 short & steep — like swapping MG filters.
    slope = 0.24 + 1.05 * (gamma / 1.25)
    hard = np.clip((grade - 2.0) / 3.0, 0.0, 1.0)
    soft = 1.0 - hard
    # Hard grades compress highlights; soft grades open them and lengthen the toe.
    shadow_gain = 1.0 + 0.85 * hard
    highlight_gain = 1.0 + 0.42 * soft
    x_shaped = np.where(x >= 0.0, x * shadow_gain, x * highlight_gain)

    core = 0.24 + 0.56 * np.tanh(slope * x_shaped)
    toe_keep = toe * (0.80 + 0.95 * soft)
    shoulder_keep = shoulder * (0.50 + 1.15 * soft)
    toe_lift = toe_keep * 0.65 * (1.0 - _smoothstep(-1.35, 0.08, x))
    shoulder_open = shoulder_keep * 0.75 * (1.0 - _smoothstep(-1.50, 0.05, x))
    hard_block = hard * 0.20 * _smoothstep(0.08, 1.20, x)
    shaped = np.clip(core - toe_lift - shoulder_open + hard_block, 0.0, 1.0)

    dens = paper.dmin + (paper.dmax - paper.dmin) * shaped
    return dens.astype(np.float32)


def _resize_stops(local_stops: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if local_stops is None:
        return None
    ls = np.asarray(local_stops, dtype=np.float32)
    if ls.shape == shape:
        return ls
    try:
        from PIL import Image

        im = Image.fromarray(ls, mode="F")
        im = im.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
        return np.asarray(im).astype(np.float32)
    except Exception:
        return np.zeros(shape, dtype=np.float32)


def _light_map(
    transmittance: np.ndarray,
    *,
    exposure_stops: float,
    grade: float,
    local_stops: np.ndarray | None,
    flash_stops: float = 0.0,
) -> np.ndarray:
    speed = _filter_speed(grade)
    light = transmittance * (2.0**exposure_stops) * speed
    ls = _resize_stops(local_stops, light.shape)
    if ls is not None:
        light = light * np.power(2.0, np.clip(ls, -3.0, 3.0))
    # Flashing: uniform pre-exposure (no negative), in the same light units.
    if flash_stops > 0.0:
        # Reference flash relative to midtone transmittance ~0.3 at grade speed.
        flash = (2.0 ** (flash_stops - 4.0)) * speed
        light = light + flash
    return light.astype(np.float32)


def _combine_densities(d1: np.ndarray, d2: np.ndarray, *, dmin: float, dmax: float) -> np.ndarray:
    """Sequential exposures via unexposed-silver product."""
    span = max(dmax - dmin, 1e-6)
    u1 = np.clip(1.0 - (d1 - dmin) / span, 0.0, 1.0)
    u2 = np.clip(1.0 - (d2 - dmin) / span, 0.0, 1.0)
    return (dmin + span * (1.0 - u1 * u2)).astype(np.float32)


def apply_test_strip_bands(
    light: np.ndarray,
    *,
    bands: int = 5,
    stop_step: float = 0.5,
    axis: int = 1,
) -> np.ndarray:
    """Multiply light by stepped exposure factors across the frame."""
    bands = int(np.clip(bands, 2, 12))
    out = np.asarray(light, dtype=np.float32).copy()
    n = out.shape[axis]
    # Centre band ~ 0 stops; neighbours step up/down.
    mid = (bands - 1) / 2.0
    for i in range(bands):
        lo = int(round(i * n / bands))
        hi = int(round((i + 1) * n / bands))
        stops = (i - mid) * float(stop_step)
        factor = float(2.0**stops)
        if axis == 1:
            out[:, lo:hi] *= factor
        else:
            out[lo:hi, :] *= factor
    return out


def apply_dry_down(reflectance: np.ndarray, percent: float) -> np.ndarray:
    """Prints dry darker — reduce reflectance by a calibrated percentage."""
    p = float(np.clip(percent, 0.0, 40.0)) / 100.0
    if p <= 1e-6:
        return reflectance
    return (np.asarray(reflectance, dtype=np.float32) * (1.0 - p)).astype(np.float32)


def apply_tone(preview: np.ndarray, tone: str | None) -> np.ndarray:
    """Map a gray preview into a toned RGB float image in [0, 1]."""
    g = np.asarray(preview, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=-1)
    name = str(tone or "none").lower()
    if name in ("", "none"):
        return g
    # Shadow→highlight colour ramps (display-referred).
    if name == "selenium":
        shadow = np.array([0.06, 0.07, 0.10], dtype=np.float32)
        mid = np.array([0.42, 0.44, 0.50], dtype=np.float32)
        highlight = np.array([0.96, 0.96, 0.97], dtype=np.float32)
        # Slight Dmax deepen
        g = np.power(np.clip(g, 0, 1), 1.06)
    elif name == "sepia":
        shadow = np.array([0.10, 0.06, 0.03], dtype=np.float32)
        mid = np.array([0.62, 0.48, 0.30], dtype=np.float32)
        highlight = np.array([0.98, 0.94, 0.88], dtype=np.float32)
    else:
        return g
    t = np.clip(g, 0.0, 1.0)[..., None]
    # Two-segment lerp: shadows→mid, mid→highlights
    lo = shadow + (mid - shadow) * np.clip(t / 0.45, 0, 1)
    hi = mid + (highlight - mid) * np.clip((t - 0.45) / 0.55, 0, 1)
    return np.where(t < 0.45, lo, hi).astype(np.float32)


def apply_border(preview: np.ndarray, border_frac: float) -> np.ndarray:
    """Unexposed white easel border around the print."""
    f = float(np.clip(border_frac, 0.0, 0.25))
    if f <= 1e-6:
        return preview
    img = np.asarray(preview, dtype=np.float32)
    h, w = img.shape[:2]
    bh, bw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    if img.ndim == 2:
        out = np.ones((h + 2 * bh, w + 2 * bw), dtype=np.float32)
        out[bh : bh + h, bw : bw + w] = img
    else:
        out = np.ones((h + 2 * bh, w + 2 * bw, img.shape[2]), dtype=np.float32)
        out[bh : bh + h, bw : bw + w] = img
    return out


def _resolve_timer(
    print_meta: dict[str, Any],
    *,
    overall_exposure: float | None,
    base_exposure_seconds: float | None,
) -> tuple[float, float]:
    if base_exposure_seconds is not None:
        base_seconds = float(base_exposure_seconds)
        exposure_stops = base_seconds_to_stops(base_seconds)
    elif overall_exposure is not None:
        exposure_stops = float(overall_exposure)
        base_seconds = stops_to_base_seconds(exposure_stops)
    else:
        if "base_exposure_seconds" in print_meta:
            base_seconds = float(print_meta.get("base_exposure_seconds", REFERENCE_BASE_SECONDS))
            exposure_stops = base_seconds_to_stops(base_seconds)
        else:
            exposure_stops = float(print_meta.get("overall_exposure", 0.0))
            base_seconds = stops_to_base_seconds(exposure_stops)
    return base_seconds, exposure_stops


def print_negative(
    transmittance: np.ndarray,
    dn: DigitalNegative,
    paper: PaperProfile,
    *,
    overall_exposure: float | None = None,
    base_exposure_seconds: float | None = None,
    grade: float | None = None,
    contrast: float | None = None,
    local_stops: np.ndarray | None = None,
    # —— Technique depth ——
    split_grade: bool = False,
    soft_grade: float = 0.0,
    hard_grade: float = 5.0,
    soft_exposure_seconds: float | None = None,
    hard_exposure_seconds: float | None = None,
    local_stops_soft: np.ndarray | None = None,
    local_stops_hard: np.ndarray | None = None,
    test_strips: bool = False,
    test_strip_bands: int = 5,
    test_strip_stops: float = 0.5,
    flash_stops: float = 0.0,
    dry_down_percent: float = 0.0,
    tone: str = "none",
    border_frac: float = 0.0,
    commit: bool = True,
) -> PrintResult:
    """Expose paper through a developed negative.

    overall_exposure: timer stops (+ = more light = darker print)
    base_exposure_seconds: enlarger timer in seconds (preferred UI unit);
        when set, overrides overall_exposure via log2(seconds / 8s reference)
    grade: multigrade filtration 00–5 (also applies filter speed factor)
    contrast: fine nudge between filter steps
    local_stops: optional HxW map of extra exposure stops (burn +, dodge −)
    commit=False: live preview only — no history entry
    """
    print_meta = dn.metadata.setdefault("print", {})
    base_seconds, exposure_stops = _resolve_timer(
        print_meta,
        overall_exposure=overall_exposure,
        base_exposure_seconds=base_exposure_seconds,
    )

    grade_value = float(
        grade if grade is not None else print_meta.get("filtration", {}).get("grade", paper.default_grade)
    )
    contrast_nudge = float(contrast if contrast is not None else print_meta.get("contrast", 0.0))
    effective_grade = float(np.clip(grade_value + 0.55 * contrast_nudge, 0.0, 5.0))

    log_t = np.log10(np.maximum(transmittance, 1e-6))
    log_center = float(np.percentile(log_t, 48))

    split = bool(split_grade)
    if split:
        soft_g = float(np.clip(soft_grade, 0.0, 5.0))
        hard_g = float(np.clip(hard_grade, 0.0, 5.0))
        soft_s = float(
            soft_exposure_seconds if soft_exposure_seconds is not None else base_seconds * 0.55
        )
        hard_s = float(
            hard_exposure_seconds if hard_exposure_seconds is not None else base_seconds * 0.45
        )
        soft_stops = base_seconds_to_stops(soft_s)
        hard_stops = base_seconds_to_stops(hard_s)
        ls_soft = local_stops_soft if local_stops_soft is not None else local_stops
        ls_hard = local_stops_hard if local_stops_hard is not None else local_stops
        light_soft = _light_map(
            transmittance,
            exposure_stops=soft_stops,
            grade=soft_g,
            local_stops=ls_soft,
            flash_stops=float(flash_stops),
        )
        light_hard = _light_map(
            transmittance,
            exposure_stops=hard_stops,
            grade=hard_g,
            local_stops=ls_hard,
            flash_stops=0.0,  # flash once with the soft pass
        )
        if test_strips:
            light_soft = apply_test_strip_bands(
                light_soft, bands=test_strip_bands, stop_step=test_strip_stops
            )
            light_hard = apply_test_strip_bands(
                light_hard, bands=test_strip_bands, stop_step=test_strip_stops
            )
        dens_soft = paper_response(light_soft, paper=paper, grade=soft_g, log_center=log_center)
        dens_hard = paper_response(light_hard, paper=paper, grade=hard_g, log_center=log_center)
        print_density = _combine_densities(
            dens_soft, dens_hard, dmin=paper.dmin, dmax=paper.dmax
        )
        speed = 0.5 * (_filter_speed(soft_g) + _filter_speed(hard_g))
        filtration = {
            "type": "split_grade",
            "grade": grade_value,
            "values": {
                "soft_grade": soft_g,
                "hard_grade": hard_g,
                "soft_seconds": soft_s,
                "hard_seconds": hard_s,
                "filter_speed": speed,
                "effective_grade": effective_grade,
            },
        }
    else:
        light = _light_map(
            transmittance,
            exposure_stops=exposure_stops,
            grade=effective_grade,
            local_stops=local_stops,
            flash_stops=float(flash_stops),
        )
        if test_strips:
            light = apply_test_strip_bands(
                light, bands=test_strip_bands, stop_step=test_strip_stops
            )
        print_density = paper_response(
            light, paper=paper, grade=effective_grade, log_center=log_center
        )
        speed = _filter_speed(effective_grade)
        filtration = {
            "type": "multigrade",
            "grade": grade_value,
            "values": {
                "effective_grade": effective_grade,
                "filter_speed": speed,
            },
        }

    reflectance = np.power(10.0, -print_density).astype(np.float32)
    reflectance = apply_dry_down(reflectance, dry_down_percent)

    white = float(np.power(10.0, -paper.dmin))
    preview = np.clip(reflectance / max(white, 1e-6), 0.0, 1.0)
    preview = np.power(preview, 0.78).astype(np.float32)
    preview = apply_tone(preview, tone)
    preview = apply_border(preview, border_frac)

    strokes = print_meta.get("dodge_burn") or []
    print_meta.update(
        {
            "enabled": True,
            "paper_id": paper.id,
            "paper_name": paper.name,
            "filtration": filtration,
            "base_exposure_seconds": round(base_seconds, 3),
            "overall_exposure": exposure_stops,
            "contrast": contrast_nudge,
            "dodge_burn": strokes,
            "flash_stops": float(flash_stops),
            "dry_down_percent": float(dry_down_percent),
            "tone": str(tone or "none"),
            "border_frac": float(border_frac),
            "test_strips": bool(test_strips),
            "split_grade": bool(split),
        }
    )
    dn.metadata.setdefault("ui_state", {})["current_stage"] = "print"

    if commit:
        dn.touch()
        hist = {
            "op": "print",
            "paper_id": paper.id,
            "grade": grade_value,
            "base_exposure_seconds": round(base_seconds, 3),
            "overall_exposure": exposure_stops,
            "contrast": contrast_nudge,
            "filter_speed": speed,
            "flash_stops": float(flash_stops),
            "dry_down_percent": float(dry_down_percent),
            "tone": str(tone or "none"),
            "border_frac": float(border_frac),
            "split_grade": bool(split),
            "test_strips": bool(test_strips),
        }
        if split:
            hist["soft_grade"] = float(soft_grade)
            hist["hard_grade"] = float(hard_grade)
        if strokes:
            hist["dodge_burn"] = list(strokes)
        dn.metadata.setdefault("history", []).append(hist)

    return PrintResult(
        print_density=print_density,
        reflectance=reflectance,
        preview=preview,
    )
