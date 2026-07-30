"""Print / enlarger stage: negative transmittance → paper response.

Multigrade filtration includes approximate filter speed factors so that
changing grade without touching the timer shifts midtone density — the
familiar enlarger behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .digital_negative import DigitalNegative
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
    preview: np.ndarray


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


def print_negative(
    transmittance: np.ndarray,
    dn: DigitalNegative,
    paper: PaperProfile,
    *,
    overall_exposure: float | None = None,
    grade: float | None = None,
    contrast: float | None = None,
    commit: bool = True,
) -> PrintResult:
    """Expose paper through a developed negative.

    overall_exposure: timer stops (+ = more light = darker print)
    grade: multigrade filtration 00–5 (also applies filter speed factor)
    contrast: fine nudge between filter steps
    commit=False: live preview only — no history entry
    """
    print_meta = dn.metadata.setdefault("print", {})
    exposure_stops = float(
        overall_exposure if overall_exposure is not None else print_meta.get("overall_exposure", 0.0)
    )
    grade_value = float(
        grade if grade is not None else print_meta.get("filtration", {}).get("grade", paper.default_grade)
    )
    contrast_nudge = float(contrast if contrast is not None else print_meta.get("contrast", 0.0))
    effective_grade = float(np.clip(grade_value + 0.55 * contrast_nudge, 0.0, 5.0))

    # Anchor midtones on the unscaled negative (pre filter-speed / timer)
    log_t = np.log10(np.maximum(transmittance, 1e-6))
    log_center = float(np.percentile(log_t, 48))

    # Timer stops + MG filter speed (harder grades need more light for same midtone)
    speed = _filter_speed(effective_grade)
    light = transmittance * (2.0**exposure_stops) * speed
    print_density = paper_response(
        light, paper=paper, grade=effective_grade, log_center=log_center
    )
    reflectance = np.power(10.0, -print_density).astype(np.float32)

    white = float(np.power(10.0, -paper.dmin))
    preview = np.clip(reflectance / max(white, 1e-6), 0.0, 1.0)
    # Mild display gamma so paper Dmax reads as print black without crushing midtones.
    preview = np.power(preview, 0.78).astype(np.float32)

    print_meta.update(
        {
            "enabled": True,
            "paper_id": paper.id,
            "paper_name": paper.name,
            "filtration": {
                "type": "multigrade",
                "grade": grade_value,
                "values": {
                    "effective_grade": effective_grade,
                    "filter_speed": speed,
                },
            },
            "overall_exposure": exposure_stops,
            "contrast": contrast_nudge,
        }
    )
    dn.metadata.setdefault("ui_state", {})["current_stage"] = "print"

    if commit:
        dn.touch()
        dn.metadata.setdefault("history", []).append(
            {
                "op": "print",
                "paper_id": paper.id,
                "grade": grade_value,
                "overall_exposure": exposure_stops,
                "contrast": contrast_nudge,
                "filter_speed": speed,
            }
        )

    return PrintResult(
        print_density=print_density,
        reflectance=reflectance,
        preview=preview,
    )
