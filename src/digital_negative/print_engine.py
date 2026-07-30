"""Print / enlarger stage: negative transmittance → paper response.

Goal: exposure (stops) and multigrade grade should feel closer to standing
at an enlarger than a generic contrast slider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .digital_negative import DigitalNegative
from .papers import PaperProfile


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
    """Map relative print exposure (linear light through negative) to paper density.

    Higher light through the negative → higher paper density → darker print.
    Soft grades (00–1) keep a long toe and open shadows; hard grades (4–5)
    steepen the straight line and block up sooner — closer to MG filtration.
    """
    params = paper.grade_params(grade)
    gamma = float(params["gamma"])
    toe = float(params["toe"])
    shoulder = float(params["shoulder"])

    log_e = np.log10(np.maximum(exposure, 1e-6))
    if log_center is None:
        log_center = float(np.percentile(log_e, 50))
    x = log_e - float(log_center)

    # Grade controls slope more aggressively than the early spike
    # Soft ~0.4–0.7 effective; hard ~1.6–2.4
    slope = 0.35 + 0.85 * (gamma / 1.25)

    # Asymmetric curve: hard grades crush shadows (positive x) faster
    hard = np.clip((grade - 2.0) / 3.0, 0.0, 1.0)
    soft = 1.0 - hard
    shadow_gain = 1.0 + 0.55 * hard
    highlight_gain = 1.0 + 0.25 * soft

    x_shaped = np.where(x >= 0.0, x * shadow_gain, x * highlight_gain)
    core = 0.28 + 0.52 * np.tanh(slope * x_shaped)

    # Soft filtration preserves toe; hard filtration shortens it
    toe_keep = toe * (0.7 + 0.8 * soft)
    shoulder_keep = shoulder * (0.5 + 0.9 * soft)
    toe_lift = toe_keep * 0.55 * (1.0 - _smoothstep(-1.2, 0.05, x))
    shoulder_open = shoulder_keep * 0.65 * (1.0 - _smoothstep(-1.4, 0.0, x))
    # Hard grades add a little shoulder compression (print blacks sooner)
    hard_block = hard * 0.12 * _smoothstep(0.15, 1.2, x)
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
) -> PrintResult:
    """Expose paper through a developed negative.

    overall_exposure: stops (+ more enlarger light / darker print, - less light)
    grade: multigrade filtration 00–5
    contrast: fine nudge around the selected grade (like dialling between filters)
    """
    print_meta = dn.metadata.setdefault("print", {})
    exposure_stops = float(
        overall_exposure if overall_exposure is not None else print_meta.get("overall_exposure", 0.0)
    )
    grade_value = float(
        grade if grade is not None else print_meta.get("filtration", {}).get("grade", paper.default_grade)
    )
    contrast_nudge = float(contrast if contrast is not None else print_meta.get("contrast", 0.0))
    effective_grade = float(np.clip(grade_value + 0.6 * contrast_nudge, 0.0, 5.0))

    # Anchor on unscaled negative so +/- stops are not normalized away.
    # Use a slightly shadow-biased pivot — matches how printers often place
    # important midtones a touch above pure average.
    log_t = np.log10(np.maximum(transmittance, 1e-6))
    log_center = float(np.percentile(log_t, 48))
    light = transmittance * (2.0**exposure_stops)
    print_density = paper_response(
        light, paper=paper, grade=effective_grade, log_center=log_center
    )
    reflectance = np.power(10.0, -print_density).astype(np.float32)

    white = float(np.power(10.0, -paper.dmin))
    preview = np.clip(reflectance / max(white, 1e-6), 0.0, 1.0)
    # Slight paper-base warmth not applied (B&W); mild display gamma only
    preview = np.power(preview, 0.80).astype(np.float32)

    print_meta.update(
        {
            "enabled": True,
            "paper_id": paper.id,
            "paper_name": paper.name,
            "filtration": {
                "type": "multigrade",
                "grade": grade_value,
                "values": {"effective_grade": effective_grade},
            },
            "overall_exposure": exposure_stops,
            "contrast": contrast_nudge,
        }
    )
    dn.metadata["ui_state"]["current_stage"] = "print"
    dn.touch()
    dn.metadata.setdefault("history", []).append(
        {
            "op": "print",
            "paper_id": paper.id,
            "grade": grade_value,
            "overall_exposure": exposure_stops,
            "contrast": contrast_nudge,
        }
    )

    return PrintResult(
        print_density=print_density,
        reflectance=reflectance,
        preview=preview,
    )
