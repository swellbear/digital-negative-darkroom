"""Print / enlarger stage: negative transmittance → paper response."""

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
    Grade mainly changes the slope; midtones stay near a printable pivot.
    `log_center` should be computed from the unscaled negative so exposure
    stops are not normalized away.
    """
    params = paper.grade_params(grade)
    gamma = float(params["gamma"])
    toe = float(params["toe"])
    shoulder = float(params["shoulder"])

    log_e = np.log10(np.maximum(exposure, 1e-6))
    if log_center is None:
        log_center = float(np.percentile(log_e, 55))
    x = log_e - float(log_center)

    # Contrast around a mid-gray pivot; gamma from filtration.
    # Negative x = denser negative / scene highlights → lower paper density.
    slope = 0.62 * gamma
    core = 0.22 + 0.55 * np.tanh(slope * x)

    # Soft toe (keep some shadow separation) and shoulder (open highlights)
    toe_lift = toe * 0.45 * (1.0 - _smoothstep(-1.0, 0.15, x))
    shoulder_open = shoulder * 0.55 * (1.0 - _smoothstep(-1.3, -0.05, x))
    shaped = np.clip(core - toe_lift - shoulder_open, 0.0, 1.0)

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

    overall_exposure: stops (+ opens / more light, - closes)
    grade: multigrade filtration 00–5
    contrast: extra contrast nudge around the selected grade
    """
    print_meta = dn.metadata.setdefault("print", {})
    exposure_stops = float(
        overall_exposure if overall_exposure is not None else print_meta.get("overall_exposure", 0.0)
    )
    grade_value = float(
        grade if grade is not None else print_meta.get("filtration", {}).get("grade", paper.default_grade)
    )
    contrast_nudge = float(contrast if contrast is not None else print_meta.get("contrast", 0.0))
    effective_grade = float(np.clip(grade_value + 0.75 * contrast_nudge, 0.0, 5.0))

    # Anchor the paper curve on the unscaled negative so +/- stops move density
    log_center = float(np.percentile(np.log10(np.maximum(transmittance, 1e-6)), 55))
    light = transmittance * (2.0**exposure_stops)
    print_density = paper_response(
        light, paper=paper, grade=effective_grade, log_center=log_center
    )
    reflectance = np.power(10.0, -print_density).astype(np.float32)

    # Preview reflectance; normalize gently so paper white isn't crushed on display
    white = float(np.power(10.0, -paper.dmin))
    preview = np.clip(reflectance / max(white, 1e-6), 0.0, 1.0)
    preview = np.power(preview, 0.75).astype(np.float32)

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
    if "print" not in dn.metadata["ui_state"].get("committed_stages", []):
        # development should already be committed; print is active
        pass
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
