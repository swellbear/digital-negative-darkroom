"""Interactive characteristic-curve editor — parametric handles, not freehand.

Dragging a handle does not invent an arbitrary tone curve. It nudges the same
darkroom controls the drawers already expose (dev time, N±, MG grade, base
exposure), then the engine re-samples the real film/paper response. That keeps
the preview honest: every shape you see is one the process can actually make.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import CurveReport, _contrast_index, reflectance_to_zone, zone_reflectance


# Screen-normalized drag sensitivities (dy > 0 = dragged toward top of plot).
_FILM_DEV_STOPS_PER_DY = 2.4          # density-up → longer development
_FILM_N_PER_DY = 1.8                  # steeper film → +contrast
_PRINT_STOPS_PER_DY = 2.2             # brighter print mid → shorter exposure
_PRINT_GRADE_PER_DY = 4.0             # steeper print → higher grade


def _norm_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> list[list[float]]:
    """Map data coords → 0..1 plot space (y=0 at bottom, matching D–logE)."""
    span_x = max(float(x1 - x0), 1e-9)
    span_y = max(float(y1 - y0), 1e-9)
    out: list[list[float]] = []
    for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
        nx = float(np.clip((float(x) - x0) / span_x, 0.0, 1.0))
        ny = float(np.clip((float(y) - y0) / span_y, 0.0, 1.0))
        out.append([round(nx, 5), round(ny, 5)])
    return out


def _downsample_polyline(points: list[list[float]], max_pts: int = 96) -> list[list[float]]:
    if len(points) <= max_pts:
        return points
    idx = np.linspace(0, len(points) - 1, max_pts).astype(int)
    return [points[int(i)] for i in idx]


def curve_overlay_payload(
    report: CurveReport,
    *,
    development_minutes: float,
    contrast: float,
    print_grade: float,
    print_exposure: float,
) -> dict[str, Any]:
    """JSON-ready film + print polylines and parametric drag handles."""
    log_e = np.asarray(report.log_e, dtype=np.float64)
    active = np.asarray(report.active_density, dtype=np.float64)
    base = np.asarray(report.base_density, dtype=np.float64)
    x0, x1 = float(log_e[0]), float(log_e[-1])
    y0 = 0.0
    y1 = float(max(np.max(active), np.max(base), 0.5) * 1.08)

    pct = report.scene_percentiles or {}
    mid_x = float(pct.get("p50", 0.5 * (x0 + x1)))
    hi_x = float(pct.get("p95", x0 + 0.78 * (x1 - x0)))
    mid_d = float(np.interp(mid_x, log_e, active))
    hi_d = float(np.interp(hi_x, log_e, active))

    film_poly = _downsample_polyline(_norm_xy(log_e, active, x0=x0, x1=x1, y0=y0, y1=y1))
    base_poly = _downsample_polyline(_norm_xy(log_e, base, x0=x0, x1=x1, y0=y0, y1=y1))

    film_handles = [
        {
            "id": "film_dev",
            "x": float(np.clip((mid_x - x0) / max(x1 - x0, 1e-9), 0.05, 0.95)),
            "y": float(np.clip((mid_d - y0) / max(y1 - y0, 1e-9), 0.05, 0.95)),
            "label": "Dev",
            "tip": "Drag up/down — development time (push / pull)",
        },
        {
            "id": "film_n",
            "x": float(np.clip((hi_x - x0) / max(x1 - x0, 1e-9), 0.05, 0.95)),
            "y": float(np.clip((hi_d - y0) / max(y1 - y0, 1e-9), 0.05, 0.95)),
            "label": "N±",
            "tip": "Drag up/down — contrast (N+ / N−)",
        },
    ]

    print_block: dict[str, Any] | None = None
    if report.print_reflectance is not None:
        refl = np.maximum(np.asarray(report.print_reflectance, dtype=np.float64), 1e-4)
        # Log reflectance so zone bands are even — matches the static plot.
        log_r = np.log10(refl)
        pr0 = float(np.min(log_r) - 0.05)
        pr1 = float(min(np.max(log_r) + 0.05, 0.0))
        print_poly = _downsample_polyline(
            _norm_xy(log_e, log_r, x0=x0, x1=x1, y0=pr0, y1=pr1)
        )
        mid_lr = float(np.interp(mid_x, log_e, log_r))
        hi_lr = float(np.interp(hi_x, log_e, log_r))
        print_handles = [
            {
                "id": "print_exp",
                "x": float(np.clip((mid_x - x0) / max(x1 - x0, 1e-9), 0.05, 0.95)),
                "y": float(np.clip((mid_lr - pr0) / max(pr1 - pr0, 1e-9), 0.05, 0.95)),
                "label": "Exp",
                "tip": "Drag up/down — base exposure (brighter print = less time)",
            },
        ]
        # MG grade handle is B&W multigrade only — RA-4 / color papers use CC
        # filtration in the Print drawer, not a grade slider.
        paper_type = str(report.stats.get("paper_type") or "").lower()
        if not paper_type:
            # Older reports may omit paper_type; treat missing as MG-capable.
            paper_type = "bw_multigrade"
        if paper_type.startswith("bw") or paper_type in {"bw_multigrade", "multigrade"}:
            print_handles.append(
                {
                    "id": "print_grade",
                    "x": float(np.clip((hi_x - x0) / max(x1 - x0, 1e-9), 0.05, 0.95)),
                    "y": float(np.clip((hi_lr - pr0) / max(pr1 - pr0, 1e-9), 0.05, 0.95)),
                    "label": "Grade",
                    "tip": "Drag up/down — MG grade / contrast filtration",
                }
            )
        zone_guides = []
        for z in range(0, 11, 2):
            zr = zone_reflectance(z)
            if zr <= 0:
                continue
            ly = np.log10(max(zr, 1e-4))
            if pr0 <= ly <= pr1:
                zone_guides.append(
                    {
                        "y": float(np.clip((ly - pr0) / max(pr1 - pr0, 1e-9), 0.0, 1.0)),
                        "label": str(z),
                    }
                )
        print_block = {
            "polyline": print_poly,
            "handles": print_handles,
            "zone_guides": zone_guides,
            "title": (
                f"{report.stats.get('paper', 'Paper')} · "
                f"grade {float(print_grade):.1f} · {float(print_exposure):g}s"
            ),
        }

    ci = float(report.stats.get("contrast_index") or _contrast_index(
        log_e, active, float(report.stats.get("base_plus_fog", active[0]))
    ))
    return {
        "ok": True,
        "film": {
            "polyline": film_poly,
            "base": base_poly,
            "handles": film_handles,
            "title": (
                f"{report.stats.get('film', 'Film')} · CI {ci:.2f} · "
                f"{float(development_minutes):g} min · N± {float(contrast):+.2f}"
            ),
        },
        "print": print_block,
        "settings": {
            "development_minutes": float(development_minutes),
            "contrast": float(contrast),
            "print_grade": float(print_grade),
            "print_exposure": float(print_exposure),
        },
        "stats": {
            "contrast_index": ci,
            "curve_source": report.stats.get("curve_source"),
            "shadow_zone": report.stats.get("shadow_zone"),
            "highlight_zone": report.stats.get("highlight_zone"),
        },
    }


def apply_curve_handle_edit(
    handle_id: str,
    *,
    dy: float,
    development_minutes: float,
    contrast: float,
    print_grade: float,
    print_exposure: float,
    minutes_min: float = 1.5,
    minutes_max: float = 24.0,
    contrast_min: float = -1.0,
    contrast_max: float = 1.0,
    grade_min: float = 0.0,
    grade_max: float = 5.0,
    exposure_min: float = 2.0,
    exposure_max: float = 64.0,
) -> dict[str, Any]:
    """Map a normalized handle drag (+dy = toward top of plot) onto UI settings."""
    hid = str(handle_id or "").strip().lower()
    dy = float(np.clip(dy, -0.85, 0.85))
    minutes = float(development_minutes)
    n_mod = float(contrast)
    grade = float(print_grade)
    seconds = float(print_exposure)
    note = ""

    if hid == "film_dev":
        # Up on the film curve = denser negative = more development.
        stops = dy * _FILM_DEV_STOPS_PER_DY
        minutes = float(np.clip(minutes * (2.0 ** stops), minutes_min, minutes_max))
        minutes = float(round(minutes * 4.0) / 4.0)  # 0.25 min steps
        note = f"Dev time → {minutes:g} min"
    elif hid == "film_n":
        n_mod = float(np.clip(n_mod + dy * _FILM_N_PER_DY, contrast_min, contrast_max))
        n_mod = float(round(n_mod * 20.0) / 20.0)  # 0.05 steps
        note = f"Contrast N± → {n_mod:+.2f}"
    elif hid == "print_exp":
        # Up on the print zone plot = brighter midtones = less enlarger time.
        stops = dy * _PRINT_STOPS_PER_DY
        seconds = float(np.clip(seconds * (2.0 ** (-stops)), exposure_min, exposure_max))
        seconds = float(round(seconds * 2.0) / 2.0)  # 0.5 s steps
        note = f"Base exposure → {seconds:g}s"
    elif hid == "print_grade":
        # Up = higher highlight separation on the system curve → harder grade.
        grade = float(np.clip(grade + dy * _PRINT_GRADE_PER_DY, grade_min, grade_max))
        grade = float(round(grade * 2.0) / 2.0)  # 0.5 steps
        note = f"MG grade → {grade:.1f}"
    else:
        return {
            "ok": False,
            "development_minutes": minutes,
            "contrast": n_mod,
            "print_grade": grade,
            "print_exposure": seconds,
            "message": f"Unknown handle `{handle_id}`.",
        }

    return {
        "ok": True,
        "development_minutes": minutes,
        "contrast": n_mod,
        "print_grade": grade,
        "print_exposure": seconds,
        "message": note,
        "handle": hid,
        "dy": dy,
    }
