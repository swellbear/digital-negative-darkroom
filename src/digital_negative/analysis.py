"""Calibration instruments: sample the curves actually in play and report
where a given scene lands on them.

The develop and print stages each shape tone, but the photographer only ever
sees the end of that chain. These helpers expose the intermediate response so
"the sky feels flat" can become "the sky sits above the shoulder at D 1.9".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .curves import FilmProfile, modify_curve
from .development import linear_to_relative_log_exposure
from .digital_negative import DigitalNegative
from .papers import PaperProfile
from .print_engine import REFERENCE_LOG_TRANSMITTANCE, _filter_speed, paper_response
from .spectral import is_color_paper_type, is_instant_film_type

# Zone system anchors: Zone V is the mid grey the meter aims for, and each
# zone is one stop. Print reflectance is what the eye judges, so zones are
# placed on reflectance rather than density.
ZONE_COUNT = 11  # 0 through X


def zone_reflectance(zone: float) -> float:
    """Reflectance for a Zone value; Zone V ~ 18% mid grey, 1 stop per zone.

    Deliberately unclamped: zones past roughly VII exceed what paper can
    return, and callers filter them out rather than stacking them all on
    paper white.
    """
    return float(0.18 * (2.0 ** (float(zone) - 5.0)))


def reflectance_to_zone(reflectance: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`zone_reflectance`, clamped to the printable range."""
    r = np.maximum(np.asarray(reflectance, dtype=np.float64), 1e-6)
    zone = 5.0 + np.log2(r / 0.18)
    return np.clip(zone, 0.0, float(ZONE_COUNT - 1))


def _as_mono_reflectance(reflectance: np.ndarray) -> np.ndarray:
    """Collapse print reflectance to luminance for Zone tools.

    - Mono HxW → as-is
    - RGB HxWx3 → Rec.709 luma
    - Spectral HxWxN (RA-4) → CIE Y under D65 (not the first three bands)
    """
    r = np.asarray(reflectance, dtype=np.float64)
    if r.ndim < 3:
        return r
    n = int(r.shape[-1])
    if n >= 3:
        from .spectral import N_WAVELENGTHS, spectral_to_xyz

        if n == N_WAVELENGTHS:
            xyz = spectral_to_xyz(r.astype(np.float32))
            return np.asarray(xyz[..., 1], dtype=np.float64)
        if n == 3:
            return (
                0.2126 * r[..., 0] + 0.7152 * r[..., 1] + 0.0722 * r[..., 2]
            )
        # Unexpected channel count — mean of first three is a last resort.
        return r[..., :3].mean(axis=-1)
    return r


def scene_log_exposure(dn: DigitalNegative, *, mid_log_e: float = 2.2) -> np.ndarray:
    """Scene relative log-E, anchored exactly as :func:`develop` does."""
    return linear_to_relative_log_exposure(dn.to_luminance(), mid_log_e=mid_log_e)


def _subsample(arr: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if flat.size <= max_samples:
        return flat
    step = int(np.ceil(flat.size / max_samples))
    return flat[::step]


@dataclass
class CurveReport:
    """Numbers a printer can act on, alongside the plotted shapes."""

    log_e: np.ndarray            # grid the curves are sampled on
    base_density: np.ndarray     # datasheet curve
    active_density: np.ndarray   # after time / chemistry / N+-
    scene_log_e: np.ndarray      # subsampled scene samples
    scene_percentiles: dict[str, float]
    print_density: np.ndarray | None = None      # system curve, over log_e
    print_reflectance: np.ndarray | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def _contrast_index(log_e: np.ndarray, density: np.ndarray, fog: float) -> float:
    """Gradient over the straight-line portion — the usual CI shorthand."""
    above = density - fog
    span = float(above[-1] - above[0])
    if span <= 1e-6:
        return 0.0
    lo_t, hi_t = 0.25, 0.75
    lo = float(above[0] + lo_t * span)
    hi = float(above[0] + hi_t * span)
    mask = (above >= lo) & (above <= hi)
    if mask.sum() < 2:
        mask = np.ones_like(above, dtype=bool)
    x = log_e[mask]
    y = density[mask]
    if x.size < 2 or float(x[-1] - x[0]) <= 1e-6:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _ra4_system_reflectance(
    neg_density: np.ndarray,
    paper: PaperProfile,
    *,
    base_exposure_seconds: float,
    print_contrast: float = 0.0,
) -> np.ndarray:
    """1D RA-4 stand-in: neg D → T → logistic paper reflectance (matches live Exp)."""
    from .color_print import REFERENCE_RA4_LOG_EXPOSURE

    spectral = paper.raw.get("spectral") or {}
    toe = float(spectral.get("toe", 0.35))
    shoulder = float(spectral.get("shoulder", 0.35))
    dmin = float(paper.dmin)
    dmax = float(paper.dmax)
    t = np.power(10.0, -np.asarray(neg_density, dtype=np.float64))
    seconds = max(float(base_exposure_seconds), 0.05)
    # Timer scale identical to print_color_negative (8s → 1.0).
    log_paper = np.log10(np.maximum(t * (seconds / 8.0), 1e-8))
    center = float(REFERENCE_RA4_LOG_EXPOSURE)
    x = (log_paper - center) * (1.0 + 0.45 * float(print_contrast))
    resp = 1.0 / (1.0 + np.exp(-x / max(toe, 0.05)))
    resp = np.clip(resp, 0.0, 1.0)
    resp = resp / (1.0 + shoulder * resp)
    dens = dmin + (dmax - dmin) * resp
    return np.power(10.0, -dens).astype(np.float64)


def _instant_layer_density(
    profile: FilmProfile,
    log_e: np.ndarray,
    *,
    development_minutes: float | None,
    contrast_modifier: float,
    process_temp_c: float = 21.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample Instant positive reflection density (mean RGB) vs log-E."""
    from .instant_process import (
        _default_layer_points,
        _interp_density,
        _temp_morph_scales,
        auto_process_minutes,
    )

    instant = dict(profile.raw.get("instant") or {})
    layers_block = instant.get("layers") or {}
    temp_c = float(process_temp_c)
    contrast_scale, dmax_scale, _time_scale = _temp_morph_scales(instant, temp_c)
    minutes = (
        float(development_minutes)
        if development_minutes is not None
        else auto_process_minutes(profile, temp_c)
    )
    normal_auto = auto_process_minutes(profile, temp_c)
    time_ratio = float(np.clip(minutes / max(normal_auto, 1e-6), 0.45, 2.2))
    contrast_scale *= 0.92 + 0.08 * time_ratio
    dmax_scale *= 0.90 + 0.10 * time_ratio
    n_mod = float(np.clip(contrast_modifier, -1.5, 1.5))
    pivot = 2.2
    log_adj = pivot + (np.asarray(log_e, dtype=np.float64) - pivot) * (
        1.0 + 0.35 * n_mod
    ) * contrast_scale

    dens_ch = []
    ceilings = []
    for name in ("red", "green", "blue"):
        pts = layers_block.get(name) or _default_layer_points(name)
        d = _interp_density(pts, log_adj.astype(np.float32)).astype(np.float64)
        dmin_curve = float(min(p[1] for p in pts))
        dmax_curve = float(max(p[1] for p in pts))
        dens_ch.append(dmin_curve + (d - dmin_curve) * dmax_scale)
        ceilings.append(dmin_curve + (dmax_curve - dmin_curve) * dmax_scale)
    density_neg = np.stack(dens_ch, axis=-1)
    ceiling = np.asarray(ceilings, dtype=np.float64).reshape(1, 3)
    density_rgb = np.clip(ceiling - density_neg, 0.0, None)
    active = (
        0.2126 * density_rgb[..., 0]
        + 0.7152 * density_rgb[..., 1]
        + 0.0722 * density_rgb[..., 2]
    )
    # Base = authored layers at N / normal time (no user morph).
    base_ch = []
    base_ceil = []
    for name in ("red", "green", "blue"):
        pts = layers_block.get(name) or _default_layer_points(name)
        d = _interp_density(pts, np.asarray(log_e, dtype=np.float32)).astype(np.float64)
        dmin_curve = float(min(p[1] for p in pts))
        dmax_curve = float(max(p[1] for p in pts))
        base_ch.append(d)
        base_ceil.append(dmax_curve)
    base_neg = np.stack(base_ch, axis=-1)
    base_ceiling = np.asarray(base_ceil, dtype=np.float64).reshape(1, 3)
    base_rgb = np.clip(base_ceiling - base_neg, 0.0, None)
    base = (
        0.2126 * base_rgb[..., 0]
        + 0.7152 * base_rgb[..., 1]
        + 0.0722 * base_rgb[..., 2]
    )
    meta = {
        "curve_source": "instant_layers",
        "process_minutes": minutes,
        "process_temp_c": temp_c,
        "base_plus_fog": float(np.min(base)),
    }
    return active.astype(np.float64), base.astype(np.float64), meta


def build_curve_report(
    dn: DigitalNegative | None,
    profile: FilmProfile,
    *,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
    developer_id: str = "standard",
    development_minutes: float | None = None,
    paper: PaperProfile | None = None,
    grade: float | None = None,
    base_exposure_seconds: float | None = None,
    print_contrast: float = 0.0,
    process_temp_c: float = 21.0,
    mid_log_e: float = 2.2,
    samples: int = 320,
) -> CurveReport:
    """Sample the film curve in play, plus the film→paper system curve.

    Chemistry-aware:
    - B&W: master H&D + MG ``paper_response``
    - Color: master morph proxy + RA-4 logistic system curve (grade unused)
    - Instant: integral layer H&D (no enlarger paper panel)
    """
    instant = is_instant_film_type(profile.type)

    if instant:
        # Instant packs author RGB print layers — not the master B&W morph.
        lo = 0.5
        hi = 4.5
        instant_block = profile.raw.get("instant") or {}
        for name in ("red", "green", "blue"):
            pts = (instant_block.get("layers") or {}).get(name) or []
            if pts:
                lo = min(lo, float(pts[0][0]))
                hi = max(hi, float(pts[-1][0]))
        grid = np.linspace(lo, hi, int(samples))
        active_d, base_d, inst_meta = _instant_layer_density(
            profile,
            grid,
            development_minutes=development_minutes,
            contrast_modifier=contrast_modifier,
            process_temp_c=process_temp_c,
        )
        fog = float(inst_meta.get("base_plus_fog", np.min(base_d)))
        curve_source = "instant_layers"
        active_profile_fog = fog
    else:
        active = modify_curve(
            profile,
            relative_time=relative_time,
            contrast_modifier=contrast_modifier,
            developer_id=developer_id,
            development_minutes=development_minutes,
        )
        lo = float(min(profile.log_exposure[0], active.log_exposure[0]))
        hi = float(max(profile.log_exposure[-1], active.log_exposure[-1]))
        grid = np.linspace(lo, hi, int(samples))
        base_d = profile.density_from_log_exposure(grid).astype(np.float64)
        active_d = active.density_from_log_exposure(grid).astype(np.float64)
        fog = float(active.base_plus_fog)
        active_profile_fog = fog
        curve_source = (active.raw.get("_last_curve_meta") or {}).get(
            "curve_source", "morph"
        )

    scene: np.ndarray = np.asarray([], dtype=np.float64)
    pct: dict[str, float] = {}
    if dn is not None:
        scene = _subsample(scene_log_exposure(dn, mid_log_e=mid_log_e)).astype(np.float64)
        if scene.size:
            for name, q in (("p1", 1), ("p5", 5), ("p50", 50), ("p95", 95), ("p99", 99)):
                pct[name] = float(np.percentile(scene, q))

    stats: dict[str, Any] = {
        "film": profile.name,
        "base_plus_fog": float(active_profile_fog),
        "d_min": float(np.min(active_d)),
        "d_max": float(np.max(active_d)),
        "contrast_index": _contrast_index(grid, active_d, float(active_profile_fog)),
        "curve_source": curve_source,
        "chemistry_mode": "instant" if instant else (
            "color" if (paper is not None and is_color_paper_type(paper.type)) else "bw"
        ),
    }
    if pct:
        stats["scene_stops"] = (pct["p99"] - pct["p1"]) / 0.30103
        stats["shadow_density"] = float(np.interp(pct["p5"], grid, active_d))
        stats["highlight_density"] = float(np.interp(pct["p95"], grid, active_d))

    print_d = None
    print_r = None
    if paper is not None and not instant:
        transmittance = np.power(10.0, -active_d)
        if is_color_paper_type(paper.type):
            seconds = float(
                8.0 if base_exposure_seconds is None else base_exposure_seconds
            )
            print_r = _ra4_system_reflectance(
                active_d,
                paper,
                base_exposure_seconds=seconds,
                print_contrast=float(print_contrast),
            )
            print_d = (-np.log10(np.maximum(print_r, 1e-6))).astype(np.float64)
            stats["paper"] = paper.name
            stats["print_process"] = "ra4"
            stats["print_contrast"] = float(print_contrast)
            stats["base_exposure_seconds"] = seconds
        else:
            eff_grade = float(paper.default_grade if grade is None else grade)
            stops = 0.0
            if base_exposure_seconds is not None:
                stops = float(np.log2(max(float(base_exposure_seconds), 1e-6) / 8.0))
            light = transmittance * (2.0**stops) * _filter_speed(eff_grade)
            centre = float(REFERENCE_LOG_TRANSMITTANCE)
            print_d = paper_response(
                light, paper=paper, grade=eff_grade, log_center=centre
            ).astype(np.float64)
            print_r = np.power(10.0, -print_d)
            stats["paper"] = paper.name
            stats["grade"] = eff_grade
            stats["print_process"] = "bw_mg"
        if scene.size and print_r is not None:
            sh = np.interp(pct["p5"], grid, print_r)
            hl = np.interp(pct["p95"], grid, print_r)
            stats["shadow_zone"] = float(reflectance_to_zone(sh))
            stats["highlight_zone"] = float(reflectance_to_zone(hl))

    return CurveReport(
        log_e=grid,
        base_density=base_d,
        active_density=active_d,
        scene_log_e=scene,
        scene_percentiles=pct,
        print_density=print_d,
        print_reflectance=print_r,
        stats=stats,
    )


# ——— Rendering ———————————————————————————————————————————————

_BG = "#1d1d21"
_PANEL = "#26262b"
_TEXT = "#eae6df"
_DIM = "#a8a49b"
_ACCENT = "#e0954f"
_ACCENT2 = "#6fd1c7"
_GRID = "#3a3a40"


def render_curve_plot(report: CurveReport, *, width: int = 760, height: int = 620) -> np.ndarray:
    """Dark-themed film + system curve plot as an RGB array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100.0
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    fig.patch.set_facecolor(_BG)

    for ax in axes:
        ax.set_facecolor(_PANEL)
        ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)
        ax.tick_params(colors=_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

    ax_film, ax_print = axes

    # Scene placement behind everything — where this frame actually sits.
    if report.scene_log_e.size:
        counts, edges = np.histogram(report.scene_log_e, bins=90)
        if counts.max() > 0:
            centres = 0.5 * (edges[:-1] + edges[1:])
            top = float(np.max(report.active_density)) * 0.95
            scaled = counts / counts.max() * top
            ax_film.fill_between(
                centres, 0, scaled, color=_ACCENT, alpha=0.13, linewidth=0, zorder=1
            )

    ax_film.plot(
        report.log_e,
        report.base_density,
        color=_DIM,
        linewidth=1.2,
        linestyle="--",
        label="datasheet",
        zorder=2,
    )
    ax_film.plot(
        report.log_e,
        report.active_density,
        color=_ACCENT,
        linewidth=2.0,
        label="in play",
        zorder=3,
    )

    pct = report.scene_percentiles
    for key, label, colour in (("p5", "shadows", _ACCENT2), ("p95", "highlights", _ACCENT2)):
        if key in pct:
            ax_film.axvline(pct[key], color=colour, linewidth=1.0, alpha=0.75, zorder=4)
            ax_film.annotate(
                label,
                xy=(pct[key], ax_film.get_ylim()[1]),
                xytext=(2, -10),
                textcoords="offset points",
                color=colour,
                fontsize=7,
                rotation=90,
                va="top",
            )

    ax_film.set_ylabel("negative density", color=_TEXT, fontsize=9)
    ax_film.set_title(
        f"{report.stats.get('film', 'film')} · CI {report.stats.get('contrast_index', 0):.2f}"
        f" · {report.stats.get('curve_source', '')}",
        color=_TEXT,
        fontsize=10,
        pad=8,
    )
    leg = ax_film.legend(loc="upper left", fontsize=8, framealpha=0.0)
    for txt in leg.get_texts():
        txt.set_color(_DIM)

    # System curve: scene log-E straight through to what the paper gives back.
    # Zones are stops, so the reflectance axis is logarithmic — that puts the
    # zone bands at even spacing instead of piling them up against paper white.
    if report.print_reflectance is not None:
        refl = np.maximum(report.print_reflectance, 1e-4)
        ax_print.plot(report.log_e, refl, color=_ACCENT, linewidth=2.0, zorder=3)
        ax_print.set_yscale("log")

        floor = max(float(np.min(refl)) * 0.8, 2e-3)
        ceiling = min(float(np.max(refl)) * 1.25, 1.0)
        for zone in range(ZONE_COUNT):
            r = zone_reflectance(zone)
            if r < floor or r > ceiling:
                continue
            ax_print.axhline(r, color=_GRID, linewidth=0.5, alpha=0.8, zorder=1)
        ticks = [
            zone_reflectance(z)
            for z in range(ZONE_COUNT)
            if floor <= zone_reflectance(z) <= ceiling
        ]
        labels = [
            _roman(z)
            for z in range(ZONE_COUNT)
            if floor <= zone_reflectance(z) <= ceiling
        ]
        ax_print.set_yticks(ticks)
        ax_print.set_yticklabels(labels)
        ax_print.minorticks_off()

        for key in ("p5", "p95"):
            if key in pct:
                ax_print.axvline(pct[key], color=_ACCENT2, linewidth=1.0, alpha=0.75, zorder=4)
        ax_print.set_ylim(floor, ceiling)
        ax_print.set_ylabel("print zone", color=_TEXT, fontsize=9)
        ax_print.set_title(
            f"{report.stats.get('paper', 'paper')} · grade {report.stats.get('grade', 0):.1f}",
            color=_TEXT,
            fontsize=9,
            pad=6,
        )
    else:
        ax_print.text(
            0.5,
            0.5,
            "commit develop to see the print curve",
            transform=ax_print.transAxes,
            ha="center",
            va="center",
            color=_DIM,
            fontsize=9,
        )
        ax_print.set_yticks([])

    ax_print.set_xlabel("relative log exposure", color=_TEXT, fontsize=9)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.93, bottom=0.09, hspace=0.22)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _roman(zone: int) -> str:
    return ["0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][
        int(np.clip(zone, 0, 10))
    ]


def curve_summary_markdown(report: CurveReport) -> str:
    """The same findings as text — readable without squinting at the plot."""
    s = report.stats
    lines: list[str] = []

    if s.get("curve_source") == "family":
        src = "digitized curve family"
    elif s.get("curve_source") == "instant_layers":
        src = "integral pod layers"
    else:
        src = "morphed base curve"
    lines.append(f"**{s.get('film', 'Film')}** — {src}")
    lines.append(
        f"CI `{s.get('contrast_index', 0):.2f}` · fog `{s.get('base_plus_fog', 0):.2f}`"
        f" · Dmax `{s.get('d_max', 0):.2f}`"
    )

    if "scene_stops" in s:
        lines.append("")
        lines.append(f"**Scene** — `{s['scene_stops']:.1f}` stops")
        lines.append(
            f"shadows land at D `{s.get('shadow_density', 0):.2f}`,"
            f" highlights D `{s.get('highlight_density', 0):.2f}`"
        )

    if "shadow_zone" in s:
        lines.append("")
        if s.get("print_process") == "ra4":
            lines.append(
                f"**On {s.get('paper', 'RA-4')} · "
                f"{float(s.get('base_exposure_seconds', 8.0)):g}s** —"
                f" shadows print Zone `{_roman(round(s['shadow_zone']))}`,"
                f" highlights Zone `{_roman(round(s['highlight_zone']))}`"
            )
        else:
            lines.append(
                f"**On {s.get('paper', 'paper')} g{s.get('grade', 0):.1f}** —"
                f" shadows print Zone `{_roman(round(s['shadow_zone']))}`,"
                f" highlights Zone `{_roman(round(s['highlight_zone']))}`"
            )
    elif s.get("chemistry_mode") == "instant":
        lines.append("")
        lines.append("_Integral card — no enlarger paper curve._")

    return "  \n".join(lines)



def _scalar_at(plane: np.ndarray, y: int, x: int) -> float:
    """Sample a mono or RGB plane as a single reflectance/density value."""
    pix = plane[y, x]
    arr = np.asarray(pix, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    # Color prints: use mean channel reflectance as Zone meter luminance.
    return float(arr.reshape(-1).mean())


def spot_at(
    reflectance: np.ndarray | None,
    density: np.ndarray | None,
    nx: float,
    ny: float,
) -> dict[str, float | str]:
    """Sample Zone / density under a normalised print pointer.

    Accepts mono (H×W) or color (H×W×3) reflectance / density maps.
    """
    if reflectance is None or np.asarray(reflectance).size == 0:
        return {"ok": 0}
    r = np.asarray(reflectance, dtype=np.float64)
    h, w = r.shape[:2]
    x = int(np.clip(round(float(nx) * (w - 1)), 0, w - 1))
    y = int(np.clip(round(float(ny) * (h - 1)), 0, h - 1))
    refl = _scalar_at(r, y, x)
    dens_arr = None if density is None else np.asarray(density, dtype=np.float64)
    if dens_arr is not None and dens_arr.shape[:2] == (h, w):
        dens = _scalar_at(dens_arr, y, x)
    else:
        dens = float(-np.log10(max(refl, 1e-6)))
    zone = float(reflectance_to_zone(refl))
    return {
        "ok": 1,
        "x": x,
        "y": y,
        "reflectance": refl,
        "density": dens,
        "zone": zone,
        "zone_label": _roman(int(round(zone))),
    }


def spot_markdown(sample: dict[str, float | str]) -> str:
    if not sample.get("ok"):
        return "_Hover the print for Zone / density._"
    # Avoid markdown `code` chips — Gradio's light code fill washes out on the
    # dark spot float (white-on-white Zone / D / R). Bold keeps values readable.
    return (
        f"**Spot** Zone **{sample['zone_label']}**"
        f" ({float(sample['zone']):.1f}) · D **{float(sample['density']):.2f}**"
        f" · R **{float(sample['reflectance']):.3f}**"
    )


def render_print_histogram(
    reflectance: np.ndarray | None,
    *,
    width: int = 520,
    height: int = 160,
) -> np.ndarray | None:
    """Compact reflectance histogram with Zone tick marks."""
    if reflectance is None or np.asarray(reflectance).size == 0:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = _as_mono_reflectance(reflectance).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    dpi = 100.0
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    log_r = np.log10(np.maximum(r, 1e-4))
    ax.hist(log_r, bins=64, color=_ACCENT, alpha=0.85, range=(-3.0, 0.0))
    for zone in range(ZONE_COUNT):
        zr = zone_reflectance(zone)
        ax.axvline(np.log10(zr), color=_GRID, linewidth=0.7, alpha=0.9)
        if zone % 2 == 0:
            ax.text(
                np.log10(zr),
                1.0,
                _roman(zone),
                color=_DIM,
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlim(-3.0, 0.0)
    ax.set_xticks([np.log10(zone_reflectance(z)) for z in range(0, 11, 2)])
    ax.set_xticklabels([_roman(z) for z in range(0, 11, 2)], color=_DIM, fontsize=8)
    ax.set_yticks([])
    ax.tick_params(colors=_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.set_title("print reflectance · zones", color=_TEXT, fontsize=9, pad=4)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.22)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def suggest_tone_fit(
    reflectance: np.ndarray | None,
    *,
    base_seconds: float,
    grade: float,
    print_contrast: float = 0.0,
    chemistry_mode: str = "bw",
    hi_zone: float = 7.2,
    lo_zone: float = 1.2,
    target_lo: float = 2.0,
    target_mid: float = 5.0,
    target_hi: float = 7.0,
    min_seconds: float = 2.0,
    max_seconds: float = 64.0,
) -> dict[str, Any]:
    """Suggest exposure / filtration so the print sits on paper instead of clipping.

    First-order darkroom intuition: +1 enlarger stop ≈ 1 Zone darker. When the
    scene is too contrasty, soften filtration a little (MG grade for B&W, RA-4
    paper contrast for Color), then park the midtone near Zone V.

    Instant has no enlarger — returns ``ok=0``.
    """
    mode = str(chemistry_mode or "bw").lower()
    if mode == "instant":
        return {
            "ok": 0,
            "base_seconds": float(base_seconds),
            "grade": float(grade),
            "print_contrast": float(print_contrast),
            "message": (
                "Instant is a finished card — no enlarger paper to fit. "
                "Adjust process time / N± / temperature instead."
            ),
        }

    if reflectance is None or np.asarray(reflectance).size == 0:
        return {
            "ok": 0,
            "base_seconds": float(base_seconds),
            "grade": float(grade),
            "print_contrast": float(print_contrast),
            "message": "No print yet — Commit Develop first.",
        }

    mono = _as_mono_reflectance(reflectance)
    zones = np.asarray(reflectance_to_zone(mono), dtype=np.float64).reshape(-1)
    zones = zones[np.isfinite(zones)]
    if zones.size < 16:
        return {
            "ok": 0,
            "base_seconds": float(base_seconds),
            "grade": float(grade),
            "print_contrast": float(print_contrast),
            "message": "Not enough print samples to fit.",
        }

    p5, p50, p95 = (float(x) for x in np.percentile(zones, (5, 50, 95)))
    blown = p95 >= hi_zone
    crushed = p5 <= lo_zone
    span = p95 - p5

    new_grade = float(np.clip(grade, 0.0, 5.0))
    new_contrast = float(np.clip(print_contrast, -1.0, 1.0))
    filt_note = ""
    # Too much scene contrast for this filtration — open the scale a notch.
    if span > 6.2 and (blown or crushed):
        if mode == "color":
            # RA-4 contrast is −1…+1; soften toward negative.
            soften = float(np.clip(0.15 + 0.12 * (span - 6.2), 0.15, 0.55))
            softened = max(-1.0, new_contrast - soften)
            if softened < new_contrast - 0.02:
                filt_note = f" · contrast {new_contrast:+.2f}→{softened:+.2f}"
                new_contrast = float(round(softened * 20.0) / 20.0)
        else:
            soften = float(np.clip(0.5 + 0.35 * (span - 6.2), 0.5, 1.5))
            softened = max(0.0, new_grade - soften)
            if softened < new_grade - 0.05:
                filt_note = f" · grade {new_grade:.1f}→{softened:.1f}"
                new_grade = softened

    # d(zone)/d(exposure_stops) ≈ −1 on the print.
    if blown and not crushed:
        delta_stops = p95 - target_hi  # darken hot highlights
        intent = "protect highlights"
    elif crushed and not blown:
        delta_stops = p5 - target_lo  # lighten crushed shadows
        intent = "lift shadows"
    else:
        delta_stops = p50 - target_mid
        intent = "balance midtones" if not (blown or crushed) else "compromise both ends"

    # Tiny moves aren't worth a timer click.
    if abs(delta_stops) < 0.08 and not filt_note:
        return {
            "ok": 1,
            "base_seconds": float(base_seconds),
            "grade": new_grade,
            "print_contrast": new_contrast,
            "delta_stops": 0.0,
            "blown": blown,
            "crushed": crushed,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "changed": False,
            "message": (
                f"Already on paper — shadows Z{_roman(round(p5))}, "
                f"mid Z{_roman(round(p50))}, highlights Z{_roman(round(p95))}."
            ),
        }

    new_seconds = float(base_seconds) * (2.0 ** float(delta_stops))
    new_seconds = float(np.clip(new_seconds, min_seconds, max_seconds))
    # Snap to the UI slider step (0.5s).
    new_seconds = float(round(new_seconds * 2.0) / 2.0)
    changed = (
        abs(new_seconds - float(base_seconds)) >= 0.25
        or bool(filt_note)
    )

    direction = "longer" if new_seconds > float(base_seconds) + 0.05 else (
        "shorter" if new_seconds < float(base_seconds) - 0.05 else "same"
    )
    message = (
        f"**Fit to paper** — {intent}: timer {float(base_seconds):g}s→{new_seconds:g}s"
        f" ({direction}){filt_note}.  \n"
        f"Was Z{_roman(round(p5))}–{_roman(round(p95))} "
        f"(mid {_roman(round(p50))}). Overlay stays on so you can judge."
    )
    return {
        "ok": 1,
        "base_seconds": new_seconds,
        "grade": new_grade,
        "print_contrast": new_contrast,
        "delta_stops": float(delta_stops),
        "blown": blown,
        "crushed": crushed,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "changed": changed,
        "message": message,
    }


def apply_clipping_overlay(
    preview_rgb: np.ndarray,
    reflectance: np.ndarray | None,
    *,
    show_highlights: bool = True,
    show_shadows: bool = True,
    hi_zone: float = 7.2,
    lo_zone: float = 1.2,
) -> np.ndarray:
    """Tint near-paper-white / near-Dmax regions on a preview RGB.

    Zone IX–X sit above what fibre paper can return, so the highlight warning
    arms around Zone VII (paper white) rather than a theoretical Zone X.
    """
    rgb = np.asarray(preview_rgb)
    if reflectance is None or rgb.size == 0:
        return rgb
    r = _as_mono_reflectance(reflectance).astype(np.float32)
    if r.shape[:2] != rgb.shape[:2]:
        try:
            from PIL import Image

            im = Image.fromarray(r, mode="F")
            im = im.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.BILINEAR)
            r = np.asarray(im, dtype=np.float32)
        except Exception:
            return rgb
    out = rgb.astype(np.float32).copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    zone = reflectance_to_zone(r)
    if show_highlights:
        hi = zone >= hi_zone
        out[hi, 0] = np.minimum(255.0, out[hi, 0] * 0.35 + 220)
        out[hi, 1] = out[hi, 1] * 0.25
        out[hi, 2] = out[hi, 2] * 0.25
    if show_shadows:
        lo = zone <= lo_zone
        out[lo, 0] = out[lo, 0] * 0.25
        out[lo, 1] = out[lo, 1] * 0.35
        out[lo, 2] = np.minimum(255.0, out[lo, 2] * 0.35 + 200)
    return np.clip(out, 0, 255).astype(np.uint8)
