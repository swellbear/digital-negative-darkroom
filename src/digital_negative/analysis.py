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
from .print_engine import _filter_speed, paper_response

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
    mid_log_e: float = 2.2,
    samples: int = 320,
) -> CurveReport:
    """Sample the film curve in play, plus the film→paper system curve."""
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

    scene: np.ndarray = np.asarray([], dtype=np.float64)
    pct: dict[str, float] = {}
    if dn is not None:
        scene = _subsample(scene_log_exposure(dn, mid_log_e=mid_log_e)).astype(np.float64)
        if scene.size:
            for name, q in (("p1", 1), ("p5", 5), ("p50", 50), ("p95", 95), ("p99", 99)):
                pct[name] = float(np.percentile(scene, q))

    stats: dict[str, Any] = {
        "film": profile.name,
        "base_plus_fog": float(active.base_plus_fog),
        "d_min": float(np.min(active_d)),
        "d_max": float(np.max(active_d)),
        "contrast_index": _contrast_index(grid, active_d, float(active.base_plus_fog)),
        "curve_source": (active.raw.get("_last_curve_meta") or {}).get("curve_source", "morph"),
    }
    if pct:
        stats["scene_stops"] = (pct["p99"] - pct["p1"]) / 0.30103
        stats["shadow_density"] = float(
            active.density_from_log_exposure(np.asarray([pct["p5"]]))[0]
        )
        stats["highlight_density"] = float(
            active.density_from_log_exposure(np.asarray([pct["p95"]]))[0]
        )

    print_d = None
    print_r = None
    if paper is not None:
        eff_grade = float(paper.default_grade if grade is None else grade)
        stops = 0.0
        if base_exposure_seconds is not None:
            stops = float(np.log2(max(float(base_exposure_seconds), 1e-6) / 8.0))
        # Negative density -> transmittance -> enlarger light -> paper.
        transmittance = np.power(10.0, -active_d)
        light = transmittance * (2.0**stops) * _filter_speed(eff_grade)
        # Anchor the paper the same way print_negative does, off the scene's
        # own midtone when we have one.
        if scene.size:
            mid_d = active.density_from_log_exposure(np.asarray([pct["p50"]]))[0]
            centre = float(np.log10(max(10.0 ** (-float(mid_d)), 1e-6)))
            centre += float(stops * np.log10(2.0) + np.log10(_filter_speed(eff_grade)))
        else:
            centre = float(np.percentile(np.log10(np.maximum(light, 1e-6)), 48))
        print_d = paper_response(
            light, paper=paper, grade=eff_grade, log_center=centre
        ).astype(np.float64)
        print_r = np.power(10.0, -print_d)
        stats["paper"] = paper.name
        stats["grade"] = eff_grade
        if scene.size:
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

    src = "digitized curve family" if s.get("curve_source") == "family" else "morphed base curve"
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
        lines.append(
            f"**On {s.get('paper', 'paper')} g{s.get('grade', 0):.1f}** —"
            f" shadows print Zone `{_roman(round(s['shadow_zone']))}`,"
            f" highlights Zone `{_roman(round(s['highlight_zone']))}`"
        )

    return "  \n".join(lines)
