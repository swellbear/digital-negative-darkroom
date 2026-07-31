"""End-to-end darkroom pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .curves import load_film_profile
from .development import DevelopmentResult, develop
from .digital_negative import DigitalNegative
from .display import save_before_after, save_gray_preview, to_pil_gray
from .ingest import ingest_path
from .papers import default_papers_dir, load_paper_profile
from .print_engine import PrintResult, print_negative


@dataclass
class DarkroomArtifacts:
    digital_negative: DigitalNegative
    development: DevelopmentResult
    print_result: PrintResult | None
    density_preview: Path
    developed_preview: Path
    print_preview: Path | None
    comparison: Path
    dn_tiff: Path
    dn_json: Path
    stats: dict[str, Any]


def default_profiles_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "profiles" / "films"


def list_film_profiles(*, chemistry_mode: str | None = None) -> list[Path]:
    """List film profiles, optionally filtered by ``bw`` / ``color`` chemistry mode."""
    paths = sorted(default_profiles_dir().glob("*-v1.json"))
    if chemistry_mode is None:
        return paths
    mode = str(chemistry_mode).lower()
    out: list[Path] = []
    for path in paths:
        try:
            data = __import__("json").loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ftype = str(data.get("type", "bw")).lower()
        is_color = ftype in {"color_negative", "color_slide", "color_ra4"}
        if mode == "color" and is_color:
            out.append(path)
        elif mode == "bw" and not is_color:
            out.append(path)
    return out


def list_paper_profiles(*, chemistry_mode: str | None = None) -> list[Path]:
    """List paper profiles, optionally filtered by ``bw`` / ``color`` chemistry mode."""
    paths = sorted(default_papers_dir().glob("*-v1.json"))
    if chemistry_mode is None:
        return paths
    mode = str(chemistry_mode).lower()
    out: list[Path] = []
    for path in paths:
        try:
            data = __import__("json").loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ptype = str(data.get("type", "bw_multigrade")).lower()
        is_color = ptype in {"color_ra4", "color"}
        if mode == "color" and is_color:
            out.append(path)
        elif mode == "bw" and not is_color:
            out.append(path)
    return out


def _resolve_profile(directory: Path, profile_id: str) -> Path:
    direct = directory / f"{profile_id}.json"
    if direct.exists():
        return direct
    matches = list(directory.glob(f"{profile_id}*.json"))
    if not matches:
        raise FileNotFoundError(f"No profile for id={profile_id} in {directory}")
    return matches[0]


def run_darkroom_pipeline(
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path = "output",
    film_id: str = "hp5-plus-v1",
    paper_id: str = "mg-standard",
    profile_path: str | Path | None = None,
    paper_path: str | Path | None = None,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
    grain_strength: float = 1.0,
    developer_id: str = "standard",
    print_exposure: float = 0.0,
    print_grade: float = 2.5,
    print_contrast: float = 0.0,
    do_print: bool = True,
) -> DarkroomArtifacts:
    """Ingest → develop → optional print → preview artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if profile_path is None:
        profile_path = _resolve_profile(default_profiles_dir(), film_id)
    profile = load_film_profile(profile_path)

    if paper_path is None:
        paper_path = _resolve_profile(default_papers_dir(), paper_id)
    paper = load_paper_profile(paper_path)

    dn = ingest_path(input_path)
    development = develop(
        dn,
        profile,
        relative_time=relative_time,
        contrast_modifier=contrast_modifier,
        grain_strength=grain_strength,
        developer_id=developer_id,
    )

    print_result = None
    if do_print:
        print_result = print_negative(
            development.transmittance,
            dn,
            paper,
            overall_exposure=print_exposure,
            grade=print_grade,
            contrast=print_contrast,
        )
        if "print" not in dn.metadata["ui_state"]["committed_stages"]:
            dn.metadata["ui_state"]["committed_stages"].append("print")

    stem = dn.uuid
    dn_tiff, dn_json = dn.save(output_dir / "negatives", stem=stem)

    luma = dn.to_luminance()
    density_norm = (development.density - development.density.min()) / max(
        float(development.density.max() - development.density.min()), 1e-6
    )
    density_path = save_gray_preview(output_dir / f"{stem}_density.png", density_norm)
    developed_path = save_gray_preview(
        output_dir / f"{stem}_developed.png", development.positive_preview
    )

    final_preview = (
        print_result.preview if print_result is not None else development.positive_preview
    )
    print_path = None
    if print_result is not None:
        print_path = save_gray_preview(output_dir / f"{stem}_print.png", print_result.preview)

    comparison_path = save_before_after(
        output_dir / f"{stem}_comparison.png",
        luma,
        final_preview,
        title="Digital Negative → Virtual Darkroom",
        subtitle=(
            f"{profile.name} · {developer_id} · rel={relative_time:.2f} · "
            f"grade={print_grade:.1f} · exp={print_exposure:+.2f} stops"
        ),
        right_label="Print" if print_result is not None else "Developed positive",
    )

    stats = {
        "uuid": dn.uuid,
        "source": dn.metadata["source"]["original_filename"],
        "film": profile.name,
        "developer": developer_id,
        "image_shape": list(dn.image.shape),
        "density_min": float(development.density.min()),
        "density_max": float(development.density.max()),
        "density_mean": float(development.density.mean()),
        "print_enabled": print_result is not None,
        "print_mean_reflectance": (
            float(np.mean(print_result.reflectance)) if print_result is not None else None
        ),
    }

    return DarkroomArtifacts(
        digital_negative=dn,
        development=development,
        print_result=print_result,
        density_preview=density_path,
        developed_preview=developed_path,
        print_preview=print_path,
        comparison=comparison_path,
        dn_tiff=dn_tiff,
        dn_json=dn_json,
        stats=stats,
    )


# Backward-compatible alias used by the original spike script/tests
def run_spike_pipeline(**kwargs: Any) -> DarkroomArtifacts:
    kwargs.setdefault("do_print", False)
    return run_darkroom_pipeline(**kwargs)


def process_preview_arrays(
    *,
    input_path: str | Path | None = None,
    film_id: str = "hp5-plus-v1",
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
    grain_strength: float = 1.0,
    developer_id: str = "standard",
    print_exposure: float = 0.0,
    print_grade: float = 2.5,
    print_contrast: float = 0.0,
):
    """Return PIL previews for the UI without requiring disk paths for display."""
    artifacts = run_darkroom_pipeline(
        input_path=input_path,
        film_id=film_id,
        relative_time=relative_time,
        contrast_modifier=contrast_modifier,
        grain_strength=grain_strength,
        developer_id=developer_id,
        print_exposure=print_exposure,
        print_grade=print_grade,
        print_contrast=print_contrast,
        do_print=True,
    )
    developed = to_pil_gray(artifacts.development.positive_preview)
    printed = to_pil_gray(artifacts.print_result.preview) if artifacts.print_result else developed
    comparison = artifacts.comparison
    return developed, printed, str(comparison), artifacts.stats
