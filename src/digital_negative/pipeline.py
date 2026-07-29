"""End-to-end spike pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .curves import load_film_profile
from .development import develop
from .digital_negative import DigitalNegative
from .display import save_before_after, save_gray_preview
from .ingest import ingest_path


@dataclass
class SpikeArtifacts:
    digital_negative: DigitalNegative
    density_preview: Path
    positive_preview: Path
    comparison: Path
    dn_tiff: Path
    dn_json: Path
    stats: dict[str, Any]


def default_profiles_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "profiles" / "films"


def run_spike_pipeline(
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path = "output",
    profile_path: str | Path | None = None,
    relative_time: float = 1.0,
    contrast_modifier: float = 0.0,
) -> SpikeArtifacts:
    """Open source → Digital Negative → HP5 curve → before/after previews."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if profile_path is None:
        profile_path = default_profiles_dir() / "hp5-plus-v1.json"
    profile = load_film_profile(profile_path)

    dn = ingest_path(input_path)
    result = develop(
        dn,
        profile,
        relative_time=relative_time,
        contrast_modifier=contrast_modifier,
    )

    stem = dn.uuid
    dn_tiff, dn_json = dn.save(output_dir / "negatives", stem=stem)

    luma = dn.to_luminance()
    density_norm = (result.density - result.density.min()) / max(
        float(result.density.max() - result.density.min()), 1e-6
    )
    density_path = save_gray_preview(output_dir / f"{stem}_density.png", density_norm)
    positive_path = save_gray_preview(output_dir / f"{stem}_positive.png", result.positive_preview)
    comparison_path = save_before_after(
        output_dir / f"{stem}_comparison.png",
        luma,
        result.positive_preview,
        title="Digital Negative → Virtual Darkroom Spike",
        subtitle=f"{profile.name} · relative_time={relative_time:.2f} · contrast={contrast_modifier:+.2f}",
    )

    stats = {
        "uuid": dn.uuid,
        "source": dn.metadata["source"]["original_filename"],
        "film": profile.name,
        "image_shape": list(dn.image.shape),
        "linear_min": float(luma.min()),
        "linear_max": float(luma.max()),
        "linear_median": float(__import__("numpy").median(luma)),
        "density_min": float(result.density.min()),
        "density_max": float(result.density.max()),
        "density_mean": float(result.density.mean()),
        "log_e_min": float(result.log_exposure.min()),
        "log_e_max": float(result.log_exposure.max()),
    }

    return SpikeArtifacts(
        digital_negative=dn,
        density_preview=density_path,
        positive_preview=positive_path,
        comparison=comparison_path,
        dn_tiff=dn_tiff,
        dn_json=dn_json,
        stats=stats,
    )
