"""Ingest camera raws (or fallback images) into a Digital Negative."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .digital_negative import DigitalNegative, default_metadata

RAW_SUFFIXES = {
    ".arw",
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".orf",
    ".raf",
    ".rw2",
    ".dng",
    ".pef",
    ".srw",
}


def _file_hash(path: Path, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(nbytes)
        h.update(chunk)
    return h.hexdigest()[:16]


def _read_raw_linear(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
        )
        make = ""
        model = ""
        iso = 0
        try:
            make = (raw.color.camera_manufacturer or b"").decode("utf-8", errors="ignore")
            model = (raw.color.camera_model or b"").decode("utf-8", errors="ignore")
        except Exception:
            pass
        meta = {
            "original_filename": path.name,
            "camera_make": make,
            "camera_model": model,
            "iso": iso,
            "shutter_speed": "",
            "aperture": "",
            "datetime_original": "",
            "raw_hash": _file_hash(path),
        }
    linear = rgb.astype(np.float32) / 65535.0
    return linear, meta


def _read_image_linear(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0
    # Approximate inverse sRGB for a near-linear working space
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    meta = {
        "original_filename": path.name,
        "camera_make": "",
        "camera_model": "",
        "iso": 0,
        "shutter_speed": "",
        "aperture": "",
        "datetime_original": "",
        "raw_hash": _file_hash(path),
    }
    return linear.astype(np.float32), meta


def create_synthetic_scene(width: int = 960, height: int = 640) -> np.ndarray:
    """Generate a linear test scene with gradients, patches, and soft shapes."""
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    # Smooth horizontal exposure ramp (covers deep shadow → bright highlight)
    ramp = 0.002 + 1.8 * (x**1.6)

    # Vertical vignette-ish falloff
    vignette = 1.0 - 0.35 * ((y - 0.5) ** 2)

    # Soft circular "subject" highlight
    cy, cx = 0.42, 0.55
    rr = ((y - cy) ** 2) + ((x - cx) ** 2)
    subject = 0.55 * np.exp(-rr / 0.045)

    # Step wedges
    steps = np.floor(x * 11.0) / 10.0
    step_band = (y > 0.78) & (y < 0.92)
    base = ramp * vignette + subject
    base = np.where(step_band, 0.01 + steps * 1.2, base)

    # Mild color for RGB path (still used as luminance for B&W)
    r = base * 1.05
    g = base * 1.00
    b = base * 0.92
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, None).astype(np.float32)


def ingest_path(path: str | Path | None = None) -> DigitalNegative:
    """Ingest a raw/image file, or build a synthetic Digital Negative."""
    if path is None:
        image = create_synthetic_scene()
        source = {
            "original_filename": "synthetic_scene.tif",
            "camera_make": "Digital Negative",
            "camera_model": "Synthetic Spike Generator",
            "iso": 400,
            "shutter_speed": "1/125",
            "aperture": "f/8",
            "datetime_original": "",
            "raw_hash": "synthetic",
        }
        meta = default_metadata(source=source)
        meta["ingest"]["white_balance"]["method"] = "none"
        return DigitalNegative(image=image, metadata=meta)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in RAW_SUFFIXES:
        image, source = _read_raw_linear(path)
    else:
        image, source = _read_image_linear(path)

    meta = default_metadata(source=source)
    return DigitalNegative(image=image, metadata=meta)
