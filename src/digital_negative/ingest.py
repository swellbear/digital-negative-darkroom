"""Ingest camera raws (or fallback images) into a Digital Negative.

Design goal (from product critique): the Digital Negative image payload should
be as close to linear scene-referred data as practical. Creative tone mapping
belongs in development / print — not at ingest.
"""

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

# Phone / consumer stills that Pillow can open (HEIF via pillow-heif).
IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".avif",
}

_HEIF_REGISTERED = False


def _ensure_heif_support() -> None:
    """Register HEIF/HEIC/AVIF openers once (no-op if pillow-heif missing)."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    _HEIF_REGISTERED = True


def _file_hash(path: Path, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(nbytes)
        h.update(chunk)
    return h.hexdigest()[:16]


def _read_raw_linear(path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Decode a camera raw to linear CIE XYZ (no display tone curve).

    Returns (xyz_image, source_meta, ingest_meta).
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        # gamma=(1,1) → linear encoding (no sRGB TRC).
        # ColorSpace.XYZ → scene-referred tristimulus after WB + camera matrix.
        # This avoids baking a display-referred sRGB look into the Digital Negative.
        xyz = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.XYZ,
            highlight_mode=rawpy.HighlightMode.Clip,
        )
        make = ""
        model = ""
        try:
            make = (raw.color.camera_manufacturer or b"").decode("utf-8", errors="ignore")
            model = (raw.color.camera_model or b"").decode("utf-8", errors="ignore")
        except Exception:
            pass
        # Camera WB multipliers when available (documented, not re-applied)
        wb = [1.0, 1.0, 1.0, 1.0]
        try:
            cam_wb = list(raw.camera_whitebalance)
            if cam_wb and len(cam_wb) >= 3:
                wb = [float(v) for v in cam_wb[:4]] if len(cam_wb) >= 4 else [
                    float(cam_wb[0]),
                    float(cam_wb[1]),
                    float(cam_wb[2]),
                    float(cam_wb[1]),
                ]
        except Exception:
            pass

        source = {
            "original_filename": path.name,
            "camera_make": make,
            "camera_model": model,
            "iso": 0,
            "shutter_speed": "",
            "aperture": "",
            "datetime_original": "",
            "raw_hash": _file_hash(path),
        }
        ingest = {
            "white_balance": {
                "method": "as_shot",
                "temperature": 0,
                "tint": 0,
                "camera_multipliers": wb,
            },
            "orientation": 1,
            "encoding": "linear",
            "working_space": "CIE_XYZ",
            "luminance_channel": "Y",
            "notes": (
                "Scene-referred linear XYZ from rawpy demosaic + camera WB. "
                "No display tone curve applied at ingest."
            ),
        }
    linear = xyz.astype(np.float32) / 65535.0
    return linear, source, ingest


def _read_image_linear(path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Decode a rendered image by undoing sRGB TRC (pragmatic approximation)."""
    from PIL import ImageOps

    _ensure_heif_support()
    suffix = path.suffix.lower()
    try:
        with Image.open(path) as im:
            # Honor embedded EXIF orientation so phone JPEGs/HEIFs aren't sideways
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001 — surface a clear ingest error
        if suffix in {".heic", ".heif", ".avif"}:
            raise RuntimeError(
                f"Could not decode {suffix} image. Install pillow-heif "
                f"(pip install pillow-heif) and retry. Underlying error: {exc}"
            ) from exc
        raise
    # Approximate inverse sRGB for a near-linear working space
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    kind = "HEIF/HEIC" if suffix in {".heic", ".heif"} else ("AVIF" if suffix == ".avif" else "Rendered")
    source = {
        "original_filename": path.name,
        "camera_make": "",
        "camera_model": "",
        "iso": 0,
        "shutter_speed": "",
        "aperture": "",
        "datetime_original": "",
        "raw_hash": _file_hash(path),
    }
    ingest = {
        "white_balance": {"method": "none", "temperature": 0, "tint": 0},
        "orientation": 1,
        "encoding": "linear_approx",
        "working_space": "linear_sRGB_primaries",
        "luminance_channel": "Rec709",
        "notes": (
            f"{kind} file: inverse sRGB TRC applied. This is an approximation — "
            "prefer camera raws for a true Digital Negative."
        ),
    }
    return linear.astype(np.float32), source, ingest


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

    # Store as faux XYZ with Y=luma and slight X/Z imbalance (still linear)
    y_ch = base
    x_ch = base * 1.02
    z_ch = base * 0.96
    xyz = np.stack([x_ch, y_ch, z_ch], axis=-1)
    return np.clip(xyz, 0.0, None).astype(np.float32)


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
        meta["ingest"] = {
            "white_balance": {"method": "none", "temperature": 0, "tint": 0},
            "orientation": 1,
            "encoding": "linear",
            "working_space": "CIE_XYZ",
            "luminance_channel": "Y",
            "notes": "Synthetic linear XYZ test scene.",
        }
        return DigitalNegative(image=image, metadata=meta)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in RAW_SUFFIXES:
        image, source, ingest = _read_raw_linear(path)
    else:
        image, source, ingest = _read_image_linear(path)

    meta = default_metadata(source=source)
    meta["ingest"] = ingest
    return DigitalNegative(image=image, metadata=meta)
