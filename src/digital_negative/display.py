"""Display helpers: linear → viewable, save comparison strips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, None)
    # Soft roll-off for values above 1 for preview only
    x = x / (1.0 + x)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_u8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_pil_gray(image: np.ndarray, *, assume_linear: bool = False) -> Image.Image:
    view = linear_to_srgb(image) if assume_linear else image
    return Image.fromarray(to_u8_gray(view), mode="L")


def negative_lightbox_preview(transmittance: np.ndarray) -> np.ndarray:
    """View a developed negative as if on a light table.

    Thin shadow areas transmit more light (bright); dense highlights hold
    light back (dark) — the classic inverted film-negative look.
    """
    t = np.clip(transmittance, 0.0, None)
    # Normalize to a readable lightbox range without crushing the toe
    lo = float(np.percentile(t, 1))
    hi = float(np.percentile(t, 99))
    span = max(hi - lo, 1e-6)
    view = np.clip((t - lo) / span, 0.0, 1.0)
    # Mild gamma so mid densities read clearly on screen
    return np.power(view, 0.85).astype(np.float32)


def original_photo_preview(path: str | Path | None = None, *, dn_image: np.ndarray | None = None) -> np.ndarray:
    """Display-referred RGB of the source for start→finish comparison.

    This is *not* the Digital Negative:
    - Camera raw → camera-WB sRGB demosaic with a normal display TRC
    - JPEG/TIFF/PNG → pixels as stored
    - Synthetic / no path → display-mapped scene from the DN image
    """
    if path:
        path = Path(path)
        suffix = path.suffix.lower()
        raw_suffixes = {
            ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
            ".dng", ".pef", ".srw",
        }
        if suffix in raw_suffixes:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(
                    output_bps=8,
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    highlight_mode=rawpy.HighlightMode.Clip,
                )
            return np.ascontiguousarray(rgb)
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    # Synthetic / fallback: display-map DN luminance (or RGB) for a readable source view
    if dn_image is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if dn_image.ndim == 3 and dn_image.shape[-1] >= 3:
        # Prefer Y if XYZ-like; otherwise Rec.709-ish luma from first three channels
        luma = dn_image[..., 1]
    else:
        luma = dn_image
    g = to_u8_gray(linear_to_srgb(luma))
    return np.stack([g, g, g], axis=-1)


def save_gray_preview(path: str | Path, image: np.ndarray, *, assume_linear: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = linear_to_srgb(image) if assume_linear else image
    Image.fromarray(to_u8_gray(view), mode="L").save(path)
    return path


def make_before_after(
    before_linear: np.ndarray,
    after_positive: np.ndarray,
    *,
    title: str = "Digital Negative Spike",
    subtitle: str = "",
    left_label: str = "Linear (Digital Negative)",
    right_label: str = "After film curve / print",
) -> Image.Image:
    left = to_u8_gray(linear_to_srgb(before_linear if before_linear.ndim == 2 else before_linear))
    right = to_u8_gray(after_positive)

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + 24
    header = 72
    canvas = Image.new("RGB", (w, h + header), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 18), title, fill=(235, 235, 235), font=font)
    if subtitle:
        draw.text((16, 40), subtitle, fill=(170, 170, 170), font=font)

    canvas.paste(Image.fromarray(left, mode="L"), (0, header))
    canvas.paste(Image.fromarray(right, mode="L"), (left.shape[1] + 24, header))
    draw.text((12, header + 8), left_label, fill=(220, 220, 220), font=font)
    draw.text((left.shape[1] + 36, header + 8), right_label, fill=(220, 220, 220), font=font)
    return canvas


def save_before_after(
    path: str | Path,
    before_linear: np.ndarray,
    after_positive: np.ndarray,
    *,
    title: str = "Digital Negative Spike",
    subtitle: str = "",
    left_label: str = "Linear (Digital Negative)",
    right_label: str = "After film curve / print",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = make_before_after(
        before_linear,
        after_positive,
        title=title,
        subtitle=subtitle,
        left_label=left_label,
        right_label=right_label,
    )
    img.save(path)
    return path
