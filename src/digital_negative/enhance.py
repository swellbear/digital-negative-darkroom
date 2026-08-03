"""Optional Real-ESRGAN-style AI enlarge for large-print *exports*.

Process integrity: Develop / Print / Instant card composition stay untouched.
Callers must pass an already-styled RGB and only use the result for download
packages — never write it back into ``dn.image``, transmittance, or live preview.

Requires the optional extra in ``requirements-enhance.txt`` (``onnxruntime``).
Weights are fetched once via ``huggingface_hub``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# Pinned Real-ESRGAN x4plus ONNX (BSD-3-Clause Real-ESRGAN lineage).
DEFAULT_HF_REPO = "anakhiu/realesrgan-onnx"
DEFAULT_HF_FILE = "realesrgan_x4plus.onnx"
# Optional dedicated x2 when scale=2 (falls back to x4 + Lanczos).
DEFAULT_HF_REPO_X2 = "SceneWorks/real-esrgan-onnx"
DEFAULT_HF_FILE_X2 = "real_esrgan_x2.onnx"

_ENV_MODEL = "DIGITAL_NEGATIVE_ESRGAN_ONNX"


def enhance_available() -> bool:
    """True when onnxruntime can be imported (model may still need download)."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def _providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    ordered = []
    for name in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if name in available:
            ordered.append(name)
    return ordered or ["CPUExecutionProvider"]


def resolve_realesrgan_model(*, scale: int = 4) -> Path:
    """Return a local path to a Real-ESRGAN ONNX file (download/cache if needed)."""
    override = os.environ.get(_ENV_MODEL, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{_ENV_MODEL}={override!r} is not a file — point it at an ONNX model."
            )
        return path

    scale = 2 if int(scale) <= 2 else 4
    if scale == 2:
        repo, filename = DEFAULT_HF_REPO_X2, DEFAULT_HF_FILE_X2
    else:
        repo, filename = DEFAULT_HF_REPO, DEFAULT_HF_FILE

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the Real-ESRGAN ONNX weights."
        ) from exc

    try:
        local = hf_hub_download(repo_id=repo, filename=filename)
    except Exception as exc:
        # x2 repo missing → caller can use x4 + downscale
        if scale == 2:
            local = hf_hub_download(repo_id=DEFAULT_HF_REPO, filename=DEFAULT_HF_FILE)
            return Path(local)
        raise RuntimeError(
            f"Could not download Real-ESRGAN ONNX from {repo}/{filename}. "
            f"Set {_ENV_MODEL} to a local .onnx path, or check network access."
        ) from exc
    return Path(local)


@lru_cache(maxsize=4)
def _session_for(model_path: str) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=_providers())


def _model_tile_hw(session: Any) -> int | None:
    """Fixed H/W if the ONNX input is static; None when dynamic."""
    inp = session.get_inputs()[0]
    shape = list(inp.shape)
    # Expect NCHW
    if len(shape) != 4:
        return None
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0 and h == w:
        return int(h)
    return None


def _to_nchw01(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _from_nchw01(batch: np.ndarray) -> np.ndarray:
    x = np.asarray(batch, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _pad_to_multiple(rgb: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    h, w = rgb.shape[:2]
    nh = int(np.ceil(h / multiple) * multiple)
    nw = int(np.ceil(w / multiple) * multiple)
    if nh == h and nw == w:
        return rgb, h, w
    out = np.zeros((nh, nw, rgb.shape[2]), dtype=rgb.dtype)
    out[:h, :w] = rgb
    # Reflect-ish edge fill via replicate last pixel row/col
    if nh > h:
        out[h:, :w] = rgb[-1:, :]
    if nw > w:
        out[:, w:] = out[:, w - 1 : w]
    return out, h, w


def _run_tiles(
    session: Any,
    rgb: np.ndarray,
    *,
    scale: int,
    tile: int,
    overlap: int,
) -> np.ndarray:
    """Tile an HxWx3 uint8 image through the ONNX session and stitch."""
    inp = session.get_inputs()[0]
    out_name = session.get_outputs()[0].name
    fixed = _model_tile_hw(session)
    tile = int(fixed or max(32, tile))
    overlap = int(np.clip(overlap, 0, tile // 4))
    step = max(1, tile - overlap)

    work, orig_h, orig_w = _pad_to_multiple(rgb, step if overlap else tile)
    h, w = work.shape[:2]
    out_h, out_w = h * scale, w * scale
    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    # Ensure coverage of the bottom/right edge.
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile, x0 + tile
            patch = work[y0:y1, x0:x1]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                # Pad short edge tiles to the model tile size.
                canvas = np.zeros((tile, tile, 3), dtype=patch.dtype)
                canvas[: patch.shape[0], : patch.shape[1]] = patch
                patch = canvas
            tensor = _to_nchw01(patch)
            feeds = {inp.name: tensor}
            pred = session.run([out_name], feeds)[0]
            up = _from_nchw01(pred).astype(np.float32)
            # Crop model output to the valid region for this tile.
            vh, vw = min(tile, h - y0) * scale, min(tile, w - x0) * scale
            up = up[:vh, :vw]
            oy0, ox0 = y0 * scale, x0 * scale
            acc[oy0 : oy0 + vh, ox0 : ox0 + vw] += up
            weight[oy0 : oy0 + vh, ox0 : ox0 + vw] += 1.0

    weight = np.maximum(weight, 1e-6)
    merged = np.clip(acc / weight, 0, 255).astype(np.uint8)
    return merged[: orig_h * scale, : orig_w * scale]


def ai_upscale_rgb(
    rgb_u8: np.ndarray,
    *,
    scale: int = 4,
    tile: int = 400,
    overlap: int = 10,
) -> np.ndarray:
    """Upscale an already-styled RGB uint8 image with Real-ESRGAN (tiled).

    ``scale`` 2 or 4. For 2×, prefers a dedicated x2 ONNX when available;
    otherwise runs x4 and Lanczos-downsamples to 2×.
    """
    if not enhance_available():
        raise RuntimeError(
            "AI enlarge needs onnxruntime. Install with: "
            "pip install -r requirements-enhance.txt"
        )
    rgb = np.asarray(rgb_u8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.size == 0:
        raise ValueError("Empty image — nothing to enlarge.")

    want = 2 if int(scale) <= 2 else 4
    use_native_x2 = False
    model_scale = 4
    if want == 2:
        try:
            path = resolve_realesrgan_model(scale=2)
            # SceneWorks x2 file vs fallback x4plus from anakhiu
            use_native_x2 = "x2" in path.name.lower()
            model_scale = 2 if use_native_x2 else 4
        except Exception:
            path = resolve_realesrgan_model(scale=4)
            model_scale = 4
            use_native_x2 = False
    else:
        path = resolve_realesrgan_model(scale=4)
        model_scale = 4

    session = _session_for(str(path.resolve()))
    fixed = _model_tile_hw(session)
    tile_use = int(fixed or tile)
    # Keep overlap modest vs tile
    overlap_use = min(overlap, max(0, tile_use // 8))

    out = _run_tiles(
        session,
        rgb,
        scale=model_scale,
        tile=tile_use,
        overlap=overlap_use,
    )
    if want == 2 and model_scale == 4:
        from PIL import Image

        h, w = rgb.shape[:2]
        target = (w * 2, h * 2)
        out = np.asarray(
            Image.fromarray(out).resize(target, resample=Image.Resampling.LANCZOS)
        )
    return np.ascontiguousarray(out)


def maybe_ai_upscale_rgb(
    rgb_u8: np.ndarray,
    *,
    enabled: bool,
    scale: int = 4,
) -> np.ndarray:
    """No-op when disabled; otherwise ``ai_upscale_rgb``."""
    if not enabled:
        return np.asarray(rgb_u8)
    return ai_upscale_rgb(rgb_u8, scale=scale)
