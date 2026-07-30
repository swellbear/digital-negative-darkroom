"""Dodge & burn: local enlarger exposure maps (darkroom-style).

Dodge holds back light (negative stops); burn adds light (positive stops).
Timer seconds map onto stop amounts; waving a freeform mask while the timer
runs accumulates exposure only where the tool covers each tick — like moving
a card under the enlarger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


# Reference: ~8 seconds of extra (or withheld) light ≈ 1 stop relative to base.
REFERENCE_SECONDS_PER_STOP = 8.0


def seconds_to_stops(seconds: float) -> float:
    """Map enlarger seconds onto relative exposure stops."""
    seconds = max(float(seconds), 0.0)
    return float(np.log2(1.0 + seconds / REFERENCE_SECONDS_PER_STOP))


def stops_per_second(total_seconds: float) -> float:
    total_seconds = max(float(total_seconds), 1e-6)
    return seconds_to_stops(total_seconds) / total_seconds


def _to_gray_float(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[2] >= 4:
            # Prefer alpha if present and used
            rgb = a[..., :3].astype(np.float32)
            alpha = a[..., 3].astype(np.float32)
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            # Brush strokes often white on transparent — use max(gray, alpha)
            return np.clip(np.maximum(gray, alpha), 0.0, 1.0).astype(np.float32)
        rgb = a[..., :3].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
            np.float32
        )
    g = a.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask.astype(np.float32)
    try:
        from PIL import Image

        im = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
        return (np.asarray(im).astype(np.float32) / 255.0).astype(np.float32)
    except Exception:
        # Nearest fallback without PIL
        ys = (np.linspace(0, mask.shape[0] - 1, height)).astype(int)
        xs = (np.linspace(0, mask.shape[1] - 1, width)).astype(int)
        return mask[ys][:, xs].astype(np.float32)


def mask_from_editor(
    editor_value: dict[str, Any] | None,
    *,
    height: int,
    width: int,
    feather_px: float = 6.0,
) -> np.ndarray:
    """Build a 0–1 tool mask from an ImageEditor value.

    Uses painted layers / composite strokes (not the background print).
    """
    blank = np.zeros((height, width), dtype=np.float32)
    if not isinstance(editor_value, dict):
        return blank

    layers = editor_value.get("layers") or []
    mask = None
    if layers:
        acc = None
        for layer in layers:
            g = _to_gray_float(layer)
            if g is None:
                continue
            g = _resize_mask(g, height, width)
            acc = g if acc is None else np.maximum(acc, g)
        mask = acc

    if mask is None:
        # Fall back: difference between composite and background
        composite = _to_gray_float(editor_value.get("composite"))
        background = _to_gray_float(editor_value.get("background"))
        if composite is not None and background is not None:
            composite = _resize_mask(composite, height, width)
            background = _resize_mask(background, height, width)
            mask = np.clip(np.abs(composite - background) * 2.0, 0.0, 1.0)
        elif composite is not None:
            mask = _resize_mask(composite, height, width)

    if mask is None:
        return blank

    # Soft threshold so faint brush haze doesn't count
    mask = np.where(mask > 0.08, mask, 0.0).astype(np.float32)
    if feather_px and feather_px > 0.5 and float(mask.max()) > 0:
        mask = gaussian_filter(mask, sigma=float(feather_px) * 0.45)
        if mask.max() > 1e-6:
            mask = mask / float(mask.max())
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def ensure_accum(state: dict[str, Any], height: int, width: int) -> np.ndarray:
    accum = state.get("db_accum")
    if (
        not isinstance(accum, np.ndarray)
        or accum.shape != (height, width)
        or accum.dtype != np.float32
    ):
        accum = np.zeros((height, width), dtype=np.float32)
        state["db_accum"] = accum
    return accum


def apply_exposure_tick(
    state: dict[str, Any],
    editor_value: dict[str, Any] | None,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, bool]:
    """Accumulate one timer second of dodge/burn. Returns (accum, still_exposing)."""
    if not state.get("db_exposing"):
        accum = ensure_accum(state, height, width)
        return accum, False

    left = int(state.get("db_seconds_left", 0))
    if left <= 0:
        state["db_exposing"] = False
        return ensure_accum(state, height, width), False

    mode = str(state.get("db_mode", "burn"))
    per_sec = float(state.get("db_stops_per_second", 0.0))
    sign = -1.0 if mode == "dodge" else 1.0
    mask = mask_from_editor(
        editor_value,
        height=height,
        width=width,
        feather_px=float(state.get("db_feather_px", 6.0)),
    )
    accum = ensure_accum(state, height, width)
    accum = accum + (sign * per_sec) * mask
    state["db_accum"] = accum.astype(np.float32)
    left -= 1
    state["db_seconds_left"] = left
    if left <= 0:
        state["db_exposing"] = False
        # Record stroke summary on the DN when a timed pass finishes
        strokes = state.setdefault("db_strokes", [])
        strokes.append(
            {
                "mode": mode,
                "seconds": int(state.get("db_total_seconds", 0)),
                "stops": round(seconds_to_stops(float(state.get("db_total_seconds", 0))), 3),
            }
        )
    return state["db_accum"], bool(state.get("db_exposing"))


def reset_local_work(state: dict[str, Any]) -> dict[str, Any]:
    state["db_accum"] = None
    state["db_exposing"] = False
    state["db_seconds_left"] = 0
    state["db_strokes"] = []
    return state


def local_stops_from_state(state: dict[str, Any] | None) -> np.ndarray | None:
    if not state:
        return None
    accum = state.get("db_accum")
    if isinstance(accum, np.ndarray) and accum.size and float(np.max(np.abs(accum))) > 1e-6:
        return accum.astype(np.float32)
    return None
