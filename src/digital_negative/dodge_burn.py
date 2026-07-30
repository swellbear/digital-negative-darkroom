"""Dodge & burn: local enlarger exposure maps (darkroom-style).

Dodge holds back light; burn adds light. The base print timer is set in
*seconds* (like an enlarger clock). Local passes accumulate light-time
delta under a freeform card/wand, then convert to stops relative to that
base: local_stops = log2((base + delta) / base).
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


# Calibrated "normal" enlarger timer — 8s ≡ 0 stops overall exposure.
REFERENCE_BASE_SECONDS = 8.0
# Legacy alias: burning for one full base interval ≈ +1 stop.
REFERENCE_SECONDS_PER_STOP = REFERENCE_BASE_SECONDS
# Server samples the waved card this often while exposing.
TICK_SECONDS = 0.25


def base_seconds_to_stops(
    base_seconds: float,
    *,
    reference: float = REFERENCE_BASE_SECONDS,
) -> float:
    """Map base enlarger seconds onto overall exposure stops (8s → 0)."""
    return float(np.log2(max(float(base_seconds), 1e-6) / max(float(reference), 1e-6)))


def stops_to_base_seconds(
    stops: float,
    *,
    reference: float = REFERENCE_BASE_SECONDS,
) -> float:
    return float(max(float(reference), 1e-6) * (2.0 ** float(stops)))


def seconds_to_stops(seconds: float, base_seconds: float | None = None) -> float:
    """Map a burn-style pass duration onto relative stops vs the base timer."""
    base = REFERENCE_BASE_SECONDS if base_seconds is None else float(base_seconds)
    seconds = max(float(seconds), 0.0)
    return float(np.log2(1.0 + seconds / max(base, 1e-6)))


def relative_pass_stops(seconds: float, base_seconds: float, mode: str) -> float:
    """Stops for a full-coverage dodge/burn pass of ``seconds`` vs base timer."""
    t = max(float(seconds), 0.0)
    base = max(float(base_seconds), 1e-6)
    if str(mode).lower().startswith("dodge"):
        t = min(t, base * 0.95)
        return float(np.log2(max(base - t, base * 0.05) / base))
    return float(np.log2(1.0 + t / base))


def stops_per_second(total_seconds: float, base_seconds: float | None = None) -> float:
    total_seconds = max(float(total_seconds), 1e-6)
    return seconds_to_stops(total_seconds, base_seconds) / total_seconds


def stops_per_tick(
    total_seconds: float,
    tick_seconds: float = TICK_SECONDS,
    base_seconds: float | None = None,
) -> float:
    return stops_per_second(total_seconds, base_seconds) * float(tick_seconds)


def delta_seconds_to_local_stops(
    delta_seconds: np.ndarray,
    base_seconds: float,
) -> np.ndarray:
    """Convert a light-time delta map (burn +, dodge −) into local stops."""
    base = max(float(base_seconds), 1e-6)
    effective = np.clip(base + np.asarray(delta_seconds, dtype=np.float32), base * 0.05, base * 8.0)
    return np.log2(effective / base).astype(np.float32)


def _layer_coverage(arr: np.ndarray | None) -> np.ndarray | None:
    """Coverage 0–1 from a paint layer. RGBA uses alpha only (not RGB)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] >= 4:
        alpha = a[..., 3].astype(np.float32)
        if alpha.max() > 1.0 + 1e-3:
            alpha = alpha / 255.0
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)
    if a.ndim == 3:
        rgb = a[..., :3].astype(np.float32)
        if rgb.max() > 1.0 + 1e-3:
            rgb = rgb / 255.0
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
            np.float32
        )
    g = a.astype(np.float32)
    if g.max() > 1.0 + 1e-3:
        g = g / 255.0
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def _to_gray_float(arr: np.ndarray | None) -> np.ndarray | None:
    """Backward-compatible alias — coverage from layers / images."""
    return _layer_coverage(arr)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask.astype(np.float32)
    try:
        from PIL import Image

        im = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
        return (np.asarray(im).astype(np.float32) / 255.0).astype(np.float32)
    except Exception:
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
    """Build a 0–1 tool coverage mask from an ImageEditor value (alpha layers)."""
    blank = np.zeros((height, width), dtype=np.float32)
    if not isinstance(editor_value, dict):
        return blank

    layers = editor_value.get("layers") or []
    mask = None
    if layers:
        acc = None
        for layer in layers:
            g = _layer_coverage(layer)
            if g is None:
                continue
            g = _resize_mask(g, height, width)
            acc = g if acc is None else np.maximum(acc, g)
        mask = acc

    if mask is None:
        # Fallback: painted coverage vs transparent / empty background
        composite = editor_value.get("composite")
        if composite is not None and np.asarray(composite).ndim == 3 and np.asarray(composite).shape[2] >= 4:
            mask = _resize_mask(_layer_coverage(composite), height, width)
        else:
            composite_g = _layer_coverage(composite)
            background_g = _layer_coverage(editor_value.get("background"))
            if composite_g is not None and background_g is not None:
                composite_g = _resize_mask(composite_g, height, width)
                background_g = _resize_mask(background_g, height, width)
                mask = np.clip(np.abs(composite_g - background_g) * 2.0, 0.0, 1.0)
            elif composite_g is not None:
                mask = _resize_mask(composite_g, height, width)

    if mask is None:
        return blank

    mask = np.where(mask > 0.08, mask, 0.0).astype(np.float32)
    if feather_px and feather_px > 0.5 and float(mask.max()) > 0:
        mask = gaussian_filter(mask, sigma=float(feather_px) * 0.45)
        if mask.max() > 1e-6:
            mask = mask / float(mask.max())
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def extract_tool_stamp(
    editor_value: dict[str, Any] | None,
    *,
    feather_px: float = 4.0,
    pad: int = 4,
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray | None:
    """Crop the painted silhouette into a reusable card/wand stamp (0–1).

    If target_height/width are set, scale the stamp so its size relative to the
    workshop canvas matches the same fraction of the print.
    """
    if not isinstance(editor_value, dict):
        return None

    layers = editor_value.get("layers") or []
    cover = None
    for layer in layers:
        g = _layer_coverage(layer)
        if g is None:
            continue
        cover = g if cover is None else np.maximum(cover, g)

    if cover is None:
        composite = editor_value.get("composite")
        if composite is not None and np.asarray(composite).ndim == 3 and np.asarray(composite).shape[2] >= 4:
            cover = _layer_coverage(composite)

    if cover is None or float(cover.max()) < 0.12:
        return None

    cover = np.where(cover > 0.08, cover, 0.0).astype(np.float32)
    workshop_h, workshop_w = cover.shape[:2]

    # Optionally rescale the whole workshop coverage into print space first.
    if target_height and target_width and (workshop_h, workshop_w) != (target_height, target_width):
        cover = _resize_mask(cover, int(target_height), int(target_width))

    ys, xs = np.where(cover > 0.08)
    if ys.size == 0:
        return None

    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, cover.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, cover.shape[1])
    stamp = cover[y0:y1, x0:x1].astype(np.float32)

    if feather_px and feather_px > 0.5:
        stamp = gaussian_filter(stamp, sigma=float(feather_px) * 0.35)
        if stamp.max() > 1e-6:
            stamp = stamp / float(stamp.max())
    return np.clip(stamp, 0.0, 1.0).astype(np.float32)


def place_stamp(
    height: int,
    width: int,
    stamp: np.ndarray,
    nx: float,
    ny: float,
) -> np.ndarray:
    """Place stamp centered at normalized (nx, ny) in [0, 1] image coords."""
    out = np.zeros((height, width), dtype=np.float32)
    sh, sw = stamp.shape[:2]
    if sh < 1 or sw < 1:
        return out

    cx = int(round(float(nx) * (width - 1)))
    cy = int(round(float(ny) * (height - 1)))
    x0 = cx - sw // 2
    y0 = cy - sh // 2
    x1 = x0 + sw
    y1 = y0 + sh

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(width, x1)
    dst_y1 = min(height, y1)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return out

    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    patch = stamp[src_y0:src_y1, src_x0:src_x1]
    out[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(out[dst_y0:dst_y1, dst_x0:dst_x1], patch)
    return out


def stamp_to_png_data_url(stamp: np.ndarray, *, tint: tuple[int, int, int] = (255, 200, 90)) -> str:
    """Encode stamp as a translucent PNG data-URL for the wave cursor overlay."""
    from PIL import Image

    s = np.clip(stamp, 0.0, 1.0)
    h, w = s.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = tint[0]
    rgba[..., 1] = tint[1]
    rgba[..., 2] = tint[2]
    rgba[..., 3] = (s * 200.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_pointer(pos: str | None) -> tuple[float, float] | None:
    """Parse 'nx,ny' normalized pointer from the UI. Returns None if missing/invalid."""
    if not pos or not isinstance(pos, str):
        return None
    text = pos.strip()
    if not text or text.lower() in {"", "none", "null"}:
        return None
    try:
        parts = text.replace(" ", "").split(",")
        if len(parts) < 2:
            return None
        nx = float(parts[0])
        ny = float(parts[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(nx) or not np.isfinite(ny):
        return None
    return (float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0)))


def _as_float_mask(value: Any) -> np.ndarray | None:
    """Coerce Gradio-roundtripped list/ndarray masks back to float32 arrays."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.astype(np.float32, copy=False)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim < 2 or arr.size == 0:
            return None
        return arr
    return None


def ensure_accum(state: dict[str, Any], height: int, width: int) -> np.ndarray:
    accum = _as_float_mask(state.get("db_accum"))
    if accum is None or accum.shape != (height, width):
        accum = np.zeros((height, width), dtype=np.float32)
    else:
        accum = accum.astype(np.float32, copy=False)
    state["db_accum"] = accum
    return accum


def apply_exposure_tick(
    state: dict[str, Any],
    editor_value: dict[str, Any] | None = None,
    *,
    height: int,
    width: int,
    position: tuple[float, float] | None = None,
) -> tuple[np.ndarray, bool]:
    """Accumulate one timer tick of dodge/burn. Returns (accum, still_exposing).

    Prefer a stored card stamp waved to ``position`` (normalized). Fall back to
    sampling the full editor mask when no stamp is set (tests / legacy).
    """
    if not state.get("db_exposing"):
        accum = ensure_accum(state, height, width)
        return accum, False

    left = float(state.get("db_seconds_left", 0.0))
    if left <= 1e-6:
        state["db_exposing"] = False
        state["db_seconds_left"] = 0.0
        return ensure_accum(state, height, width), False

    mode = str(state.get("db_mode", "burn"))
    tick_s = float(state.get("db_tick_seconds", TICK_SECONDS))
    sign = -1.0 if mode == "dodge" else 1.0

    # Prefer explicit pointer; otherwise keep the card where it last was.
    if position is None:
        last = state.get("db_last_pos")
        if isinstance(last, (list, tuple)) and len(last) == 2:
            try:
                position = (float(last[0]), float(last[1]))
            except (TypeError, ValueError):
                position = None
    if position is not None:
        state["db_last_pos"] = [float(position[0]), float(position[1])]

    stamp = _as_float_mask(state.get("db_stamp"))
    if stamp is not None:
        state["db_stamp"] = stamp  # keep coerced copy on state
        if position is None:
            mask = np.zeros((height, width), dtype=np.float32)
        else:
            mask = place_stamp(height, width, stamp, position[0], position[1])
    else:
        mask = mask_from_editor(
            editor_value,
            height=height,
            width=width,
            feather_px=float(state.get("db_feather_px", 6.0)),
        )

    accum = ensure_accum(state, height, width)
    # Accumulate light-time under the card (seconds), not prebaked stops.
    # Positive = extra burn light; negative = dodged (held back) light.
    applied = float(mask.max()) > 1e-4
    if applied:
        state["db_applied_ticks"] = int(state.get("db_applied_ticks", 0)) + 1
    accum = accum + (sign * tick_s) * mask
    state["db_accum"] = accum.astype(np.float32)

    left = max(0.0, left - tick_s)
    state["db_seconds_left"] = left
    if left <= 1e-6:
        state["db_exposing"] = False
        state["db_seconds_left"] = 0.0
        base = float(state.get("db_base_seconds", REFERENCE_BASE_SECONDS))
        strokes = state.setdefault("db_strokes", [])
        strokes.append(
            {
                "mode": mode,
                "seconds": int(state.get("db_total_seconds", 0)),
                "base_seconds": round(base, 3),
                "stops": round(
                    relative_pass_stops(
                        float(state.get("db_total_seconds", 0)),
                        base,
                        mode,
                    ),
                    3,
                ),
                "applied_ticks": int(state.get("db_applied_ticks", 0)),
            }
        )
    return state["db_accum"], bool(state.get("db_exposing"))


def reset_local_work(state: dict[str, Any]) -> dict[str, Any]:
    state["db_accum"] = None
    state["db_exposing"] = False
    state["db_seconds_left"] = 0.0
    state["db_strokes"] = []
    state["db_stamp"] = None
    state["db_stamp_url"] = None
    state["db_stamp_frac"] = None
    state["db_last_pos"] = None
    state["db_applied_ticks"] = 0
    return state


def local_stops_from_state(state: dict[str, Any] | None) -> np.ndarray | None:
    """Local stop map from accumulated light-time, relative to the base timer."""
    if not state:
        return None
    accum = _as_float_mask(state.get("db_accum"))
    if accum is None or not accum.size:
        return None
    if float(np.max(np.abs(accum))) <= 1e-6:
        return None
    state["db_accum"] = accum
    base = float(state.get("print_base_seconds") or state.get("db_base_seconds") or REFERENCE_BASE_SECONDS)
    return delta_seconds_to_local_stops(accum, base)


def tool_workshop_canvas(height: int = 480, width: int = 480) -> dict[str, Any]:
    """Dark blank canvas for cutting a card/wand — not painted over the print."""
    h = max(64, int(height))
    w = max(64, int(width))
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (32, 32, 34)
    return {"background": bg, "layers": [], "composite": bg}
