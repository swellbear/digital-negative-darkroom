"""Classical-composition auto-crop suggestions (numpy/scipy only)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

RULE_CHOICES = [
    ("Auto (best score)", "auto"),
    ("Rule of thirds", "rule_of_thirds"),
    ("Golden ratio", "golden_ratio"),
    ("Center subject", "center"),
    ("Horizon on thirds", "horizon_thirds"),
    ("Leading room", "leading_room"),
]

RULE_LABELS = {value: label for label, value in RULE_CHOICES}

# Candidate aspect ratios when the UI ratio is Free.
_FREE_ASPECTS = (1.0, 3 / 2, 2 / 3, 4 / 3, 3 / 4, 5 / 4, 4 / 5, 16 / 9, 9 / 16)


def parse_aspect_ratio(key: str | None, image_aspect: float) -> float | None:
    """Return width/height, or None when Free (engine may pick)."""
    k = str(key or "free").strip().lower()
    if not k or k == "free":
        return None
    if k == "original":
        return float(image_aspect) if image_aspect > 0 else None
    if ":" in k:
        a, b = k.split(":", 1)
        try:
            aw, ah = float(a), float(b)
            if aw > 0 and ah > 0:
                return aw / ah
        except ValueError:
            return None
    return None


def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        x = arr.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    g = arr.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    return g


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def interest_map(image: np.ndarray, *, max_side: int = 384) -> np.ndarray:
    """Edge + local-contrast interest map on a downscaled gray frame."""
    gray = _to_gray_f32(image)
    h, w = gray.shape[:2]
    m = max(h, w)
    if m > max_side:
        step = int(np.ceil(m / max_side))
        gray = np.ascontiguousarray(gray[::step, ::step])
    g = _norm01(gray)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    edge = _norm01(np.hypot(gx, gy))
    blur = ndimage.gaussian_filter(g, sigma=2.5)
    contrast = _norm01(np.abs(g - blur))
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    cy, cx = (g.shape[0] - 1) / 2.0, (g.shape[1] - 1) / 2.0
    # Mild center bias — classical subjects often live near mid-frame.
    dist = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    center = _norm01(1.0 - np.clip(dist, 0.0, 1.5) / 1.5)
    sal = 0.50 * edge + 0.35 * contrast + 0.15 * center
    sal = ndimage.gaussian_filter(sal, sigma=1.6)
    return _norm01(sal)


def subject_centroid(saliency: np.ndarray) -> tuple[float, float]:
    """Normalized (cx, cy) of weighted interest mass."""
    s = np.asarray(saliency, dtype=np.float32)
    w = np.maximum(s, 0.0) ** 2
    total = float(w.sum())
    h, ww = s.shape
    if total < 1e-8:
        return 0.5, 0.5
    yy, xx = np.mgrid[0:h, 0:ww]
    cx = float((xx * w).sum() / total) / max(ww - 1, 1)
    cy = float((yy * w).sum() / total) / max(h - 1, 1)
    return float(np.clip(cx, 0.02, 0.98)), float(np.clip(cy, 0.02, 0.98))


def _horizon_row(saliency: np.ndarray) -> float:
    """Normalized y of strongest horizontal structure (row energy)."""
    # Prefer horizontal edges: sobel already mixed; use row mean of saliency.
    row = saliency.mean(axis=1)
    # Ignore extreme top/bottom strips.
    n = row.shape[0]
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi <= lo + 2:
        return 0.5
    segment = row[lo:hi]
    idx = int(np.argmax(segment)) + lo
    return float(idx / max(n - 1, 1))


def _power_points(rule: str) -> list[tuple[float, float]]:
    if rule == "golden_ratio":
        g1, g2 = 0.382, 0.618
        return [(g1, g1), (g1, g2), (g2, g1), (g2, g2)]
    if rule == "center":
        return [(0.5, 0.5)]
    if rule == "horizon_thirds":
        # Subject/horizon sit on upper or lower third; x free at thirds + center.
        return [(x, y) for y in (1 / 3, 2 / 3) for x in (1 / 3, 0.5, 2 / 3)]
    if rule == "leading_room":
        # Subject off-center with room on one side.
        return [(1 / 3, 0.5), (2 / 3, 0.5), (1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    # rule_of_thirds default
    return [(1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3)]


def _max_box_for_aspect(aspect: float) -> tuple[float, float]:
    """Largest normalized (w, h) of the given aspect that fits in 1×1."""
    if aspect >= 1.0:
        return 1.0, min(1.0, 1.0 / aspect)
    return min(1.0, aspect), 1.0


def _place_box(
    sub_x: float,
    sub_y: float,
    target_x: float,
    target_y: float,
    box_w: float,
    box_h: float,
) -> tuple[float, float, float, float]:
    """Place crop so subject lands on (target_x, target_y) inside the box."""
    x = sub_x - target_x * box_w
    y = sub_y - target_y * box_h
    x = float(np.clip(x, 0.0, max(0.0, 1.0 - box_w)))
    y = float(np.clip(y, 0.0, max(0.0, 1.0 - box_h)))
    # If clamp moved the subject off-target, still keep a valid box.
    return x, y, box_w, box_h


def _score_box(
    saliency: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    target: tuple[float, float],
    sub: tuple[float, float],
    horizon_y: float | None = None,
    prefer_horizon: bool = False,
) -> float:
    x, y, w, h = box
    hs, ws = saliency.shape
    x0 = int(round(x * ws))
    y0 = int(round(y * hs))
    x1 = int(round((x + w) * ws))
    y1 = int(round((y + h) * hs))
    x0 = int(np.clip(x0, 0, max(ws - 2, 0)))
    y0 = int(np.clip(y0, 0, max(hs - 2, 0)))
    x1 = int(np.clip(x1, x0 + 2, ws))
    y1 = int(np.clip(y1, y0 + 2, hs))
    region = saliency[y0:y1, x0:x1]
    if region.size == 0:
        return -1e9
    # Mean interest inside + coverage of total interest.
    inside = float(region.mean())
    frac = float(region.sum()) / max(float(saliency.sum()), 1e-6)
    # Prefer crops that still leave a little breathing room (not full frame).
    tightness = 1.0 - min(w * h, 1.0)
    # Subject should land near the chosen power point after clamp.
    sx = (sub[0] - x) / max(w, 1e-6)
    sy = (sub[1] - y) / max(h, 1e-6)
    power = 1.0 - min(1.0, abs(sx - target[0]) + abs(sy - target[1]))
    # Soft edge penalty: high interest crushed against the crop border.
    border = 0.0
    bw = max(1, int(0.06 * region.shape[1]))
    bh = max(1, int(0.06 * region.shape[0]))
    border += float(region[:bh, :].mean()) + float(region[-bh:, :].mean())
    border += float(region[:, :bw].mean()) + float(region[:, -bw:].mean())
    border *= 0.15
    score = 0.45 * inside + 0.25 * frac + 0.20 * power + 0.12 * tightness - border
    if prefer_horizon and horizon_y is not None:
        hy = (horizon_y - y) / max(h, 1e-6)
        # Want horizon near 1/3 or 2/3 inside the crop.
        score += 0.18 * (1.0 - min(abs(hy - 1 / 3), abs(hy - 2 / 3)))
    return float(score)


def _candidate_aspects(requested: float | None, image_aspect: float) -> list[float]:
    if requested is not None and requested > 0:
        return [float(requested)]
    out = [float(image_aspect)]
    for a in _FREE_ASPECTS:
        if all(abs(a - b) > 0.03 for b in out):
            out.append(float(a))
    return out


def _scale_grid(max_w: float, max_h: float) -> list[tuple[float, float]]:
    """Try several crop sizes from tight to loose."""
    scales = (0.62, 0.72, 0.82, 0.90, 0.96)
    boxes = []
    for s in scales:
        w = max_w * s
        h = max_h * s
        if w >= 0.28 and h >= 0.28:
            boxes.append((w, h))
    return boxes or [(max_w, max_h)]


def suggest_crop_box(
    image: np.ndarray,
    *,
    rule: str = "auto",
    aspect_ratio: float | None = None,
) -> dict[str, Any]:
    """Suggest a normalized crop box ``{x,y,w,h}`` using classical composition rules.

    Parameters
    ----------
    image:
        RGB uint8 / float, or gray array used for interest analysis.
    rule:
        One of RULE_CHOICES values, or ``auto`` to score several.
    aspect_ratio:
        Desired width/height. ``None`` lets Free mode try common ratios.
    """
    sal = interest_map(image)
    hs, ws = sal.shape
    image_aspect = ws / max(hs, 1)
    sub = subject_centroid(sal)
    horizon = _horizon_row(sal)

    # Leading-room: bias target away from the denser side around the subject.
    left_mass = float(sal[:, : max(ws // 2, 1)].sum())
    right_mass = float(sal[:, max(ws // 2, 1) :].sum())

    rules = (
        ["rule_of_thirds", "golden_ratio", "center", "horizon_thirds", "leading_room"]
        if str(rule).lower() in {"auto", "", "best"}
        else [str(rule).lower()]
    )

    best: dict[str, Any] | None = None
    for rule_id in rules:
        targets = list(_power_points(rule_id))
        if rule_id == "leading_room":
            # Prefer putting mass toward the denser half → more open space opposite.
            if right_mass >= left_mass:
                targets = [(1 / 3, ty) for _, ty in targets] + targets
            else:
                targets = [(2 / 3, ty) for _, ty in targets] + targets
        if rule_id == "horizon_thirds":
            # Snap vertical target toward measured horizon band.
            prefer_low = horizon >= 0.5
            targets = [(tx, 2 / 3 if prefer_low else 1 / 3) for tx, _ in targets]

        for aspect in _candidate_aspects(aspect_ratio, image_aspect):
            max_w, max_h = _max_box_for_aspect(aspect)
            for box_w, box_h in _scale_grid(max_w, max_h):
                for tx, ty in targets:
                    box = _place_box(sub[0], sub[1], tx, ty, box_w, box_h)
                    score = _score_box(
                        sal,
                        box,
                        target=(tx, ty),
                        sub=sub,
                        horizon_y=horizon,
                        prefer_horizon=(rule_id == "horizon_thirds"),
                    )
                    cand = {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "w": float(box[2]),
                        "h": float(box[3]),
                        "score": score,
                        "rule": rule_id,
                        "aspect": float(aspect),
                        "subject": {"x": sub[0], "y": sub[1]},
                    }
                    if best is None or score > best["score"]:
                        best = cand

    if best is None:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.0, "rule": "center", "aspect": image_aspect}

    # Clamp final box.
    w = float(np.clip(best["w"], 0.2, 1.0))
    h = float(np.clip(best["h"], 0.2, 1.0))
    x = float(np.clip(best["x"], 0.0, 1.0 - w))
    y = float(np.clip(best["y"], 0.0, 1.0 - h))
    best.update({"x": x, "y": y, "w": w, "h": h})
    return best


def format_crop_rect(box: dict[str, Any]) -> str:
    return ",".join(f"{float(box[k]):.5f}" for k in ("x", "y", "w", "h"))


def _axis_alignment_score(gray: np.ndarray) -> float:
    """Higher when strong edges are axis-aligned (horizontals + verticals).

    Architectural scenes (window mullions, door frames) are dominated by
    verticals; horizons and shelves by horizontals. Scoring only row-projection
    variance made building tilts look "already level."
    """
    # Edge magnitude — prefer structure over smooth tone ramps.
    gy, gx = np.gradient(gray.astype(np.float32))
    # Horizontal-edge energy projected per row; vertical-edge energy per column.
    row_proj = np.mean(np.abs(gy), axis=1)
    col_proj = np.mean(np.abs(gx), axis=0)
    # Peakiness of those projections = how consistently edges share an axis.
    h_score = float(np.var(row_proj))
    v_score = float(np.var(col_proj))
    # Also reward discrete jumps in the 1-D profiles (lined-up edges).
    h_score += 0.35 * float(np.var(np.diff(row_proj)))
    v_score += 0.35 * float(np.var(np.diff(col_proj)))
    return h_score + v_score


def estimate_straighten_degrees(
    image: np.ndarray,
    *,
    max_degrees: float = 12.0,
    max_side: int = 480,
    step: float = 0.25,
) -> float:
    """Estimate a fine straighten angle (degrees CW) that levels the frame.

    Searches small rotations and picks the one that maximizes axis-aligned
    edge energy (horizontals *and* verticals). Uses the same CW convention as
    :func:`digital_negative.display.straighten_image`.
    """
    from .display import straighten_image

    gray = _to_gray_f32(image)
    h, w = gray.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        nh, nw = max(24, int(round(h * scale))), max(24, int(round(w * scale)))
        yy = np.linspace(0, h - 1, nh).astype(np.int32)
        xx = np.linspace(0, w - 1, nw).astype(np.int32)
        gray = np.ascontiguousarray(gray[yy][:, xx])

    # Mild high-pass so flat fields don't dominate the score.
    blur = ndimage.gaussian_filter(gray, sigma=1.2)
    work = _norm01(np.abs(gray - blur) + 0.25 * gray)

    limit = float(np.clip(max_degrees, 1.0, 20.0))
    step = float(max(step, 0.1))
    # Coarse then refine around the winner for better building/horizon hits.
    coarse = float(max(step, 0.5))
    candidates = list(np.arange(-limit, limit + 0.5 * coarse, coarse, dtype=np.float64))

    best_deg = 0.0
    best_score = -1.0
    score_at_zero = None
    for deg in candidates:
        trial = straighten_image(work, float(deg), fill=0.0)
        # Crop the filled corners so the black wedges don't fake a score.
        pad = int(round(0.08 * min(trial.shape[:2])))
        if pad > 0 and trial.shape[0] > 2 * pad + 8 and trial.shape[1] > 2 * pad + 8:
            core = trial[pad:-pad, pad:-pad]
        else:
            core = trial
        score = _axis_alignment_score(core)
        if abs(float(deg)) < 1e-9:
            score_at_zero = score
        if score > best_score:
            best_score = score
            best_deg = float(deg)

    # Local refine ± coarse step.
    refine_lo = max(-limit, best_deg - coarse)
    refine_hi = min(limit, best_deg + coarse)
    for deg in np.arange(refine_lo, refine_hi + 0.5 * step, step, dtype=np.float64):
        trial = straighten_image(work, float(deg), fill=0.0)
        pad = int(round(0.08 * min(trial.shape[:2])))
        if pad > 0 and trial.shape[0] > 2 * pad + 8 and trial.shape[1] > 2 * pad + 8:
            core = trial[pad:-pad, pad:-pad]
        else:
            core = trial
        score = _axis_alignment_score(core)
        if score > best_score:
            best_score = score
            best_deg = float(deg)

    # Snap near-zero noise to exactly 0 — but only when 0° is nearly as good.
    if abs(best_deg) < 0.15:
        return 0.0
    if score_at_zero is not None and best_score > 0:
        # Require a clear improvement over level before claiming a tilt.
        if best_score < score_at_zero * 1.02 and abs(best_deg) < 0.5:
            return 0.0
    return float(np.clip(round(best_deg / step) * step, -limit, limit))
