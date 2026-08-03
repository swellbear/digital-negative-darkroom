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


def _horizon_flatness_score(gray: np.ndarray) -> tuple[float, float]:
    """Classical horizon level: dominant tone-break stays on one row across x.

    Returns ``(score, confidence)``. Soft sky/ground splits and hard shelf
    lines both peak when that break is level; tilted horizons raise the
    weighted row variance and lower the score. Border bands are ignored so
    rotate-fill edges cannot impersonate a horizon.
    """
    g = _norm01(np.asarray(gray, dtype=np.float32))
    h, w = g.shape[:2]
    if h < 16 or w < 16:
        return 0.0, 0.0
    gy = np.abs(ndimage.sobel(g, axis=0))
    gy = ndimage.gaussian_filter(gy, sigma=(1.2, 0.6))
    n = int(np.clip(w // 16, 8, 24))
    rows: list[float] = []
    strengths: list[float] = []
    for i in range(n):
        x0 = int(i * w / n)
        x1 = max(x0 + 1, int((i + 1) * w / n))
        col = np.mean(gy[:, x0:x1], axis=1)
        yy = np.linspace(0.0, 1.0, col.size, dtype=np.float64)
        # Prefer mid-frame / lower-third horizons; still allow high skies.
        prior = 0.25 + 0.75 * np.exp(-0.5 * ((yy - 0.55) / 0.32) ** 2)
        score = col * prior
        # Never pick the rotate-fill lip at the top/bottom of the trial.
        margin = max(2, int(0.08 * col.size))
        score[:margin] = 0.0
        score[-margin:] = 0.0
        if float(np.max(score)) <= 1e-12:
            continue
        yi = int(np.argmax(score))
        rows.append(float(yi))
        strengths.append(float(score[yi] / (float(np.mean(col)) + 1e-8)))
    if len(rows) < 4:
        return 0.0, 0.0
    rows_a = np.asarray(rows, dtype=np.float64)
    sw = np.clip(np.asarray(strengths, dtype=np.float64), 0.0, None)
    if float(sw.sum()) < 1e-8:
        return 0.0, 0.0
    mean = float(np.sum(rows_a * sw) / sw.sum())
    var = float(np.sum(sw * (rows_a - mean) ** 2) / sw.sum())
    # Normalize variance in row-fraction units so tall Instant frames
    # are not punished more than wide landscapes.
    var_norm = var / max(h * h * 0.0025, 1.0)
    flat = 1.0 / (1.0 + var_norm)
    strength = float(np.mean(strengths))
    # Scale ~O(1..10) so horizon can compete with plumb mass.
    score = float(flat * np.clip(strength, 0.0, 12.0))
    conf = float(np.clip(strength * flat, 0.0, 10.0))
    return score, conf


def _vertical_plumb_score(gray: np.ndarray) -> tuple[float, float]:
    """Classical plumb: strong edges stand near vertical (buildings, trees)."""
    g = _norm01(np.asarray(gray, dtype=np.float32))
    g = ndimage.gaussian_filter(g, sigma=1.0)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, 80))
    mask = mag >= max(thr, 1e-6)
    if int(np.count_nonzero(mask)) < 64:
        return 0.0, 0.0
    line_deg = np.degrees(np.arctan2(gx[mask], -gy[mask]))
    line_deg = np.asarray(line_deg, dtype=np.float64)
    # Wrap to [-90, 90).
    line_deg = ((line_deg + 90.0) % 180.0) - 90.0
    wts = mag[mask].astype(np.float64)
    to_v = 90.0 - np.abs(line_deg)
    to_h = np.abs(line_deg)
    # Tight 15° lobe — wide 45° lobes stayed high on tilted mullions and
    # flattened the search peak away from true plumb.
    lobe_v = np.clip(1.0 - to_v / 15.0, 0.0, 1.0) ** 2
    lobe_h = np.clip(1.0 - to_h / 15.0, 0.0, 1.0) ** 2
    v_mass = float(np.sum(wts * lobe_v))
    h_mass = float(np.sum(wts * lobe_h))
    total = float(np.sum(wts)) + 1e-8
    score = 10.0 * (v_mass / total)
    conf = float(
        np.clip((v_mass / (h_mass + v_mass + 1e-8)) * 10.0 * (v_mass / total), 0.0, 10.0)
    )
    return float(score), conf


def _axis_structure_score(gray: np.ndarray) -> float:
    """Hard stripe / mullion support — keeps architectural leveling sharp."""
    g = np.asarray(gray, dtype=np.float32)
    gy, gx = np.gradient(g)
    row_proj = np.mean(np.abs(gy), axis=1)
    col_proj = np.mean(np.abs(gx), axis=0)
    h_score = float(np.var(row_proj)) + 0.35 * float(np.var(np.diff(row_proj)))
    v_score = float(np.var(col_proj)) + 0.35 * float(np.var(np.diff(col_proj)))
    return h_score + v_score


def _scene_level_weights(gray: np.ndarray) -> tuple[float, float]:
    """Per-frame mix of horizon vs vertical emphasis (sums to 1).

    Sky/ground tone splits and a confident horizon band push weight to the
    horizon term; mullion-like vertical mass pushes to plumb. Local diagonal
    subject texture (dress stripes, leading lines) should not dominate.
    """
    g = _norm01(np.asarray(gray, dtype=np.float32))
    h = g.shape[0]
    top = float(np.mean(g[: max(1, h // 3)]))
    bot = float(np.mean(g[min(h - 1, (2 * h) // 3) :]))
    tone_split = abs(top - bot)
    _hf, h_conf = _horizon_flatness_score(g)
    _vf, v_conf = _vertical_plumb_score(g)
    w_h = 0.30 + 2.0 * tone_split + 0.10 * h_conf
    w_v = 0.30 + 0.16 * v_conf
    # Strong landscape split → trust the horizon even if noisy verticals exist.
    if tone_split >= 0.35:
        w_h += 0.55
    s = w_h + w_v
    return float(w_h / s), float(w_v / s)


def _composition_straighten_score(
    gray: np.ndarray,
    *,
    w_horizon: float,
    w_vertical: float,
) -> float:
    """Photographic leveling score for one trial orientation."""
    hf, _ = _horizon_flatness_score(gray)
    vf, _ = _vertical_plumb_score(gray)
    struct = _axis_structure_score(gray)
    # Structure term is light — sharpens mullion/stripe peaks without letting
    # busy subject texture outvote the scene geometry.
    return float(w_horizon * hf + w_vertical * vf + 0.04 * struct)


def _direct_structural_tilt_degrees(gray: np.ndarray) -> tuple[float, float]:
    """CW correction from near-axis structural edge orientations.

    Reads the median tilt of near-horizontal and near-vertical edges directly
    (classical "level the horizon / plumb the verticals") so a later search
    cannot jump to a stripe-harmonic angle far from the true composition.
    Returns ``(degrees_cw, confidence)``.
    """
    g = _norm01(np.asarray(gray, dtype=np.float32))
    g = ndimage.gaussian_filter(g, sigma=1.2)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, 78))
    mask = mag >= max(thr, 1e-6)
    if int(np.count_nonzero(mask)) < 80:
        return 0.0, 0.0
    line = np.degrees(np.arctan2(gx[mask], -gy[mask]))
    line = ((np.asarray(line, dtype=np.float64) + 90.0) % 180.0) - 90.0
    wts = mag[mask].astype(np.float64)

    def _weighted_median(vals: np.ndarray, weights: np.ndarray) -> float | None:
        if vals.size < 48:
            return None
        order = np.argsort(vals)
        v = vals[order]
        w = weights[order]
        cdf = np.cumsum(w)
        if float(cdf[-1]) <= 0:
            return None
        return float(v[min(int(np.searchsorted(cdf, 0.5 * cdf[-1])), len(v) - 1)])

    # Near-horizontal: line angle itself is the tilt from level.
    near_h = np.abs(line) <= 16.0
    # Near-vertical: map to signed deviation from ±90.
    near_v = np.abs(line) >= (90.0 - 16.0)
    candidates: list[tuple[float, float]] = []
    if np.count_nonzero(near_h) >= 48:
        # Sign: a horizon at +θ in this convention needs CW −θ to level — match
        # building/stripe fixtures via calibration against straighten_image.
        h_med = _weighted_median(line[near_h], wts[near_h])
        if h_med is not None:
            # Empirical CW correction for horizontal family.
            candidates.append((-float(h_med), float(np.sum(wts[near_h]))))
    if np.count_nonzero(near_v) >= 48:
        ld = line[near_v]
        tilt = np.where(ld >= 0.0, ld - 90.0, ld + 90.0)
        v_med = _weighted_median(tilt, wts[near_v])
        if v_med is not None:
            candidates.append((-float(v_med), float(np.sum(wts[near_v]))))
    if not candidates:
        return 0.0, 0.0
    # Prefer the family with more edge mass.
    candidates.sort(key=lambda t: t[1], reverse=True)
    ang, mass = candidates[0]
    conf = float(np.clip(mass / (float(np.sum(wts)) + 1e-8) * 12.0, 0.0, 10.0))
    return float(ang), conf


def estimate_straighten_degrees(
    image: np.ndarray,
    *,
    max_degrees: float = 12.0,
    max_side: int = 480,
    step: float = 0.25,
) -> float:
    """Estimate a fine straighten angle (degrees CW) for this composition.

    Classical darkroom leveling, not a generic "maximize edges" filter:

    1. Read the frame — sky/ground split vs vertical structure — and weight
       **horizon flatness** vs **plumb verticals** accordingly.
    2. Search small rotations for the orientation that best levels those
       cues (with a soft prior toward 0° so diagonal subjects do not invent
       large tilts).
    3. Accept a non-zero angle only when it clearly beats 0°.

    Uses the same CW convention as
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

    # Keep tone for soft horizons; a little high-pass helps mullions.
    blur = ndimage.gaussian_filter(gray, sigma=1.2)
    work = _norm01(0.70 * gray + 0.30 * np.abs(gray - blur))
    # Median fill avoids black wedges creating fake "horizons" during search.
    fill = float(np.median(work))

    w_h, w_v = _scene_level_weights(work)
    direct_deg, direct_conf = _direct_structural_tilt_degrees(work)
    limit = float(np.clip(max_degrees, 1.0, 20.0))
    step = float(max(step, 0.1))
    coarse = float(max(step, 0.5))
    direct_deg = float(np.clip(direct_deg, -limit, limit))

    def _score_at(deg: float) -> float:
        trial = straighten_image(work, float(deg), fill=fill)
        pad = int(round(0.12 * min(trial.shape[:2])))
        if pad > 0 and trial.shape[0] > 2 * pad + 8 and trial.shape[1] > 2 * pad + 8:
            core = trial[pad:-pad, pad:-pad]
        else:
            core = trial
        base = _composition_straighten_score(core, w_horizon=w_h, w_vertical=w_v)
        # Mild prior toward 0° — strong enough to block dress-stripe wild
        # angles, weak enough not to flatten a real 2–3° horizon peak.
        prior = float(np.exp(-0.5 * (float(deg) / 9.0) ** 2))
        return float(base * (0.78 + 0.22 * prior))

    scores: dict[float, float] = {}
    # When structural lines vote confidently, only search near that angle
    # (plus a discrete 0° alternative) — prevents clean stripe harmonics
    # (true 2.5° → false 6°) and opposite-sign plateaus around 0°.
    if direct_conf >= 1.8 and abs(direct_deg) >= 0.2:
        radius = 2.25
        lo = max(-limit, float(direct_deg) - radius)
        hi = min(limit, float(direct_deg) + radius)
        coarse_degs = list(np.arange(lo, hi + 0.5 * coarse, coarse, dtype=np.float64))
        coarse_degs.append(0.0)
        coarse_degs = sorted({float(np.round(d / step) * step) for d in coarse_degs})
    else:
        coarse_degs = list(
            np.arange(-limit, limit + 0.5 * coarse, coarse, dtype=np.float64)
        )

    for deg in coarse_degs:
        scores[float(deg)] = _score_at(float(deg))

    # Refine around the top coarse peaks so a flat horizon plateau cannot
    # trap the search at a nearby local bump (e.g. +0.5° vs true +2.25°).
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    seeds = [0.0, direct_deg] + [d for d, _ in ranked[:3]]
    best_deg = 0.0
    best_score = -1.0e9
    for seed in seeds:
        refine_lo = max(-limit, float(seed) - coarse)
        refine_hi = min(limit, float(seed) + coarse)
        for deg in np.arange(refine_lo, refine_hi + 0.5 * step, step, dtype=np.float64):
            key = float(deg)
            if key not in scores:
                scores[key] = _score_at(key)
            score = scores[key]
            if score > best_score:
                best_score = score
                best_deg = key

    score_at_zero = scores.get(0.0)
    if score_at_zero is None:
        score_at_zero = _score_at(0.0)
        scores[0.0] = score_at_zero

    if abs(best_deg) < 0.15:
        return 0.0
    if score_at_zero > 0:
        rel = best_score / max(score_at_zero, 1e-12)
        # Demand a real compositional lift over "already level."
        if rel < 1.008 and abs(best_deg) < 1.0:
            return 0.0
        if rel < 1.003:
            return 0.0
        # Large swings need stronger evidence (avoids dress-stripe traps).
        if abs(best_deg) >= 6.0 and rel < 1.03:
            return 0.0
        # If structural lines disagree with the search winner, trust lines.
        # Clean stripe frames often score a near-zero plateau slightly above the
        # true 2–3° peak; the orientation vote is the photographic ground truth.
        if direct_conf >= 2.0 and abs(direct_deg) >= 0.75:
            opposite = (best_deg * direct_deg) < 0 and abs(best_deg) > 0.35
            far_larger = abs(best_deg - direct_deg) >= 2.5 and abs(best_deg) > abs(direct_deg) + 1.5
            collapsed = abs(best_deg - direct_deg) >= 1.25 and abs(direct_deg) >= abs(best_deg) + 0.75
            if opposite or far_larger or collapsed:
                best_deg = float(direct_deg)
    return float(np.clip(round(best_deg / step) * step, -limit, limit))
