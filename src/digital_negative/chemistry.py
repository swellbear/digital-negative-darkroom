"""Film × chemistry development: named developers, minutes, CI-based relative time.

Public manufacturer datasheets publish normal times and (for some stocks) contrast-index
vs time. Each film profile may declare a ``chemistries`` list; the base characteristic
curve remains the digitized stock curve for ``is_base`` chemistry. Other developers
share that curve shape but apply chemistry character biases, and minutes are mapped
to the existing relative-development engine via CI ratio when ``ci_points`` exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

# Character families — stand-ins for solvent / balanced / vigorous / acutance chemistries.
# Legacy style ids (standard / high_definition / high_energy) alias into these.
CHARACTER_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "name": "Balanced",
        "contrast_bias": 0.0,
        "density_bias": 1.0,
        "grain_bias": 1.0,
        "fog_lift": 0.0,
        "toe_softness": 0.0,
        "shoulder_roll": 0.0,
    },
    "solvent": {
        "name": "Solvent / fine-grain",
        "contrast_bias": 0.06,
        "density_bias": 0.94,
        "grain_bias": 0.50,
        "fog_lift": -0.012,
        "toe_softness": 0.10,
        "shoulder_roll": 0.06,
    },
    "vigorous": {
        "name": "Vigorous / speed",
        "contrast_bias": 0.36,
        "density_bias": 1.14,
        "grain_bias": 1.38,
        "fog_lift": 0.022,
        "toe_softness": -0.05,
        "shoulder_roll": -0.035,
    },
    "acutance": {
        "name": "Acutance",
        "contrast_bias": 0.18,
        "density_bias": 1.05,
        "grain_bias": 1.45,
        "fog_lift": 0.01,
        "toe_softness": 0.04,
        "shoulder_roll": -0.02,
    },
}

_LEGACY_STYLE_ALIASES = {
    "standard": "balanced",
    "high_definition": "solvent",
    "high_energy": "vigorous",
}

# Keep old names importable for tests / CLI.
DEVELOPER_STYLES: dict[str, dict[str, Any]] = {
    "standard": {**CHARACTER_PRESETS["balanced"], "name": "Standard"},
    "high_definition": {**CHARACTER_PRESETS["solvent"], "name": "High Definition"},
    "high_energy": {**CHARACTER_PRESETS["vigorous"], "name": "High Energy"},
}


def character_style(character_id: str) -> dict[str, Any]:
    cid = _LEGACY_STYLE_ALIASES.get(character_id, character_id)
    if cid in CHARACTER_PRESETS:
        return dict(CHARACTER_PRESETS[cid])
    if character_id in DEVELOPER_STYLES:
        return dict(DEVELOPER_STYLES[character_id])
    return dict(CHARACTER_PRESETS["balanced"])


def _profile_raw(profile: Any) -> dict[str, Any]:
    if isinstance(profile, dict):
        return profile
    raw = getattr(profile, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _profile_defaults(profile: Any) -> dict[str, Any]:
    if isinstance(profile, dict):
        return dict(profile.get("defaults") or {})
    defaults = getattr(profile, "defaults", None)
    return dict(defaults) if isinstance(defaults, dict) else {}


def chemistries_map(profile: Any) -> dict[str, dict[str, Any]]:
    raw = _profile_raw(profile)
    items = raw.get("chemistries") or []
    if isinstance(items, dict):
        return {str(k): dict(v) for k, v in items.items()}
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        cid = str(item["id"])
        out[cid] = dict(item)
    return out


def default_chemistry_id(profile: Any) -> str:
    chems = chemistries_map(profile)
    if not chems:
        return str(_profile_defaults(profile).get("developer_id", "standard"))
    for cid, chem in chems.items():
        if chem.get("is_base"):
            return cid
    preferred = _profile_defaults(profile).get("developer_id")
    if preferred in chems:
        return str(preferred)
    return next(iter(chems))


def get_chemistry(profile: Any, developer_id: str) -> dict[str, Any] | None:
    chems = chemistries_map(profile)
    if developer_id in chems:
        return chems[developer_id]
    return None


def chemistry_choices(profile: Any) -> list[tuple[str, str]]:
    """Return Gradio choices ``(label, id)`` for this film."""
    chems = chemistries_map(profile)
    if chems:
        out = []
        for cid, c in chems.items():
            name = str(c.get("name", cid))
            family = c.get("curve_family")
            if isinstance(family, list) and len(family) >= 2:
                name = f"{name} · curve family"
            out.append((name, cid))
        return out
    return [(v["name"], k) for k, v in DEVELOPER_STYLES.items()]


def resolve_style(
    profile: Any, developer_id: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return ``(style_dict, chemistry_or_none)`` for curve / grain modifiers."""
    chem = get_chemistry(profile, developer_id)
    if chem is not None:
        style = character_style(str(chem.get("character", "balanced")))
        overrides = chem.get("character_overrides") or {}
        style.update({k: overrides[k] for k in overrides})
        style["name"] = str(chem.get("name", style.get("name", developer_id)))
        return style, chem
    if developer_id in DEVELOPER_STYLES:
        return dict(DEVELOPER_STYLES[developer_id]), None
    # Unknown id — balanced fallback
    style = character_style("balanced")
    style["name"] = developer_id
    return style, None

def _interp_ci(ci_points: list[list[float]], minutes: float) -> float:
    pts = np.asarray(ci_points, dtype=np.float64)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    x = pts[:, 0]
    y = pts[:, 1]
    if minutes <= float(x[0]):
        return float(y[0])
    if minutes >= float(x[-1]):
        return float(y[-1])
    return float(PchipInterpolator(x, y)(minutes))


def minutes_to_relative(chem: dict[str, Any], minutes: float) -> float:
    """Map tank minutes → relative development (1.0 = datasheet normal)."""
    normal = float(chem.get("normal_minutes", 1.0))
    normal = max(normal, 1e-3)
    minutes = float(minutes)
    ci_points = chem.get("ci_points")
    if ci_points and len(ci_points) >= 2:
        ci_now = _interp_ci(ci_points, minutes)
        ci_n = _interp_ci(ci_points, normal)
        if ci_n > 1e-6:
            return float(np.clip(ci_now / ci_n, 0.45, 2.2))
    return float(np.clip(minutes / normal, 0.45, 2.2))


def relative_to_minutes(chem: dict[str, Any], relative: float) -> float:
    """Approximate inverse: relative → minutes (for display / migration)."""
    normal = float(chem.get("normal_minutes", 1.0))
    relative = float(np.clip(relative, 0.45, 2.2))
    ci_points = chem.get("ci_points")
    if ci_points and len(ci_points) >= 2:
        target_ci = _interp_ci(ci_points, normal) * relative
        pts = np.asarray(ci_points, dtype=np.float64)
        order = np.argsort(pts[:, 1])
        # CI usually increases with time — invert via PCHIP on (ci, minutes)
        ci = pts[order, 1]
        mins = pts[order, 0]
        # Ensure strictly increasing CI for interpolator
        keep = np.concatenate([[True], np.diff(ci) > 1e-6])
        ci = ci[keep]
        mins = mins[keep]
        if len(ci) >= 2:
            target_ci = float(np.clip(target_ci, float(ci[0]), float(ci[-1])))
            return float(PchipInterpolator(ci, mins)(target_ci))
    return float(normal * relative)


def time_slider_bounds(chem: dict[str, Any]) -> tuple[float, float, float]:
    """Return ``(min, max, normal)`` minutes for UI slider."""
    normal = float(chem.get("normal_minutes", 8.0))
    tmin = float(chem.get("time_min", max(2.0, normal * 0.45)))
    tmax = float(chem.get("time_max", normal * 2.2))
    return tmin, tmax, normal


def resolve_relative_time(
    profile: Any,
    developer_id: str,
    *,
    development_minutes: float | None = None,
    relative_time: float | None = None,
) -> tuple[float, float | None, dict[str, Any]]:
    """Resolve working relative time and optional minutes for metadata.

    Prefer ``development_minutes`` when a chemistry is selected; otherwise use
    ``relative_time`` (legacy / abstract styles).
    """
    style, chem = resolve_style(profile, developer_id)
    if chem is not None and development_minutes is not None:
        rel = minutes_to_relative(chem, float(development_minutes))
        return rel, float(development_minutes), style
    if chem is not None and relative_time is not None:
        mins = relative_to_minutes(chem, float(relative_time))
        return float(relative_time), mins, style
    if chem is not None:
        normal = float(chem.get("normal_minutes", 1.0))
        return 1.0, normal, style
    rel = 1.0 if relative_time is None else float(relative_time)
    return rel, None, style