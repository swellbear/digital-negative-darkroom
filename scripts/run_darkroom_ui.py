#!/usr/bin/env python3
"""Darkroom UI: large commit-accurate live preview beside always-visible controls."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from gradio import SelectData
import numpy as np

from digital_negative.chemistry import (
    chemistry_choices,
    default_chemistry_id,
    get_chemistry,
    time_slider_bounds,
)
from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative
from digital_negative.display import (
    linear_to_srgb,
    negative_lightbox_preview,
    original_photo_preview,
    rotate_image,
    to_u8_gray,
)
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import print_negative

# Match commit look as closely as practical while staying interactive.
LIVE_MAX_SIDE = 2000
DRAG_MAX_SIDE = 1280  # high enough for critical judgment while dragging
INSPECT_MAX_SIDE = 3600  # high-res for zoom / inspect panel
REF_MAX_SIDE = 420

FILM_CHOICES = []
for path in list_film_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    FILM_CHOICES.append((f"{data['name']} (ISO {data['iso']})", data["id"]))

PAPER_CHOICES = []
for path in list_paper_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    PAPER_CHOICES.append((data["name"], data["id"]))

# Filled after helper defs below (film-specific chemistry dropdown + minutes).
_INIT_DEV_CHOICES: list = []
_INIT_DEV_ID = "standard"
_INIT_TMIN, _INIT_TMAX, _INIT_TNORM = 4.0, 16.0, 8.0

SAMPLE_DIR = ROOT / "samples" / "raws"
SAMPLE_CHOICES = [("Synthetic test scene (no file)", "")]
if SAMPLE_DIR.exists():
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}:
            SAMPLE_CHOICES.append((path.name, str(path)))

UI_CSS = """
.gradio-container {
  max-width: 100% !important;
  padding: 8px 12px !important;
}
/* Force controls | preview side-by-side; do not stack on typical laptop widths */
#main_workspace {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: flex-start !important;
  gap: 12px !important;
}
#main_workspace > div {
  min-width: 0 !important;
}
#controls_col {
  flex: 0 0 340px !important;
  width: 340px !important;
  max-width: 340px !important;
  position: sticky !important;
  top: 6px !important;
  max-height: calc(100vh - 16px) !important;
  /* Scroll the column; keep x visible so open dropdowns are not clipped */
  overflow-y: auto !important;
  overflow-x: visible !important;
  padding-right: 8px !important;
  align-self: flex-start !important;
  z-index: 30 !important;
}
#preview_col {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  z-index: 1 !important;
}
/* Compact control density */
#controls_col .block {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
  padding: 0 !important;
  overflow: visible !important;
}
#controls_col .label-wrap,
#controls_col label,
#controls_col .svelte-1b6s6s {
  margin-bottom: 0 !important;
  font-size: 0.82rem !important;
}
#controls_col .form {
  gap: 4px !important;
  overflow: visible !important;
}
#controls_col button {
  min-height: 34px !important;
  font-size: 0.9rem !important;
}
/* Accordion: never clip body — show full Develop/Print/Inspect controls */
#controls_col .accordion,
#controls_col .accordion > .wrap,
#controls_col .accordion > div,
#preview_col .accordion,
#preview_col .accordion > .wrap,
#preview_col .accordion > div {
  overflow: visible !important;
  max-height: none !important;
}
#controls_col .accordion .wrap,
#preview_col .accordion .wrap {
  height: auto !important;
}
#controls_col .accordion {
  margin-bottom: 6px !important;
}
#controls_col .accordion > .label-wrap {
  padding: 6px 8px !important;
  font-size: 0.9rem !important;
}
/* Open dropdown lists (film / developer / paper) must escape parents */
#controls_col [role="listbox"],
#controls_col ul.options,
#controls_col .options,
#controls_col .secondary-wrap {
  z-index: 9999 !important;
}
#controls_col [role="listbox"],
#controls_col ul.options {
  max-height: min(55vh, 420px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}
/* Avoid clipping long chemistry names in the closed dropdown */
#controls_col .wrap.fullscreen-allowed,
#controls_col .container {
  overflow: visible !important;
}
#app_header h1 {
  font-size: 1.35rem !important;
  margin: 0 0 2px 0 !important;
}
#app_header p, #app_header {
  margin: 0 0 6px 0 !important;
  font-size: 0.88rem !important;
  line-height: 1.35 !important;
}
#ritual_status {
  padding: 6px 8px !important;
  margin-bottom: 6px !important;
  border-left: 3px solid #c45c26;
  background: linear-gradient(90deg, rgba(196,92,38,0.12), transparent);
  font-size: 0.82rem !important;
  line-height: 1.3 !important;
  max-height: 7.5rem;
  overflow-y: auto;
}
#history_box {
  margin-top: 4px !important;
  padding: 6px 8px !important;
  border: 1px solid rgba(128,128,128,0.25);
  background: rgba(0,0,0,0.08);
  font-size: 0.8rem !important;
  max-height: 140px;
  overflow-y: auto;
}
#live_preview {
  min-height: 0 !important;
}
#live_preview .image-frame,
#live_preview .image-container,
#inspect_preview .image-frame,
#inspect_preview .image-container {
  overflow: auto !important;
  max-height: calc(100vh - 160px) !important;
  background: #0c0c0c !important;
}
#live_preview img,
#live_preview .image-container img,
#live_preview .image-frame img {
  max-height: calc(100vh - 160px) !important;
  width: auto !important;
  max-width: 100% !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
  transform-origin: center center;
  cursor: zoom-in;
}
#inspect_preview {
  min-height: 60vh !important;
}
#inspect_preview img,
#inspect_preview .image-container img,
#inspect_preview .image-frame img {
  max-height: none !important;
  max-width: none !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
  transform-origin: center center;
  cursor: grab;
}
#inspect_hint {
  font-size: 0.8rem !important;
  opacity: 0.85;
  margin: 2px 0 6px 0 !important;
}
#ref_row {
  gap: 6px !important;
  margin-top: 6px !important;
}
#ref_row img {
  max-height: 88px !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
}
@media (max-width: 900px) {
  #main_workspace { flex-wrap: wrap !important; }
  #controls_col {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    position: relative !important;
    max-height: none !important;
  }
}
"""

# Wheel / trackpad zoom + drag pan for main and inspect viewers
UI_JS = """
() => {
  function enhance(sel) {
    const root = document.querySelector(sel);
    if (!root || root.dataset.zoomReady === '1') return;
    const findImg = () => root.querySelector('img');
    let scale = 1;
    let panX = 0, panY = 0;
    let dragging = false, lastX = 0, lastY = 0;

    const apply = (img) => {
      if (!img) return;
      img.style.transformOrigin = 'center center';
      img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
      img.style.maxWidth = scale > 1.02 ? 'none' : '';
      img.style.maxHeight = scale > 1.02 ? 'none' : '';
      img.style.cursor = scale > 1.02 ? (dragging ? 'grabbing' : 'grab') : 'zoom-in';
    };

    root.addEventListener('wheel', (e) => {
      const img = findImg();
      if (!img) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      scale = Math.min(10, Math.max(0.4, scale * factor));
      if (scale <= 1.02) { panX = 0; panY = 0; }
      apply(img);
    }, { passive: false });

    root.addEventListener('pointerdown', (e) => {
      const img = findImg();
      if (!img || scale <= 1.02) return;
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      root.setPointerCapture?.(e.pointerId);
      apply(img);
    });
    root.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const img = findImg();
      if (!img) return;
      panX += e.clientX - lastX;
      panY += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      apply(img);
    });
    const endDrag = () => {
      dragging = false;
      apply(findImg());
    };
    root.addEventListener('pointerup', endDrag);
    root.addEventListener('pointercancel', endDrag);
    root.addEventListener('dblclick', () => {
      scale = 1; panX = 0; panY = 0;
      apply(findImg());
    });

    const mo = new MutationObserver(() => {
      scale = 1; panX = 0; panY = 0;
      apply(findImg());
    });
    mo.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    root.dataset.zoomReady = '1';
    apply(findImg());
  }

  const boot = () => {
    enhance('#live_preview');
    enhance('#inspect_preview');
  };
  boot();
  new MutationObserver(boot).observe(document.body, { childList: true, subtree: true });
}
"""


def _profile_path(paths, profile_id: str) -> Path:
    for p in paths:
        if json.loads(p.read_text(encoding="utf-8"))["id"] == profile_id:
            return p
    raise FileNotFoundError(profile_id)


def _film_profile(film_id: str):
    return load_film_profile(_profile_path(list_film_profiles(), film_id))


def _chem_time_update(film_id: str, developer_id: str, *, reset_to_normal: bool = True):
    """Gradio updates for developer dropdown / minutes slider."""
    profile = _film_profile(film_id)
    chem = get_chemistry(profile, developer_id)
    if chem is None:
        label = "Dev time (rel. ×8 min stand-in)"
        return gr.update(minimum=4.0, maximum=16.0, value=8.0, label=label, step=0.25)
    tmin, tmax, normal = time_slider_bounds(chem)
    family = chem.get("curve_family") or []
    if isinstance(family, list) and len(family) >= 2:
        times = ", ".join(f"{float(m['minutes']):g}" for m in sorted(family, key=lambda x: x["minutes"]))
        label = f"Dev time (min) · N={normal:g} · family [{times}]"
    else:
        label = f"Dev time (min) · N={normal:g} @ 20°C (morph)"
    return gr.update(minimum=tmin, maximum=tmax, value=normal, label=label, step=0.25)


def on_film_change(film_id: str):
    profile = _film_profile(film_id)
    chem_id = default_chemistry_id(profile)
    choices = chemistry_choices(profile)
    return (
        gr.update(choices=choices, value=chem_id),
        _chem_time_update(film_id, chem_id, reset_to_normal=True),
    )


def on_developer_change(film_id: str, developer_id: str):
    return _chem_time_update(film_id, developer_id, reset_to_normal=True)


# Initial Develop controls (film-specific chemistry + datasheet normal minutes).
if FILM_CHOICES:
    _boot_film = FILM_CHOICES[0][1]
    _boot_profile = _film_profile(_boot_film)
    _INIT_DEV_CHOICES = chemistry_choices(_boot_profile)
    _INIT_DEV_ID = default_chemistry_id(_boot_profile)
    _boot_chem = get_chemistry(_boot_profile, _INIT_DEV_ID) or {
        "normal_minutes": 8.0,
        "time_min": 4.0,
        "time_max": 16.0,
    }
    _INIT_TMIN, _INIT_TMAX, _INIT_TNORM = time_slider_bounds(_boot_chem)


def _resolve_input(file_obj, sample_path: str | None) -> str | None:
    """Prefer an uploaded file over the sample dropdown when both are set."""
    upload_path = None
    if file_obj is not None:
        if isinstance(file_obj, str):
            upload_path = file_obj
        else:
            upload_path = getattr(file_obj, "name", None) or str(file_obj)
        if upload_path:
            return upload_path
    if sample_path:
        return sample_path
    return None


def _to_rgb_u8(gray_float: np.ndarray, *, assume_linear: bool = False) -> np.ndarray:
    view = linear_to_srgb(gray_float) if assume_linear else gray_float
    g = to_u8_gray(view)
    return np.stack([g, g, g], axis=-1)


def _downscale_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return np.ascontiguousarray(rgb[::step, ::step])


def _proxy_dn(dn: DigitalNegative, max_side: int = LIVE_MAX_SIDE) -> DigitalNegative:
    img = dn.image
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    if step == 1:
        return dn
    return DigitalNegative(
        image=np.ascontiguousarray(img[::step, ::step]),
        metadata=copy.deepcopy(dn.metadata),
    )


def _locked(state, stage: str) -> bool:
    if not state or state.get("dn") is None:
        return False
    return stage in state["dn"].metadata.get("ui_state", {}).get("locked_stages", [])


def _history_md(dn) -> str:
    hist = dn.metadata.get("history", [])
    lines = ["### Decision log", "_Locked decisions only — exploring does not write here._", ""]
    if not hist:
        lines.append("_No locked decisions yet. Commit a stage to record it._")
    for i, h in enumerate(hist, 1):
        op = h.get("op", "?")
        if op == "ingest":
            lines.append(f"{i}. **Ingest** — `{h.get('source')}`")
        elif op == "develop":
            chem = h.get("developer_name") or h.get("developer_id")
            if h.get("development_minutes") is not None:
                time_bit = f"{float(h['development_minutes']):g} min"
            else:
                time_bit = f"rel={h.get('relative_time')}"
            lines.append(
                f"{i}. **Develop** — `{h.get('film_profile_id')}` · {chem} · "
                f"{time_bit} · N±={h.get('contrast_modifier')} · "
                f"grain={h.get('grain_strength')}"
            )
        elif op == "print":
            lines.append(
                f"{i}. **Print** — `{h.get('paper_id')}` · grade {h.get('grade')} · "
                f"exp {h.get('overall_exposure'):+g} stops"
                + (f" · nudge={h.get('contrast')}" if h.get("contrast") not in (None, 0, 0.0) else "")
            )
        elif op == "unlock":
            stage = h.get("stage", "?")
            label = {"development": "Develop", "print": "Print", "ingest": "Ingest"}.get(stage, stage)
            lines.append(f"{i}. **← Unlocked {label}** — previous lock opened for revision")
        elif op == "rotate":
            lines.append(
                f"{i}. **Rotate** — {h.get('degrees_cw'):+g}° "
                f"(total {h.get('total_degrees')}° CW)"
            )
        else:
            lines.append(f"{i}. **{op}**")
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lock_labels = []
    for s in ("ingest", "development", "print"):
        if s in locks:
            lock_labels.append({"ingest": "Ingest", "development": "Develop", "print": "Print"}[s])
    lines.append("")
    lines.append(
        f"**Currently locked:** {', '.join(lock_labels) or '—'}  \n"
        f"**Process seed:** `{dn.metadata.get('process_seed')}` "
        f"_(mild tank variation; same seed = repeatable)_"
    )
    return "\n".join(lines)


def _stage_banner(stage: str, locked: list | None = None) -> str:
    """Ritual progress: which stage you're working, which are locked."""
    steps = [("ingest", "Ingest"), ("development", "Develop"), ("print", "Print")]
    order = {"ingest": 0, "development": 1, "print": 2}
    cur = order.get(stage, -1)
    locked_set = set(locked or [])
    parts = []
    for i, (key, label) in enumerate(steps):
        n = i + 1
        done = key in locked_set
        if i == cur and not done:
            parts.append(f"**{n}. {label} — working**")
        elif i == cur and done:
            parts.append(f"**{n}. {label} — locked**")
        elif done:
            parts.append(f"{n}. {label} — locked")
        else:
            parts.append(f"{n}. {label}")
    return " → ".join(parts)


def _locks(state) -> list:
    if not state or state.get("dn") is None:
        return []
    return list(state["dn"].metadata.get("ui_state", {}).get("locked_stages", []))


def _split_summary(full: str) -> tuple[str, str]:
    """Split status blurb from decision log for separate UI panels."""
    if "### Decision log" in full:
        status, hist = full.split("### Decision log", 1)
        return status.strip(), "### Decision log" + hist
    if "### Decision history" in full:
        status, hist = full.split("### Decision history", 1)
        return status.strip(), "### Decision log" + hist
    return full, ""


_VIEWER_LABELS = {
    "live": "Commit preview (live) — theoretical print",
    "original": "Original photo (enlarged) — click Live print to return",
    "latent": "Latent DN (enlarged) — click Live print to return",
    "negative": "Developed negative (enlarged) — click Live print to return",
}


def _viewer_frame(state, live=None, original=None, latent=None, neg=None):
    """Large preview image + label for the current viewer mode."""
    mode = (state or {}).get("viewer_mode", "live")
    if mode == "original":
        img = (state or {}).get("original_view")
        if img is None:
            img = original if original is not None else (state or {}).get("original_ref")
    elif mode == "latent":
        img = (state or {}).get("latent_view")
        if img is None:
            img = latent if latent is not None else (state or {}).get("latent_ref")
    elif mode == "negative":
        img = (state or {}).get("neg_view")
        if img is None:
            img = neg if neg is not None else (state or {}).get("neg_ref")
    else:
        mode = "live"
        img = live if live is not None else (state or {}).get("live_rgb")
    return gr.update(value=img, label=_VIEWER_LABELS.get(mode, _VIEWER_LABELS["live"]))


def _inspect_frame(state, live=None):
    """High-res inspect panel image for zooming the active sequence stage."""
    mode = (state or {}).get("viewer_mode", "live")
    if mode == "original":
        img = (state or {}).get("original_inspect")
        if img is None:
            img = (state or {}).get("original_view")
    elif mode == "latent":
        img = (state or {}).get("latent_inspect")
        if img is None:
            img = (state or {}).get("latent_view")
    elif mode == "negative":
        img = (state or {}).get("neg_inspect")
        if img is None:
            img = (state or {}).get("neg_view")
    else:
        mode = "live"
        img = (state or {}).get("live_inspect")
        if img is None:
            img = live if live is not None else (state or {}).get("live_rgb")
    label = {
        "live": "Inspect — Live theoretical print (scroll-wheel zoom, drag to pan, double-click reset)",
        "original": "Inspect — Original (scroll-wheel zoom, drag to pan, double-click reset)",
        "latent": "Inspect — Latent DN (scroll-wheel zoom, drag to pan, double-click reset)",
        "negative": "Inspect — Developed negative (scroll-wheel zoom, drag to pan, double-click reset)",
    }.get(mode, "Inspect")
    return gr.update(value=img, label=label)


def _pack_preview(live, original, latent, neg, summary, state):
    status, hist = _split_summary(summary or "")
    if state is not None:
        if live is not None:
            state = {**state, "live_rgb": live, "live_inspect": live}
        if neg is not None:
            state = {
                **state,
                "neg_ref": neg,
                "neg_view": neg,
                "neg_inspect": state.get("neg_inspect")
                if state.get("neg_inspect") is not None
                else neg,
            }
    shown = _viewer_frame(state, live=live, original=original, latent=latent, neg=neg)
    return shown, original, latent, neg, status, hist, _inspect_frame(state, live=live), state


def focus_viewer(mode: str):
    """Return a handler that puts a reference (or live print) in the large preview."""

    def _fn(state, evt: SelectData | None = None):
        if not state or state.get("dn") is None:
            empty = gr.update()
            return empty, empty, "*Commit Ingest first.*", gr.update(open=False), state
        if evt is not None and getattr(evt, "selected", True) is False:
            mode_use = "live"
        else:
            mode_use = mode
        state = {**state, "viewer_mode": mode_use}
        tip = {
            "live": "_Large + Inspect: **live theoretical print**. Scroll-wheel zooms; drag pans._",
            "original": "_Large + Inspect: **Original**. Scroll-wheel zooms; drag pans._",
            "latent": "_Large + Inspect: **Latent DN**. Scroll-wheel zooms; drag pans._",
            "negative": "_Large + Inspect: **Developed negative**. Scroll-wheel zooms; drag pans._",
        }.get(mode_use, "")
        banner = _stage_banner(state.get("stage", "development"), _locks(state))
        status = f"{banner}\n\n{tip}"
        return (
            _viewer_frame(state),
            _inspect_frame(state),
            status,
            gr.update(open=True),
            state,
        )

    return _fn


def focus_viewer_button(mode: str):
    def _fn(state):
        return focus_viewer(mode)(state, None)

    return _fn


def _rebuild_views_from_dn(state: dict) -> dict:
    """Refresh latent / proxy caches from the current DN image."""
    dn = state["dn"]
    latent_full = _to_rgb_u8(dn.to_luminance(), assume_linear=True)
    latent_view = _downscale_rgb(latent_full, LIVE_MAX_SIDE)
    latent_inspect = _downscale_rgb(latent_full, INSPECT_MAX_SIDE)
    latent_ref = _downscale_rgb(latent_full, REF_MAX_SIDE)
    return {
        **state,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "latent_view": latent_view,
        "latent_inspect": latent_inspect,
        "latent_ref": latent_ref,
        "live_rgb": latent_view if state.get("development") is None else state.get("live_rgb"),
        "live_inspect": latent_inspect if state.get("development") is None else state.get("live_inspect"),
        "development": None,
        "development_full": None,
        "transmittance_proxy": None,
        "print": None,
        "print_draft": None,
        "neg_ref": None,
        "neg_view": None,
    }


def rotate_working(turns_cw: int, state):
    """Rotate the Digital Negative and reference previews; clears Develop/Print locks."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")

    dn = state["dn"]
    dn.image = rotate_image(dn.image, turns_cw)
    ingest = dn.metadata.setdefault("ingest", {})
    degrees = int(ingest.get("rotation_degrees", 0)) + (90 * int(turns_cw))
    ingest["rotation_degrees"] = degrees % 360

    for key in (
        "original_view",
        "original_ref",
        "original_inspect",
        "latent_view",
        "latent_ref",
        "latent_inspect",
        "live_rgb",
        "live_inspect",
    ):
        if state.get(key) is not None:
            state[key] = rotate_image(state[key], turns_cw)

    ui = dn.metadata.setdefault("ui_state", {})
    locks = ui.setdefault("locked_stages", [])
    committed = ui.setdefault("committed_stages", [])
    for stage in ("print", "development"):
        if stage in locks:
            locks.remove(stage)
        if stage in committed:
            committed.remove(stage)
    dn.metadata.setdefault("history", []).append(
        {
            "op": "rotate",
            "degrees_cw": 90 * int(turns_cw),
            "total_degrees": ingest["rotation_degrees"],
        }
    )
    dn.touch()

    state = _rebuild_views_from_dn(
        {
            **state,
            "dn": dn,
            "viewer_mode": "live",
            "stage": "development",
            "original_view": state.get("original_view"),
            "original_ref": state.get("original_ref"),
        }
    )
    deg = ingest["rotation_degrees"]
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Rotated {90 * int(turns_cw):+d}°** (total {deg}° CW).  \n"
        f"_Develop/Print unlocked — check orientation, then Commit Develop._\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    return (
        _viewer_frame(state, live=state.get("live_rgb")),
        state.get("original_ref"),
        state.get("latent_ref"),
        None,
        *_split_summary(summary),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=True),
        gr.update(open=True),
        state,
    )


def rotate_cw(state):
    return rotate_working(1, state)


def rotate_ccw(state):
    return rotate_working(-1, state)


def rotate_180(state):
    return rotate_working(2, state)


def commit_ingest(sample_path, file_obj, state):
    path = _resolve_input(file_obj, sample_path)
    dn = ingest_path(path or None)
    ui = dn.metadata.setdefault("ui_state", {})
    stages = ui.setdefault("committed_stages", [])
    locks = ui.setdefault("locked_stages", [])
    if "ingest" not in stages:
        stages.append("ingest")
    if "ingest" not in locks:
        locks.append("ingest")
    ui["current_stage"] = "ingest"
    dn.metadata.setdefault("history", []).append(
        {
            "op": "ingest",
            "source": dn.metadata["source"]["original_filename"],
            "working_space": dn.metadata.get("ingest", {}).get("working_space"),
        }
    )

    latent_full = _to_rgb_u8(dn.to_luminance(), assume_linear=True)
    latent_view = _downscale_rgb(latent_full, LIVE_MAX_SIDE)
    latent_inspect = _downscale_rgb(latent_full, INSPECT_MAX_SIDE)
    latent_ref = _downscale_rgb(latent_full, REF_MAX_SIDE)
    original_full = original_photo_preview(path, dn_image=dn.image)
    original_view = _downscale_rgb(original_full, LIVE_MAX_SIDE)
    original_inspect = _downscale_rgb(original_full, INSPECT_MAX_SIDE)
    original_ref = _downscale_rgb(original_full, REF_MAX_SIDE)
    summary = (
        f"{_stage_banner('development', ['ingest'])}\n\n"
        f"**Ingest locked** — `{dn.metadata['source']['original_filename']}`  \n"
        f"_Open a sequence below, then use **Inspect & zoom** (scroll-wheel zoom, drag pan)._\n\n"
        f"{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "original_ref": original_ref,
        "original_view": original_view,
        "original_inspect": original_inspect,
        "latent_ref": latent_ref,
        "latent_view": latent_view,
        "latent_inspect": latent_inspect,
        "neg_ref": None,
        "neg_view": None,
        "neg_inspect": None,
        "live_rgb": latent_view,
        "live_inspect": latent_inspect,
        "viewer_mode": "live",
        "development": None,
        "development_full": None,
        "stage": "development",
        "summary_cache": summary,
        "source_path": path,
    }
    return (
        gr.update(value=latent_view, label=_VIEWER_LABELS["live"]),
        original_ref,
        latent_ref,
        None,
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(open=False),  # collapse Ingest
        gr.update(open=True),   # show Develop
        gr.update(open=True),   # keep Print reachable
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        _inspect_frame(state, live=latent_inspect),
        gr.update(open=True),
        state,
    )


def _run_live_develop_then_print(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
    *,
    max_side: int = LIVE_MAX_SIDE,
):
    """Develop with current settings, then print with current/default print settings.

    Large viewer shows the theoretical final print — what you'd get after
    Commit Develop + Commit Print with these controls.
    """
    if max_side <= DRAG_MAX_SIDE:
        proxy = state.get("proxy_drag") or _proxy_dn(state["dn"], DRAG_MAX_SIDE)
    else:
        proxy = state.get("proxy") or _proxy_dn(state["dn"], LIVE_MAX_SIDE)
    proxy.metadata["process_seed"] = state["dn"].metadata.get("process_seed")
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        proxy,
        profile,
        development_minutes=float(development_minutes),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
        process_variation=1.0,
        commit=False,
    )
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    printed = print_negative(
        development.transmittance,
        state["dn"],
        paper,
        overall_exposure=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        commit=False,
    )
    live_rgb = _to_rgb_u8(printed.preview)
    neg_full = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    speed = state["dn"].metadata.get("print", {}).get("filtration", {}).get("values", {}).get(
        "filter_speed", 1.0
    )
    quality_note = "drag" if max_side <= DRAG_MAX_SIDE else "hq"
    curve_src = proxy.metadata.get("development", {}).get("curve_source", "?")
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Live print** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note})  \n"
        f"{profile.name} · {developer_id} · {float(development_minutes):g} min · "
        f"curve={curve_src} · N±={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
        f"{paper.name} · g{float(print_grade):.1f} · exp {float(print_exposure):+.2f} "
        f"· ×{float(speed):.2f}\n\n"
        f"{_history_md(state['dn'])}"
    )
    state = {
        **state,
        "proxy": state.get("proxy") or _proxy_dn(state["dn"], LIVE_MAX_SIDE),
        "proxy_drag": state.get("proxy_drag") or _proxy_dn(state["dn"], DRAG_MAX_SIDE),
        "development": development,
        "live_rgb": live_rgb,
        "live_inspect": live_rgb,
        "neg_ref": neg_ref,
        "neg_view": neg_view,
        "neg_inspect": neg_inspect,
        "stage": "development",
        "summary_cache": summary,
        "draft_print": {
            "paper_id": paper_id,
            "print_exposure": float(print_exposure),
            "print_grade": float(print_grade),
            "print_contrast": float(print_contrast),
        },
    }
    return live_rgb, neg_ref, summary, state


def live_preview(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
    quality: str = "high",
):
    """Unified live viewer: develop+print while developing; print-only after Develop lock.

    quality='drag' uses a faster lower-res proxy while sliders move;
    quality='high' (release / change) uses commit-accurate resolution.
    """
    max_side = DRAG_MAX_SIDE if quality == "drag" else LIVE_MAX_SIDE

    if not state or state.get("dn") is None:
        return _pack_preview(None, None, None, None, "*Commit Ingest first.*", state)

    if _locked(state, "print"):
        return _pack_preview(
            state.get("live_rgb"),
            state.get("original_ref"),
            state.get("latent_ref"),
            state.get("neg_ref"),
            state.get("summary_cache", ""),
            state,
        )

    if _locked(state, "development"):
        # Print-only commit preview
        if state.get("development_full") is not None:
            t = state["development_full"].transmittance
            step = max(1, int(np.ceil(max(t.shape) / max_side)))
            t = np.ascontiguousarray(t[::step, ::step])
        else:
            t = state.get("transmittance_proxy")
            if t is None and state.get("development") is not None:
                t = state["development"].transmittance
            if t is None:
                return _pack_preview(
                    state.get("live_rgb"),
                    state.get("original_ref"),
                    state.get("latent_ref"),
                    state.get("neg_ref"),
                    state.get("summary_cache", ""),
                    state,
                )
            if max(t.shape) > max_side:
                step = max(1, int(np.ceil(max(t.shape) / max_side)))
                t = np.ascontiguousarray(t[::step, ::step])

        paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
        result = print_negative(
            t,
            state["dn"],
            paper,
            overall_exposure=float(print_exposure),
            grade=float(print_grade),
            contrast=float(print_contrast),
            commit=False,
        )
        live_rgb = _to_rgb_u8(result.preview)
        speed = state["dn"].metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
        quality_note = "drag" if quality == "drag" else "hq"
        summary = (
            f"{_stage_banner('print', _locks(state))}\n\n"
            f"**Print preview** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note})  \n"
            f"{paper.name} · g{float(print_grade):.1f} · exp {float(print_exposure):+.2f} · "
            f"×{float(speed):.2f}\n\n{_history_md(state['dn'])}"
        )
        state = {**state, "print_draft": result, "live_rgb": live_rgb, "summary_cache": summary}
        return _pack_preview(
            live_rgb,
            state.get("original_ref"),
            state.get("latent_ref"),
            state.get("neg_ref"),
            summary,
            state,
        )

    # Develop unlocked: show print through the working negative
    live_rgb, neg_ref, summary, state = _run_live_develop_then_print(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        max_side=max_side,
    )
    return _pack_preview(
        live_rgb, state.get("original_ref"), state.get("latent_ref"), neg_ref, summary, state
    )


def live_preview_drag(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
):
    return live_preview(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        quality="drag",
    )


def live_preview_high(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
):
    return live_preview(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        quality="high",
    )


def commit_develop(film_id, developer_id, development_minutes, contrast, grain, state):
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    if _locked(state, "development"):
        raise gr.Error("Develop is already locked.")

    dn = state["dn"]
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        dn,
        profile,
        development_minutes=float(development_minutes),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
        process_variation=1.0,
        commit=True,
    )
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "development" not in locks:
        locks.append("development")

    step = max(1, int(np.ceil(max(development.transmittance.shape) / LIVE_MAX_SIDE)))
    t_proxy = np.ascontiguousarray(development.transmittance[::step, ::step])

    # Keep last theoretical print on screen until .then refreshes with Print controls;
    # fall back to positive if no live print was generated yet.
    # Do not use `or` — live_rgb is a numpy array (ambiguous truth value).
    live_view = state.get("live_rgb")
    if live_view is None:
        live_view = _downscale_rgb(
            _to_rgb_u8(development.positive_preview), LIVE_MAX_SIDE
        )
    neg_full = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Develop locked** — refine Print below, then Commit Print.\n\n{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": state.get("proxy"),
        "proxy_drag": state.get("proxy_drag"),
        "original_ref": state.get("original_ref"),
        "original_view": state.get("original_view"),
        "original_inspect": state.get("original_inspect"),
        "latent_ref": state.get("latent_ref"),
        "latent_view": state.get("latent_view"),
        "latent_inspect": state.get("latent_inspect"),
        "neg_ref": neg_ref,
        "neg_view": neg_view,
        "neg_inspect": neg_inspect,
        "live_rgb": live_view,
        "live_inspect": live_view,
        "viewer_mode": state.get("viewer_mode", "live"),
        "development": development,
        "development_full": development,
        "transmittance_proxy": t_proxy,
        "stage": "print",
        "summary_cache": summary,
        "source_path": state.get("source_path"),
    }
    return (
        _viewer_frame(state, live=live_view, neg=neg_view),
        state.get("original_ref"),
        state.get("latent_ref"),
        neg_ref,
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(open=False),  # collapse Develop
        gr.update(open=True),   # show Print
        state,
    )


def commit_print(paper_id, print_exposure, print_grade, print_contrast, state):
    if not state or state.get("development_full") is None:
        raise gr.Error("Commit Develop first.")
    if _locked(state, "print"):
        raise gr.Error("Print is already locked.")

    dn = state["dn"]
    development = state["development_full"]
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    result = print_negative(
        development.transmittance,
        dn,
        paper,
        overall_exposure=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        commit=True,
    )
    stages = dn.metadata["ui_state"].setdefault("committed_stages", [])
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "print" not in stages:
        stages.append("print")
    if "print" not in locks:
        locks.append("print")

    live_rgb = _downscale_rgb(_to_rgb_u8(result.preview), LIVE_MAX_SIDE)
    speed = dn.metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Print locked** — {paper.name} · g{float(print_grade):.1f} · "
        f"exp {float(print_exposure):+.2f}\n\n{_history_md(dn)}"
    )
    state = {**state, "print": result, "live_rgb": live_rgb, "summary_cache": summary}
    return (
        live_rgb,
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        state,
    )


def _unlock_stage(dn, stage: str) -> None:
    ui = dn.metadata.setdefault("ui_state", {})
    locks = ui.setdefault("locked_stages", [])
    committed = ui.setdefault("committed_stages", [])
    if stage in locks:
        locks.remove(stage)
    if stage in committed and stage != "ingest":
        committed.remove(stage)
    dn.metadata.setdefault("history", []).append({"op": "unlock", "stage": stage})
    dn.touch()


def unlock_develop(state):
    if not state or state.get("dn") is None:
        raise gr.Error("Nothing to unlock — Commit Ingest first.")
    if not _locked(state, "development"):
        raise gr.Error("Develop is not locked.")

    dn = state["dn"]
    if _locked(state, "print"):
        _unlock_stage(dn, "print")
    _unlock_stage(dn, "development")

    state = {
        **state,
        "development_full": None,
        "transmittance_proxy": None,
        "print": None,
        "print_draft": None,
        "stage": "development",
    }
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Develop unlocked** — adjust Develop below, then Commit again.\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        on,
        off,
        off,
        on,
        on,
        on,
        on,
        off,
        gr.update(open=True),
        gr.update(open=True),
        state,
    )


def unlock_print(state):
    if not state or state.get("dn") is None:
        raise gr.Error("Nothing to unlock.")
    if not _locked(state, "print"):
        raise gr.Error("Print is not locked.")
    if not _locked(state, "development"):
        raise gr.Error("Develop must stay committed to revise Print.")

    dn = state["dn"]
    _unlock_stage(dn, "print")
    state = {**state, "print": None, "stage": "print"}
    summary = (
        f"{_stage_banner('print', _locks(state))}\n\n"
        f"**Print unlocked** — adjust Print below, then Commit Print.\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        off,
        gr.update(open=True),
        state,
    )


def reset_session():
    summary = (
        "**1. Ingest — working** → 2. Develop → 3. Print\n\n"
        "*Commit Ingest to begin.*"
    )
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        None,
        None,
        None,
        None,
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        on,
        on,
        on,
        off,
        off,
        on,
        on,
        on,
        on,
        off,
        off,
        gr.update(open=True),
        gr.update(open=True),
        gr.update(open=True),
        off,
        off,
        off,
        None,
    )


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        gr.Markdown(
            """
            # Digital Negative Darkroom
            **Ingest → Develop → Print** · Commit locks · Unlock revises · Large image = theoretical print
            """,
            elem_id="app_header",
        )

        with gr.Row(elem_id="main_workspace", equal_height=False):
            with gr.Column(scale=0, elem_id="controls_col", min_width=300):
                status = gr.Markdown(
                    "**1. Ingest — working** → 2. Develop → 3. Print  \n"
                    "_Commit Ingest to begin._",
                    elem_id="ritual_status",
                )
                with gr.Accordion("1 · Ingest", open=True) as ingest_acc:
                    sample = gr.Dropdown(
                        choices=SAMPLE_CHOICES,
                        value=default_sample,
                        label="Sample (if no upload)",
                    )
                    file_in = gr.File(
                        label="Upload (overrides sample)",
                        file_types=[
                            ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2",
                            ".tif", ".tiff", ".jpg", ".jpeg", ".png",
                        ],
                        height=56,
                    )
                    ingest_btn = gr.Button("Commit Ingest", variant="primary", size="sm")

                with gr.Accordion("2 · Develop", open=True) as develop_acc:
                    film = gr.Dropdown(
                        choices=FILM_CHOICES,
                        value=FILM_CHOICES[0][1] if FILM_CHOICES else None,
                        label="Film",
                    )
                    developer = gr.Dropdown(
                        choices=_INIT_DEV_CHOICES,
                        value=_INIT_DEV_ID,
                        label="Developer",
                    )
                    development_minutes = gr.Slider(
                        _INIT_TMIN,
                        _INIT_TMAX,
                        value=_INIT_TNORM,
                        step=0.25,
                        label=f"Dev time (min) · N={_INIT_TNORM:g} @ 20°C",
                    )
                    contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast N− / N+")
                    grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain")
                    with gr.Row():
                        develop_btn = gr.Button(
                            "Commit Develop", interactive=False, variant="primary", size="sm"
                        )
                        unlock_develop_btn = gr.Button("Unlock", interactive=False, size="sm")

                with gr.Accordion("3 · Print", open=True) as print_acc:
                    paper = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value=PAPER_CHOICES[0][1] if PAPER_CHOICES else None,
                        label="Paper",
                    )
                    print_exposure = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Exposure (stops)")
                    print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="MG filtration")
                    print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Filter nudge")
                    with gr.Row():
                        print_btn = gr.Button(
                            "Commit Print", interactive=False, variant="primary", size="sm"
                        )
                        unlock_print_btn = gr.Button("Unlock", interactive=False, size="sm")

                reset_btn = gr.Button("New negative", size="sm")
                with gr.Accordion("Decision log", open=False):
                    history = gr.Markdown(
                        "_Locked decisions only — exploring does not write here._",
                        elem_id="history_box",
                    )

            with gr.Column(scale=1, elem_id="preview_col", min_width=480):
                live_out = gr.Image(
                    label="Commit preview (live) — theoretical print",
                    type="numpy",
                    elem_id="live_preview",
                    height=620,
                    buttons=["fullscreen", "download"],
                )
                with gr.Row():
                    rotate_ccw_btn = gr.Button("Rotate ⟲ 90°", size="sm", interactive=False)
                    rotate_180_btn = gr.Button("Rotate 180°", size="sm", interactive=False)
                    rotate_cw_btn = gr.Button("Rotate 90° ⟳", size="sm", interactive=False)
                with gr.Row(elem_id="ref_row"):
                    original_out = gr.Image(
                        label="Original (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                    latent_out = gr.Image(
                        label="Latent DN (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                    neg_out = gr.Image(
                        label="Negative (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                with gr.Row():
                    view_orig_btn = gr.Button("Original", size="sm")
                    view_lat_btn = gr.Button("Latent DN", size="sm")
                    view_neg_btn = gr.Button("Negative", size="sm")
                    view_live_btn = gr.Button("Live print", size="sm", variant="primary")
                with gr.Accordion("Inspect & zoom", open=True) as inspect_acc:
                    gr.Markdown(
                        "Scroll-wheel zooms · drag pans when zoomed · double-click resets · "
                        "fullscreen button for an even larger view",
                        elem_id="inspect_hint",
                    )
                    inspect_out = gr.Image(
                        label="Inspect — pick a sequence above",
                        type="numpy",
                        elem_id="inspect_preview",
                        height=720,
                        buttons=["fullscreen", "download"],
                    )

        # Always pass develop + print controls so the large viewer can show a
        # theoretical print through the working negative while developing.
        preview_inputs = [
            film,
            developer,
            development_minutes,
            contrast,
            grain,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            state,
        ]
        preview_outputs = [
            live_out, original_out, latent_out, neg_out, status, history, inspect_out, state
        ]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                sample, file_in, ingest_btn, develop_btn, print_btn,
                ingest_acc, develop_acc, print_acc,
                rotate_ccw_btn, rotate_180_btn, rotate_cw_btn,
                inspect_out, inspect_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        for ctrl in (
            development_minutes,
            contrast,
            grain,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
        ):
            # Drag = fast lower-res; release/change = commit-quality preview
            ctrl.input(fn=live_preview_drag, inputs=preview_inputs, outputs=preview_outputs)
            ctrl.change(fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs)

        # Film / developer swap chemistry list + datasheet-normal minutes, then refresh.
        film.change(
            fn=on_film_change,
            inputs=[film],
            outputs=[developer, development_minutes],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        developer.change(
            fn=on_developer_change,
            inputs=[film, developer],
            outputs=[development_minutes],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, development_minutes, contrast, grain, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
                develop_acc, print_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        unlock_develop_btn.click(
            fn=unlock_develop,
            inputs=[state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn,
                paper, print_exposure, print_grade, print_contrast, unlock_print_btn,
                develop_acc, print_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        print_btn.click(
            fn=commit_print,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, state,
            ],
        )

        unlock_print_btn.click(
            fn=unlock_print,
            inputs=[state],
            outputs=[
                live_out, status, history,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, print_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                sample, file_in, ingest_btn,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn,
                ingest_acc, develop_acc, print_acc,
                rotate_ccw_btn, rotate_180_btn, rotate_cw_btn, state,
            ],
        )

        rotate_outputs = [
            live_out, original_out, latent_out, neg_out, status, history,
            develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
            develop_acc, print_acc, state,
        ]
        rotate_ccw_btn.click(fn=rotate_ccw, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        rotate_cw_btn.click(fn=rotate_cw, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        rotate_180_btn.click(fn=rotate_180, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )

        # Thumbnail / button → enlarge in main preview + open inspect/zoom
        focus_outputs = [live_out, inspect_out, status, inspect_acc, state]
        original_out.select(fn=focus_viewer("original"), inputs=[state], outputs=focus_outputs)
        latent_out.select(fn=focus_viewer("latent"), inputs=[state], outputs=focus_outputs)
        neg_out.select(fn=focus_viewer("negative"), inputs=[state], outputs=focus_outputs)
        view_orig_btn.click(fn=focus_viewer_button("original"), inputs=[state], outputs=focus_outputs)
        view_lat_btn.click(fn=focus_viewer_button("latent"), inputs=[state], outputs=focus_outputs)
        view_neg_btn.click(fn=focus_viewer_button("negative"), inputs=[state], outputs=focus_outputs)
        view_live_btn.click(fn=focus_viewer_button("live"), inputs=[state], outputs=focus_outputs)
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=UI_CSS,
        js=UI_JS,
    )
