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
import numpy as np

from digital_negative.curves import DEVELOPER_STYLES, load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative
from digital_negative.display import (
    linear_to_srgb,
    negative_lightbox_preview,
    original_photo_preview,
    to_u8_gray,
)
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import print_negative

# Match commit look as closely as practical while staying interactive.
LIVE_MAX_SIDE = 2000
DRAG_MAX_SIDE = 1280  # high enough for critical judgment while dragging
REF_MAX_SIDE = 420

FILM_CHOICES = []
for path in list_film_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    FILM_CHOICES.append((f"{data['name']} (ISO {data['iso']})", data["id"]))

PAPER_CHOICES = []
for path in list_paper_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    PAPER_CHOICES.append((data["name"], data["id"]))

DEVELOPER_CHOICES = [(v["name"], k) for k, v in DEVELOPER_STYLES.items()]

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
  flex: 0 0 320px !important;
  width: 320px !important;
  max-width: 320px !important;
  position: sticky !important;
  top: 6px !important;
  max-height: calc(100vh - 16px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  padding-right: 6px !important;
  align-self: flex-start !important;
}
#preview_col {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}
/* Compact control density */
#controls_col .block {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
  padding: 0 !important;
}
#controls_col .label-wrap,
#controls_col label,
#controls_col .svelte-1b6s6s {
  margin-bottom: 0 !important;
  font-size: 0.82rem !important;
}
#controls_col .form {
  gap: 4px !important;
}
#controls_col button {
  min-height: 34px !important;
  font-size: 0.9rem !important;
}
#controls_col .accordion {
  margin-bottom: 4px !important;
}
#controls_col .accordion > .label-wrap {
  padding: 6px 8px !important;
  font-size: 0.9rem !important;
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
#live_preview img,
#live_preview .image-container img,
#live_preview .image-frame img {
  max-height: calc(100vh - 160px) !important;
  width: 100% !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
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


def _profile_path(paths, profile_id: str) -> Path:
    for p in paths:
        if json.loads(p.read_text(encoding="utf-8"))["id"] == profile_id:
            return p
    raise FileNotFoundError(profile_id)


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
            lines.append(
                f"{i}. **Develop** — `{h.get('film_profile_id')}` · {h.get('developer_id')} · "
                f"rel={h.get('relative_time')} · N±={h.get('contrast_modifier')} · "
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


def _pack_preview(live, original, latent, neg, summary, state):
    status, hist = _split_summary(summary or "")
    return live, original, latent, neg, status, hist, state


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
    latent_ref = _downscale_rgb(latent_full, REF_MAX_SIDE)
    original_full = original_photo_preview(path, dn_image=dn.image)
    original_ref = _downscale_rgb(original_full, REF_MAX_SIDE)
    summary = (
        f"{_stage_banner('development', ['ingest'])}\n\n"
        f"**Ingest locked** — `{dn.metadata['source']['original_filename']}`  \n"
        f"_Develop controls below — Commit Develop when ready._\n\n"
        f"{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "original_ref": original_ref,
        "latent_ref": latent_ref,
        "neg_ref": None,
        "live_rgb": _downscale_rgb(latent_full, LIVE_MAX_SIDE),
        "development": None,
        "development_full": None,
        "stage": "development",
        "summary_cache": summary,
        "source_path": path,
    }
    return (
        state["live_rgb"],
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
        state,
    )


def _run_live_develop_then_print(
    film_id,
    developer_id,
    relative_time,
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
        relative_time=float(relative_time),
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
    neg_ref = _downscale_rgb(
        _to_rgb_u8(negative_lightbox_preview(development.transmittance)), REF_MAX_SIDE
    )
    speed = state["dn"].metadata.get("print", {}).get("filtration", {}).get("values", {}).get(
        "filter_speed", 1.0
    )
    quality_note = "drag" if max_side <= DRAG_MAX_SIDE else "hq"
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Live print** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note})  \n"
        f"{profile.name} · {developer_id} · rel={float(relative_time):.2f} · "
        f"N±={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
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
        "neg_ref": neg_ref,
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
    relative_time,
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
        relative_time,
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
    relative_time,
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
        relative_time,
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
    relative_time,
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
        relative_time,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        quality="high",
    )


def commit_develop(film_id, developer_id, relative_time, contrast, grain, state):
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    if _locked(state, "development"):
        raise gr.Error("Develop is already locked.")

    dn = state["dn"]
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        dn,
        profile,
        relative_time=float(relative_time),
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
    neg_ref = _downscale_rgb(
        _to_rgb_u8(negative_lightbox_preview(development.transmittance)), REF_MAX_SIDE
    )
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Develop locked** — refine Print below, then Commit Print.\n\n{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": state.get("proxy"),
        "proxy_drag": state.get("proxy_drag"),
        "original_ref": state.get("original_ref"),
        "latent_ref": state.get("latent_ref"),
        "neg_ref": neg_ref,
        "live_rgb": live_view,
        "development": development,
        "development_full": development,
        "transmittance_proxy": t_proxy,
        "stage": "print",
        "summary_cache": summary,
        "source_path": state.get("source_path"),
    }
    return (
        live_view,
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
                        choices=DEVELOPER_CHOICES, value="standard", label="Developer"
                    )
                    relative_time = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Rel. development")
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
                    label="Commit preview (live)",
                    type="numpy",
                    elem_id="live_preview",
                    height=620,
                )
                with gr.Row(elem_id="ref_row"):
                    original_out = gr.Image(label="Original", type="numpy", height=96)
                    latent_out = gr.Image(label="Latent DN", type="numpy", height=96)
                    neg_out = gr.Image(label="Negative", type="numpy", height=96)

        # Always pass develop + print controls so the large viewer can show a
        # theoretical print through the working negative while developing.
        preview_inputs = [
            film,
            developer,
            relative_time,
            contrast,
            grain,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            state,
        ]
        preview_outputs = [live_out, original_out, latent_out, neg_out, status, history, state]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                sample, file_in, ingest_btn, develop_btn, print_btn,
                ingest_acc, develop_acc, print_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        for ctrl in (
            film,
            developer,
            relative_time,
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

        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, relative_time, contrast, grain, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, relative_time, contrast, grain,
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
                film, developer, relative_time, contrast, grain,
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
                film, developer, relative_time, contrast, grain,
                develop_btn, unlock_develop_btn,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn,
                ingest_acc, develop_acc, print_acc, state,
            ],
        )
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
    )
