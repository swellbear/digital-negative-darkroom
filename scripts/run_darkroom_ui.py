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
DRAG_MAX_SIDE = 960  # fast proxy while dragging sliders
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
.gradio-container { max-width: 1700px !important; }
#controls_col {
  max-height: 92vh;
  overflow-y: auto;
  padding-right: 8px;
}
#live_preview {
  min-height: 72vh;
}
#live_preview img,
#live_preview .image-container img {
  max-height: 72vh !important;
  width: 100% !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
}
#ref_row img {
  max-height: 140px !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
}
#ref_row {
  gap: 8px;
}
"""


def _profile_path(paths, profile_id: str) -> Path:
    for p in paths:
        if json.loads(p.read_text(encoding="utf-8"))["id"] == profile_id:
            return p
    raise FileNotFoundError(profile_id)


def _resolve_input(file_obj, sample_path: str | None) -> str | None:
    if sample_path:
        return sample_path
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None) or str(file_obj)


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
    lines = ["### Decision history"]
    if not hist:
        lines.append("_No locked decisions yet — explore freely, then Commit to record._")
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
            lines.append(f"{i}. **Unlocked {label}** — revise and re-commit")
        else:
            lines.append(f"{i}. **{op}**")
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lock_labels = []
    for s in ("ingest", "development", "print"):
        if s in locks:
            lock_labels.append({"ingest": "Ingest", "development": "Develop", "print": "Print"}[s])
    lines.append(
        f"\n**Locked:** {', '.join(lock_labels) or '—'} · "
        f"**process seed:** `{dn.metadata.get('process_seed')}`"
    )
    return "\n".join(lines)


def _stage_banner(stage: str, locked: list | None = None) -> str:
    """Ritual progress with lock markers for committed stages."""
    steps = [("ingest", "1 Ingest"), ("development", "2 Develop"), ("print", "3 Print")]
    order = {"ingest": 0, "development": 1, "print": 2}
    cur = order.get(stage, -1)
    locked_set = set(locked or [])
    parts = []
    for i, (key, label) in enumerate(steps):
        done = key in locked_set
        tag = f"{label}·locked" if done else label
        if i < cur:
            parts.append(f"✓ {tag}")
        elif i == cur:
            parts.append(f"**→ {tag}**")
        else:
            parts.append(tag)
    return " · ".join(parts)


def _locks(state) -> list:
    if not state or state.get("dn") is None:
        return []
    return list(state["dn"].metadata.get("ui_state", {}).get("locked_stages", []))


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
        f"**Ingest locked.** Large image = theoretical print through the working negative.  \n"
        f"References below: **Original** (camera/display) → **Latent DN** → developed negative.  \n"
        f"`{dn.metadata['source']['original_filename']}`  \n\n{_history_md(dn)}"
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
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
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
    quality_note = "fast drag" if max_side <= DRAG_MAX_SIDE else "commit-quality"
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Print preview through working negative** "
        f"({live_rgb.shape[1]}×{live_rgb.shape[0]}, {quality_note}) — develop changes "
        f"shown as a print with current Print settings.  \n"
        f"Develop: {profile.name} · {developer_id} · rel={float(relative_time):.2f} · "
        f"contrast={float(contrast):+.2f} · grain={float(grain):.2f} · "
        f"density μ={float(development.density.mean()):.2f}  \n"
        f"Print: {paper.name} · grade {float(print_grade):.1f} · "
        f"exp {float(print_exposure):+.2f} · filter ×{float(speed):.2f}  \n"
        f"_Commit Develop locks the negative; then refine Print and Commit Print._  \n\n"
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
        return None, None, None, None, "*Commit Ingest first.*", state

    if _locked(state, "print"):
        return (
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
                return (
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
        quality_note = "fast drag" if quality == "drag" else "commit-quality"
        summary = (
            f"{_stage_banner('print', _locks(state))}\n\n"
            f"**Commit Print preview** ({live_rgb.shape[1]}×{live_rgb.shape[0]}, {quality_note}) — "
            f"this is what Commit Print will lock.  \n"
            f"{paper.name} · grade {float(print_grade):.1f} · exp {float(print_exposure):+.2f} · "
            f"filter ×{float(speed):.2f}  \n\n{_history_md(state['dn'])}"
        )
        state = {**state, "print_draft": result, "live_rgb": live_rgb, "summary_cache": summary}
        return (
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
    return live_rgb, state.get("original_ref"), state.get("latent_ref"), neg_ref, summary, state


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
        f"**Develop locked.** Large image is now a **Commit Print preview** — "
        f"refine Print controls, then Commit Print.  \n\n{_history_md(dn)}"
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
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
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
        f"**Print locked** — {paper.name} · grade {float(print_grade):.1f} · "
        f"exp {float(print_exposure):+.2f} · filter ×{float(speed):.2f}  \n"
        f"Use Unlock Print (or Unlock Develop) to revise.  \n\n{_history_md(dn)}"
    )
    state = {**state, "print": result, "live_rgb": live_rgb, "summary_cache": summary}
    return (
        live_rgb,
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        summary,
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
        f"**Develop unlocked.** Large image again shows a theoretical print through "
        f"the working negative (current Print settings). Move sliders freely, then "
        f"Commit Develop again when ready.  \n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        summary,
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
        f"**Print unlocked.** Large image is a Commit Print preview again — "
        f"adjust exposure/grade, then Commit Print.  \n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        summary,
        on,
        on,
        on,
        on,
        on,
        off,
        state,
    )


def reset_session():
    summary = (
        "*Commit Ingest. While developing, the large image shows a theoretical print "
        "using current Print settings. References show Original → Latent DN → negative.*"
    )
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        None,
        None,
        None,
        None,
        summary,
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
        None,
    )


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        gr.Markdown(
            """
            # Digital Negative Darkroom
            **Large image = theoretical print** while you develop (current Print settings), then **Commit Print preview** after Develop is locked.  
            References under the viewer: **Original photo → Latent DN → Developed negative**.  
            Sliders update live (fast while dragging, higher quality on release). **Commit** locks · **Unlock** revises.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_id="controls_col", min_width=320):
                with gr.Accordion("1 · Ingest", open=True):
                    sample = gr.Dropdown(choices=SAMPLE_CHOICES, value=default_sample, label="Sample raw")
                    file_in = gr.File(
                        label="Or upload",
                        file_types=[
                            ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2",
                            ".tif", ".tiff", ".jpg", ".jpeg", ".png",
                        ],
                    )
                    ingest_btn = gr.Button("Commit Ingest", variant="primary")

                with gr.Accordion("2 · Develop", open=True):
                    film = gr.Dropdown(
                        choices=FILM_CHOICES,
                        value=FILM_CHOICES[0][1] if FILM_CHOICES else None,
                        label="Film stock",
                    )
                    developer = gr.Dropdown(
                        choices=DEVELOPER_CHOICES, value="standard", label="Developer style"
                    )
                    relative_time = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Relative development")
                    contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast (N− / N+)")
                    grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain strength")
                    with gr.Row():
                        develop_btn = gr.Button("Commit Develop", interactive=False, variant="primary")
                        unlock_develop_btn = gr.Button("Unlock Develop", interactive=False)

                with gr.Accordion("3 · Print", open=True):
                    paper = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value=PAPER_CHOICES[0][1] if PAPER_CHOICES else None,
                        label="Paper",
                    )
                    print_exposure = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Exposure (stops)")
                    print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="Multigrade filtration")
                    print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Between-filter nudge")
                    with gr.Row():
                        print_btn = gr.Button("Commit Print", interactive=False, variant="primary")
                        unlock_print_btn = gr.Button("Unlock Print", interactive=False)

                reset_btn = gr.Button("New negative")
                summary = gr.Markdown(
                    "*Commit Ingest. Develop sliders update a theoretical print "
                    "(current Print settings) in the large viewer.*"
                )

            with gr.Column(scale=4, min_width=900):
                live_out = gr.Image(
                    label="Commit preview (live) — theoretical print / what locking will look like",
                    type="numpy",
                    elem_id="live_preview",
                )
                with gr.Row(elem_id="ref_row"):
                    original_out = gr.Image(
                        label="Original photo (reference)", type="numpy", height=140
                    )
                    latent_out = gr.Image(label="Latent DN (reference)", type="numpy", height=140)
                    neg_out = gr.Image(label="Developed negative (reference)", type="numpy", height=140)

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
        preview_outputs = [live_out, original_out, latent_out, neg_out, summary, state]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, summary,
                sample, file_in, ingest_btn, develop_btn, print_btn, state,
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
                live_out, original_out, latent_out, neg_out, summary,
                film, developer, relative_time, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn, unlock_print_btn, state,
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
                live_out, original_out, latent_out, neg_out, summary,
                film, developer, relative_time, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn,
                paper, print_exposure, print_grade, print_contrast, unlock_print_btn, state,
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
                live_out, original_out, latent_out, neg_out, summary,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, state,
            ],
        )

        unlock_print_btn.click(
            fn=unlock_print,
            inputs=[state],
            outputs=[
                live_out, summary,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, state,
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
                live_out, original_out, latent_out, neg_out, summary,
                sample, file_in, ingest_btn,
                film, developer, relative_time, contrast, grain,
                develop_btn, unlock_develop_btn,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, state,
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
