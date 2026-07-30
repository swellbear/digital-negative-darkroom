#!/usr/bin/env python3
"""Sequential Gradio UI with live Develop/Print picture previews + Commit locks."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np
from PIL import Image

from digital_negative.curves import DEVELOPER_STYLES, load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative
from digital_negative.display import linear_to_srgb, negative_lightbox_preview, to_u8_gray
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import print_negative

LIVE_MAX_SIDE = 720

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
    """HWC uint8 RGB for reliable Gradio Image updates."""
    view = linear_to_srgb(gray_float) if assume_linear else gray_float
    g = to_u8_gray(view)
    return np.stack([g, g, g], axis=-1)


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
        lines.append("_No locked decisions yet — watch the big preview while you adjust, then Commit._")
    for i, h in enumerate(hist, 1):
        op = h.get("op", "?")
        if op == "ingest":
            lines.append(f"{i}. **Ingest** — `{h.get('source')}` · {h.get('working_space')}")
        elif op == "develop":
            lines.append(
                f"{i}. **Develop** — {h.get('film_profile_id')} · {h.get('developer_id')} · "
                f"rel={h.get('relative_time')} · contrast={h.get('contrast_modifier')}"
            )
        elif op == "print":
            lines.append(
                f"{i}. **Print** — {h.get('paper_id')} · grade {h.get('grade')} · "
                f"exp {h.get('overall_exposure'):+g} stops"
            )
        else:
            lines.append(f"{i}. **{op}**")
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lines.append("")
    lines.append(f"**Locked:** {', '.join(locks) or '—'}  ")
    lines.append(f"**Process seed:** `{dn.metadata.get('process_seed')}`")
    return "\n".join(lines)


def _stage_banner(stage: str) -> str:
    steps = [("ingest", "1 Ingest"), ("development", "2 Develop"), ("print", "3 Print")]
    order = {"ingest": 0, "development": 1, "print": 2}
    cur = order.get(stage, -1)
    parts = []
    for i, (_, label) in enumerate(steps):
        if i < cur:
            parts.append(f"✓ {label}")
        elif i == cur:
            parts.append(f"**→ {label}**")
        else:
            parts.append(label)
    return " · ".join(parts)


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

    latent = _to_rgb_u8(dn.to_luminance(), assume_linear=True)
    summary = (
        f"{_stage_banner('development')}\n\n"
        f"**Ingest locked.** Move Develop sliders — the **large preview** is the live picture.  \n"
        f"Source: `{dn.metadata['source']['original_filename']}`  \n\n"
        f"{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": _proxy_dn(dn),
        "latent_rgb": latent,
        "development": None,
        "development_full": None,
        "stage": "development",
    }
    # Placeholder live preview until first develop pass
    return (
        latent,  # reference
        latent,  # main live (starts as latent; live_develop replaces immediately)
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        state,
    )


def _run_live_develop(film_id, developer_id, relative_time, contrast, grain, state):
    proxy = state.get("proxy") or _proxy_dn(state["dn"])
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        proxy,
        profile,
        relative_time=float(relative_time),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
        process_variation=0.25,
        commit=False,
    )
    # Main picture: contact-style positive so slider changes read like the photo
    # (lightbox negative alone can look subtle). Also keep a negative strip feel
    # by packing positive as the hero image.
    positive = development.positive_preview
    live_rgb = _to_rgb_u8(positive)
    neg_rgb = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    # Side-by-side composite: negative | positive so both read clearly
    gap = np.full((live_rgb.shape[0], 12, 3), 30, dtype=np.uint8)
    # Match heights
    if neg_rgb.shape[0] != live_rgb.shape[0] or neg_rgb.shape[1] != live_rgb.shape[1]:
        neg_img = Image.fromarray(neg_rgb).resize(
            (live_rgb.shape[1], live_rgb.shape[0]), Image.Resampling.BILINEAR
        )
        neg_rgb = np.asarray(neg_img)
    composite = np.concatenate([neg_rgb, gap, live_rgb], axis=1)

    summary = (
        f"{_stage_banner('development')}\n\n"
        f"**Live picture** updating now — left: negative · right: positive proof  \n"
        f"{profile.name} · {developer_id} · rel={float(relative_time):.2f} · "
        f"contrast={float(contrast):+.2f} · grain={float(grain):.2f} · "
        f"density μ={float(development.density.mean()):.2f}  \n"
        f"_Commit Develop when it looks right._  \n\n"
        f"{_history_md(state['dn'])}"
    )
    state = {
        **state,
        "proxy": proxy,
        "development": development,
        "stage": "development",
        "live_rgb": composite,
    }
    return composite, summary, state


def live_develop(film_id, developer_id, relative_time, contrast, grain, state):
    if not state or state.get("dn") is None:
        return None, None, "*Commit Ingest first.*", state
    if _locked(state, "development"):
        return (
            state.get("latent_rgb"),
            state.get("live_rgb"),
            state.get("summary_cache", ""),
            state,
        )
    live_rgb, summary, state = _run_live_develop(
        film_id, developer_id, relative_time, contrast, grain, state
    )
    state["summary_cache"] = summary
    # reference latent stays; main live picture changes
    return state.get("latent_rgb"), live_rgb, summary, state


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
    pos_proxy = np.ascontiguousarray(development.positive_preview[::step, ::step])

    live_rgb = _to_rgb_u8(development.positive_preview)
    neg_rgb = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    gap = np.full((live_rgb.shape[0], 12, 3), 30, dtype=np.uint8)
    neg_rgb = np.asarray(
        Image.fromarray(neg_rgb).resize((live_rgb.shape[1], live_rgb.shape[0]), Image.Resampling.BILINEAR)
    )
    composite = np.concatenate([neg_rgb, gap, live_rgb], axis=1)

    summary = (
        f"{_stage_banner('print')}\n\n"
        f"**Develop locked.** Move Print sliders — the large preview becomes the print.  \n"
        f"{profile.name} · density μ={float(development.density.mean()):.2f}  \n\n"
        f"{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": state.get("proxy"),
        "latent_rgb": state.get("latent_rgb"),
        "development": development,
        "development_full": development,
        "transmittance_proxy": t_proxy,
        "positive_proxy": pos_proxy,
        "live_rgb": composite,
        "stage": "print",
        "summary_cache": summary,
    }
    return (
        state["latent_rgb"],
        composite,
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        state,
    )


def live_print(paper_id, print_exposure, print_grade, print_contrast, state):
    if not state or not _locked(state, "development"):
        return state.get("live_rgb") if state else None, (
            state.get("summary_cache", "*Commit Develop before printing.*") if state else "*Commit Develop first.*"
        ), state
    if _locked(state, "print"):
        return state.get("live_rgb"), state.get("summary_cache", ""), state

    t = state.get("transmittance_proxy")
    if t is None:
        t = state["development_full"].transmittance
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
    summary = (
        f"{_stage_banner('print')}\n\n"
        f"**Live print picture** updating now — {paper.name} · "
        f"grade {float(print_grade):.1f} · exp {float(print_exposure):+.2f} stops · "
        f"filter ×{float(speed):.2f} · preview μ={float(live_rgb.mean()):.1f}  \n"
        f"_Commit Print to finalize._  \n\n"
        f"{_history_md(state['dn'])}"
    )
    state = {**state, "print_draft": result, "live_rgb": live_rgb, "summary_cache": summary, "stage": "print"}
    return live_rgb, summary, state


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

    live_rgb = _to_rgb_u8(result.preview)
    speed = dn.metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
    summary = (
        f"{_stage_banner('print')}\n\n"
        f"**Print locked.**  \n"
        f"{paper.name} · grade {float(print_grade):.1f} · "
        f"exp {float(print_exposure):+.2f} stops · filter ×{float(speed):.2f}  \n\n"
        f"{_history_md(dn)}"
    )
    state = {**state, "print": result, "live_rgb": live_rgb, "summary_cache": summary, "stage": "print"}
    return (
        state.get("latent_rgb"),
        live_rgb,
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        state,
    )


def reset_session():
    summary = (
        "1 Ingest · 2 Develop · 3 Print\n\n"
        "*Commit Ingest, then drag Develop/Print controls — the large image is the live picture.*"
    )
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
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
        on,
        on,
        on,
        on,
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
            The **large image** is the live picture — it should change as you move sliders.  
            **Commit** locks that stage. Small image on the left is the locked latent Digital Negative (reference).
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
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

                with gr.Accordion("2 · Develop (live picture)", open=True):
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
                    develop_btn = gr.Button("Commit Develop", interactive=False)

                with gr.Accordion("3 · Print (live picture)", open=True):
                    paper = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value=PAPER_CHOICES[0][1] if PAPER_CHOICES else None,
                        label="Paper",
                    )
                    print_exposure = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Exposure (stops)")
                    print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="Multigrade filtration")
                    print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Between-filter nudge")
                    print_btn = gr.Button("Commit Print", interactive=False)

                reset_btn = gr.Button("New negative")

            with gr.Column(scale=2):
                with gr.Row():
                    ref_out = gr.Image(
                        label="Latent DN (locked reference)",
                        type="numpy",
                        height=220,
                    )
                    live_out = gr.Image(
                        label="LIVE PICTURE — updates with sliders",
                        type="numpy",
                        height=520,
                    )
                summary = gr.Markdown("*Commit Ingest, then move Develop sliders and watch the large image.*")

        develop_inputs = [film, developer, relative_time, contrast, grain, state]
        print_inputs = [paper, print_exposure, print_grade, print_contrast, state]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                ref_out,
                live_out,
                summary,
                sample,
                file_in,
                ingest_btn,
                develop_btn,
                print_btn,
                state,
            ],
        ).then(
            fn=live_develop,
            inputs=develop_inputs,
            outputs=[ref_out, live_out, summary, state],
        )

        # .input = while dragging; .change = on release — both refresh the picture
        for ctrl in (film, developer, relative_time, contrast, grain):
            ctrl.input(
                fn=live_develop,
                inputs=develop_inputs,
                outputs=[ref_out, live_out, summary, state],
            )
            ctrl.change(
                fn=live_develop,
                inputs=develop_inputs,
                outputs=[ref_out, live_out, summary, state],
            )

        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, relative_time, contrast, grain, state],
            outputs=[
                ref_out,
                live_out,
                summary,
                film,
                developer,
                relative_time,
                contrast,
                grain,
                develop_btn,
                print_btn,
                state,
            ],
        ).then(
            fn=live_print,
            inputs=print_inputs,
            outputs=[live_out, summary, state],
        )

        for ctrl in (paper, print_exposure, print_grade, print_contrast):
            ctrl.input(
                fn=live_print,
                inputs=print_inputs,
                outputs=[live_out, summary, state],
            )
            ctrl.change(
                fn=live_print,
                inputs=print_inputs,
                outputs=[live_out, summary, state],
            )

        print_btn.click(
            fn=commit_print,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[
                ref_out,
                live_out,
                summary,
                paper,
                print_exposure,
                print_grade,
                print_contrast,
                print_btn,
                state,
            ],
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                ref_out,
                live_out,
                summary,
                sample,
                file_in,
                ingest_btn,
                film,
                developer,
                relative_time,
                contrast,
                grain,
                develop_btn,
                paper,
                print_exposure,
                print_grade,
                print_contrast,
                print_btn,
                state,
            ],
        )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
