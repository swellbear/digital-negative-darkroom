#!/usr/bin/env python3
"""Sequential Gradio UI: Ingest → Develop → Print with stage locks + history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from digital_negative.curves import DEVELOPER_STYLES, load_film_profile
from digital_negative.development import develop
from digital_negative.display import negative_lightbox_preview, to_pil_gray
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import print_negative


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


def _history_md(dn) -> str:
    hist = dn.metadata.get("history", [])
    if not hist:
        return "_No decisions committed yet._"
    lines = ["### Decision history"]
    for i, h in enumerate(hist, 1):
        op = h.get("op", "?")
        if op == "develop":
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
            lines.append(f"{i}. **{op}** — {h}")
    stages = dn.metadata.get("ui_state", {}).get("committed_stages", [])
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lines.append("")
    lines.append(f"**Committed:** {', '.join(stages) or '—'}  ")
    lines.append(f"**Locked:** {', '.join(locks) or '—'}  ")
    lines.append(f"**Process seed:** `{dn.metadata.get('process_seed')}`")
    return "\n".join(lines)


def _stage_banner(stage: str) -> str:
    steps = [("ingest", "1 Ingest"), ("development", "2 Develop"), ("print", "3 Print")]
    parts = []
    order = {"ingest": 0, "development": 1, "print": 2}
    cur = order.get(stage, -1)
    for i, (key, label) in enumerate(steps):
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

    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    ingest = dn.metadata.get("ingest", {})
    summary = (
        f"{_stage_banner('ingest')}\n\n"
        f"**Ingest locked.** Digital Negative created.  \n"
        f"Source: `{dn.metadata['source']['original_filename']}`  \n"
        f"`{ingest.get('working_space')}` · `{ingest.get('encoding')}`  \n\n"
        f"{_history_md(dn)}"
    )
    state = {"dn": dn, "development": None, "stage": "ingest"}
    return (
        latent,
        None,
        None,
        summary,
        gr.update(interactive=False),  # sample
        gr.update(interactive=False),  # file
        gr.update(interactive=False),  # ingest btn
        gr.update(interactive=True),  # develop btn
        gr.update(interactive=False),  # print btn
        state,
    )


def commit_develop(film_id, developer_id, relative_time, contrast, grain, state):
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
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
    )
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "development" not in locks:
        locks.append("development")

    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    negative = to_pil_gray(negative_lightbox_preview(development.transmittance))
    summary = (
        f"{_stage_banner('development')}\n\n"
        f"**Develop locked.** Negative is on the lightbox.  \n"
        f"{profile.name} · {developer_id} · rel={float(relative_time):.2f} · "
        f"contrast={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
        f"Density mean {float(development.density.mean()):.2f}  \n\n"
        f"{_history_md(dn)}"
    )
    state = {"dn": dn, "development": development, "stage": "development"}
    return (
        latent,
        negative,
        None,
        summary,
        gr.update(interactive=False),  # film
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),  # develop btn
        gr.update(interactive=True),  # print btn
        state,
    )


def commit_print(paper_id, print_exposure, print_grade, print_contrast, state):
    if not state or state.get("development") is None:
        raise gr.Error("Commit Develop first.")
    dn = state["dn"]
    development = state["development"]
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    result = print_negative(
        development.transmittance,
        dn,
        paper,
        overall_exposure=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
    )
    stages = dn.metadata["ui_state"].setdefault("committed_stages", [])
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "print" not in stages:
        stages.append("print")
    if "print" not in locks:
        locks.append("print")

    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    negative = to_pil_gray(negative_lightbox_preview(development.transmittance))
    printed = to_pil_gray(result.preview)
    speed = result and dn.metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
    summary = (
        f"{_stage_banner('print')}\n\n"
        f"**Print locked.**  \n"
        f"{paper.name} · grade {float(print_grade):.1f} · "
        f"exp {float(print_exposure):+.2f} stops · filter speed ×{float(speed):.2f}  \n\n"
        f"{_history_md(dn)}"
    )
    state = {"dn": dn, "development": development, "print": result, "stage": "print"}
    return (
        latent,
        negative,
        printed,
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),  # print btn locked after commit
        state,
    )


def reset_session():
    """Start a new negative — ritual begins again."""
    summary = (
        "1 Ingest · 2 Develop · 3 Print\n\n"
        "*New session. Commit Ingest to create a Digital Negative.*"
    )
    on = gr.update(interactive=True)
    off = gr.update(interactive=False)
    return (
        None,
        None,
        None,
        summary,
        on,  # sample
        on,  # file
        on,  # ingest btn
        on,  # film
        on,  # developer
        on,  # relative
        on,  # contrast
        on,  # grain
        off,  # develop btn
        on,  # paper
        on,  # exp
        on,  # grade
        on,  # print contrast
        off,  # print btn
        None,  # state
    )


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        gr.Markdown(
            """
            # Digital Negative Darkroom
            Work in order. Each **Commit** locks that stage — decisions stay visible in the history.
            Start a **New negative** to begin again. Mild process variation is seed-controlled.
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
                    develop_btn = gr.Button("Commit Develop", interactive=False)

                with gr.Accordion("3 · Print", open=True):
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
                    latent_out = gr.Image(label="Digital Negative (latent)", type="pil")
                    negative_out = gr.Image(label="Developed negative (lightbox)", type="pil")
                    printed_out = gr.Image(label="Print", type="pil")
                summary = gr.Markdown("*Commit Ingest to begin.*")

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                latent_out, negative_out, printed_out, summary,
                sample, file_in, ingest_btn, develop_btn, print_btn, state,
            ],
        )
        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, relative_time, contrast, grain, state],
            outputs=[
                latent_out, negative_out, printed_out, summary,
                film, developer, relative_time, contrast, grain, develop_btn, print_btn, state,
            ],
        )
        print_btn.click(
            fn=commit_print,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[
                latent_out, negative_out, printed_out, summary,
                paper, print_exposure, print_grade, print_contrast, print_btn, state,
            ],
        )
        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                latent_out, negative_out, printed_out, summary,
                sample, file_in, ingest_btn,
                film, developer, relative_time, contrast, grain, develop_btn,
                paper, print_exposure, print_grade, print_contrast, print_btn,
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
