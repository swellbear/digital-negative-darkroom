#!/usr/bin/env python3
"""Sequential Gradio UI: Ingest → Develop → Print (ritual stages)."""

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


def _resolve_input(file_obj, sample_path: str | None) -> str | None:
    if sample_path:
        return sample_path
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None) or str(file_obj)


def commit_ingest(sample_path, file_obj, state):
    """Stage 1: create the Digital Negative (latent)."""
    path = _resolve_input(file_obj, sample_path)
    dn = ingest_path(path or None)
    stages = dn.metadata.setdefault("ui_state", {}).setdefault("committed_stages", [])
    if "ingest" not in stages:
        stages.append("ingest")
    dn.metadata["ui_state"]["current_stage"] = "ingest"

    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    ingest = dn.metadata.get("ingest", {})
    summary = (
        f"**Committed: Ingest**  \n"
        f"Source: `{dn.metadata['source']['original_filename']}`  \n"
        f"Working space: `{ingest.get('working_space', '?')}` · "
        f"encoding: `{ingest.get('encoding', '?')}`  \n"
        f"{ingest.get('notes', '')}"
    )
    state = {
        "dn": dn,
        "development": None,
        "stage": "ingest",
    }
    return (
        latent,
        None,
        None,
        summary,
        gr.update(interactive=True),  # develop btn
        gr.update(interactive=False),  # print btn
        state,
    )


def commit_develop(
    film_id,
    developer_id,
    relative_time,
    contrast,
    grain,
    state,
):
    """Stage 2: develop the committed Digital Negative."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first — there is no Digital Negative yet.")

    dn = state["dn"]
    profile = load_film_profile(
        next(p for p in list_film_profiles() if json.loads(p.read_text())["id"] == film_id)
    )
    development = develop(
        dn,
        profile,
        relative_time=float(relative_time),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
    )
    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    negative = to_pil_gray(negative_lightbox_preview(development.transmittance))
    summary = (
        f"**Committed: Develop**  \n"
        f"{profile.name} · {developer_id} · rel={float(relative_time):.2f} · "
        f"contrast={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
        f"Density mean {float(development.density.mean()):.2f} "
        f"(min {float(development.density.min()):.2f} / max {float(development.density.max()):.2f})"
    )
    state = {
        "dn": dn,
        "development": development,
        "stage": "development",
    }
    return (
        latent,
        negative,
        None,
        summary,
        gr.update(interactive=True),
        state,
    )


def commit_print(paper_id, print_exposure, print_grade, print_contrast, state):
    """Stage 3: print through the developed negative."""
    if not state or state.get("development") is None:
        raise gr.Error("Commit Develop first — there is no negative to print.")

    dn = state["dn"]
    development = state["development"]
    paper = load_paper_profile(
        next(p for p in list_paper_profiles() if json.loads(p.read_text())["id"] == paper_id)
    )
    result = print_negative(
        development.transmittance,
        dn,
        paper,
        overall_exposure=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
    )
    if "print" not in dn.metadata["ui_state"]["committed_stages"]:
        dn.metadata["ui_state"]["committed_stages"].append("print")

    latent = to_pil_gray(dn.to_luminance(), assume_linear=True)
    negative = to_pil_gray(negative_lightbox_preview(development.transmittance))
    printed = to_pil_gray(result.preview)
    summary = (
        f"**Committed: Print**  \n"
        f"{paper.name} · grade {float(print_grade):.1f} · "
        f"exposure {float(print_exposure):+.2f} stops  \n"
        f"Stages: {', '.join(dn.metadata['ui_state']['committed_stages'])}"
    )
    state = {
        "dn": dn,
        "development": development,
        "print": result,
        "stage": "print",
    }
    return latent, negative, printed, summary, state


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        gr.Markdown(
            """
            # Digital Negative Darkroom
            Work in order: **Ingest → Develop → Print**.  
            Each stage is committed before the next unlocks — closer to a real darkroom sequence.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("1 · Ingest (Digital Negative)", open=True):
                    sample = gr.Dropdown(
                        choices=SAMPLE_CHOICES,
                        value=default_sample,
                        label="Built-in sample raws",
                    )
                    file_in = gr.File(
                        label="Or upload from your computer",
                        file_types=[
                            ".arw",
                            ".cr2",
                            ".cr3",
                            ".nef",
                            ".dng",
                            ".raf",
                            ".orf",
                            ".rw2",
                            ".tif",
                            ".tiff",
                            ".jpg",
                            ".jpeg",
                            ".png",
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
                        choices=DEVELOPER_CHOICES,
                        value="standard",
                        label="Developer style",
                    )
                    relative_time = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.05, label="Relative development"
                    )
                    contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast")
                    grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain strength")
                    develop_btn = gr.Button("Commit Develop", interactive=False)

                with gr.Accordion("3 · Print", open=True):
                    paper = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value=PAPER_CHOICES[0][1] if PAPER_CHOICES else None,
                        label="Paper",
                    )
                    print_exposure = gr.Slider(
                        -2.0, 2.0, value=0.0, step=0.05, label="Exposure (stops)"
                    )
                    print_grade = gr.Slider(
                        0.0, 5.0, value=2.5, step=0.5, label="Multigrade filtration"
                    )
                    print_contrast = gr.Slider(
                        -1.0, 1.0, value=0.0, step=0.05, label="Print contrast nudge"
                    )
                    print_btn = gr.Button("Commit Print", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    latent_out = gr.Image(label="Digital Negative (latent)", type="pil")
                    negative_out = gr.Image(label="Developed negative (lightbox)", type="pil")
                    printed_out = gr.Image(label="Print", type="pil")
                summary = gr.Markdown("*Start by committing Ingest.*")

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                latent_out,
                negative_out,
                printed_out,
                summary,
                develop_btn,
                print_btn,
                state,
            ],
        )
        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, relative_time, contrast, grain, state],
            outputs=[
                latent_out,
                negative_out,
                printed_out,
                summary,
                print_btn,
                state,
            ],
        )
        print_btn.click(
            fn=commit_print,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[latent_out, negative_out, printed_out, summary, state],
        )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link (useful from cloud agents)",
    )
    args = parser.parse_args()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
