#!/usr/bin/env python3
"""Sequential Gradio UI: ingest → develop → print."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from digital_negative.curves import DEVELOPER_STYLES
from digital_negative.pipeline import list_film_profiles, run_darkroom_pipeline
from digital_negative.display import to_pil_gray


FILM_CHOICES = []
for path in list_film_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    FILM_CHOICES.append((f"{data['name']} (ISO {data['iso']})", data["id"]))

DEVELOPER_CHOICES = [(v["name"], k) for k, v in DEVELOPER_STYLES.items()]


def _resolve_input(file_obj) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None) or str(file_obj)


def process(
    file_obj,
    film_id,
    developer_id,
    relative_time,
    contrast,
    grain,
    print_exposure,
    print_grade,
    print_contrast,
):
    input_path = _resolve_input(file_obj)
    artifacts = run_darkroom_pipeline(
        input_path=input_path,
        film_id=film_id,
        developer_id=developer_id,
        relative_time=float(relative_time),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        print_exposure=float(print_exposure),
        print_grade=float(print_grade),
        print_contrast=float(print_contrast),
        do_print=True,
        output_dir=ROOT / "output",
    )
    developed = to_pil_gray(artifacts.development.positive_preview)
    printed = to_pil_gray(artifacts.print_result.preview)
    summary = (
        f"**{artifacts.stats['film']}** · {artifacts.stats['developer']} · "
        f"density mean {artifacts.stats['density_mean']:.2f}\n\n"
        f"Saved comparison: `{artifacts.comparison}`"
    )
    return developed, printed, str(artifacts.comparison), summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        gr.Markdown(
            """
            # Digital Negative Darkroom
            Ingest a camera raw (or leave empty for a synthetic scene), choose film and
            development, then print with enlarger-style controls.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Camera raw or image (optional)",
                    file_types=[".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".tif", ".tiff", ".jpg", ".jpeg", ".png"],
                )
                gr.Markdown("### Development")
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
                relative_time = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Relative development")
                contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast")
                grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain strength")

                gr.Markdown("### Print")
                print_exposure = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Exposure (stops)")
                print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="Multigrade filtration")
                print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Print contrast nudge")
                run_btn = gr.Button("Develop & Print", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    developed_out = gr.Image(label="Developed positive", type="pil")
                    printed_out = gr.Image(label="Print", type="pil")
                comparison_out = gr.Image(label="Linear DN vs final", type="filepath")
                summary = gr.Markdown()

        run_btn.click(
            fn=process,
            inputs=[
                file_in,
                film,
                developer,
                relative_time,
                contrast,
                grain,
                print_exposure,
                print_grade,
                print_contrast,
            ],
            outputs=[developed_out, printed_out, comparison_out, summary],
        )
        demo.load(
            fn=process,
            inputs=[
                file_in,
                film,
                developer,
                relative_time,
                contrast,
                grain,
                print_exposure,
                print_grade,
                print_contrast,
            ],
            outputs=[developed_out, printed_out, comparison_out, summary],
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
